# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Google Cloud as a backend for the Anthropic plugin."""

import asyncio
import os
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from anthropic import NOT_GIVEN, AsyncAnthropicVertex

from genkit import GenkitError
from genkit_anthropic._backend import Backend, client_kwargs

# Every project that enables Claude gets quota on the global endpoint;
# regional endpoints need regional quota.
DEFAULT_LOCATION = 'global'


@dataclass(frozen=True, kw_only=True)
class GoogleCloud:
    """Serve Claude from Google Cloud, billed to a Google Cloud project.

    This is Claude from Vertex AI Model Garden: enable the model there, then
    pass ``Anthropic(backend=GoogleCloud(...))``. Requests authenticate
    with application default credentials, so there is no API key. Refs,
    ``AnthropicConfig`` and ``.prompt`` files are the same as on the
    Anthropic API.

    Example:
        ```python
        from genkit import Genkit
        from genkit_anthropic import Anthropic, GoogleCloud

        # 1. Bill Claude to your Google Cloud project instead of an Anthropic key
        ai = Genkit(plugins=[Anthropic(backend=GoogleCloud(project='bistro-prod'))])

        # 2. Same typed ref and config as the Anthropic API
        res = await ai.generate(
            model=Anthropic.claude('claude-sonnet-4-6'),
            prompt='Suggest a dish for a guest with a shellfish allergy.',
        )
        print(res.text)
        # => Herb-Roasted Chicken with Lemon Rice

        # 3. On Cloud Run, the project comes from the environment
        Anthropic(backend=GoogleCloud())
        # => project from GOOGLE_CLOUD_PROJECT or the credentials, location 'global'
        ```

    Attributes:
        project: Google Cloud project. Defaults to ``GCLOUD_PROJECT``, then
            ``GOOGLE_CLOUD_PROJECT``, then the credentials' project.
        location: Google Cloud location. Defaults to ``GOOGLE_CLOUD_LOCATION``,
            then ``global``.
    """

    project: str | None = None
    location: str | None = None


def resolve_project(backend: GoogleCloud) -> str | None:
    """The project from the argument or the environment; None means ask the credentials."""
    return backend.project or os.environ.get('GCLOUD_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT') or None


def resolve_location(backend: GoogleCloud) -> str:
    """The location from the argument or the environment, else ``global``."""
    return backend.location or os.environ.get('GOOGLE_CLOUD_LOCATION') or DEFAULT_LOCATION


def require_google_auth() -> None:
    """Fail at setup, on the user's line, when the google-cloud extra isn't installed.

    Raises:
        ImportError: ``google-auth`` isn't installed.
    """
    # Optional dependency: google-auth ships with the [google-cloud] extra only.
    try:
        import google.auth  # noqa: F401, PLC0415
    except ModuleNotFoundError as e:
        raise ImportError(
            "Anthropic(backend=GoogleCloud()) needs the google-cloud extra: uv add 'genkit-anthropic[google-cloud]'"
        ) from e


# The scope AsyncAnthropicVertex asks for when it loads credentials itself.
_SCOPES = ('https://www.googleapis.com/auth/cloud-platform',)
_LOGIN_FIX = (
    'run `gcloud auth application-default login` on your machine, '
    'or give the service account roles/aiplatform.user when running on Google Cloud'
)


def load_adc() -> tuple[Any, str | None]:  # noqa: ANN401 - google.auth credentials are untyped
    """Application default credentials, refreshed, and their project (blocking: may hit the metadata server).

    Refreshing here makes a missing or stale login fail on the first call
    with the command that fixes it. Inside the SDK it surfaces mid-request
    as an unmapped google.auth error, which reaches the caller as INTERNAL.

    Raises:
        GenkitError: FAILED_PRECONDITION when there are no credentials, or
            they need a fresh login.
    """
    # Optional dependency, checked by require_google_auth at setup.
    import google.auth  # noqa: PLC0415
    from google.auth.exceptions import DefaultCredentialsError, RefreshError  # noqa: PLC0415
    from google.auth.transport.requests import Request  # noqa: PLC0415

    try:
        credentials, project = google.auth.default(scopes=list(_SCOPES))
        credentials.refresh(Request())
    except DefaultCredentialsError as e:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=f'Anthropic(backend=GoogleCloud()) found no Google Cloud credentials: {_LOGIN_FIX}',
        ) from e
    except RefreshError as e:
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=f'Anthropic(backend=GoogleCloud()) could not refresh Google Cloud credentials: {_LOGIN_FIX}',
        ) from e
    return credentials, project


# region and project_id come from GoogleCloud(project=..., location=...), so they aren't client options.
_SET_BY_GOOGLE_CLOUD = frozenset({'region', 'project_id'})
# Either one in client_options means the caller brings their own credential, so ADC isn't loaded.
_OWN_CREDENTIAL = frozenset({'access_token', 'credentials'})


class GoogleCloudBackend(Backend[AsyncAnthropicVertex]):
    """Claude on Google Cloud, authenticated with application default credentials."""

    # Vertex doesn't offer every Anthropic API beta, so none are assumed.
    default_betas = ()

    def __init__(
        self,
        config: GoogleCloud,
        *,
        api_key: str | None,
        base_url: str | None,
        client_options: Mapping[str, Any] | None,
    ) -> None:
        """Validate the client settings at setup, on the user's line.

        Raises:
            ValueError: ``api_key`` was passed; Google Cloud doesn't use one.
            ImportError: The ``google-cloud`` extra isn't installed.
            TypeError: An unknown or repeated ``client_options`` key.
        """
        if api_key is not None:
            raise ValueError(
                'Anthropic(backend=GoogleCloud()) authenticates with Google Cloud credentials; drop api_key'
            )
        require_google_auth()
        # Vertex URLs end in /v1, so the Anthropic API's /v1 guard doesn't apply to base_url here.
        self._options = client_kwargs(
            AsyncAnthropicVertex, client_options, reserved=_SET_BY_GOOGLE_CLOUD, base_url=base_url
        )
        # An explicit setting beats the environment; the SDK's own
        # ANTHROPIC_VERTEX_PROJECT_ID / CLOUD_ML_REGION never pick these.
        self._project = resolve_project(config)
        self._location = resolve_location(config)
        self._uses_adc = not (_OWN_CREDENTIAL & self._options.keys())
        self._ready = False

    def _new_client(self) -> AsyncAnthropicVertex:
        # Callers await request_client first, which resolves the project; NOT_GIVEN is only for tests.
        # With ADC, request_client has also put the loaded credentials into the options.
        project_id = self._project if self._project is not None else NOT_GIVEN
        return AsyncAnthropicVertex(region=self._location, project_id=project_id, **self._options)

    async def request_client(self, api_key: str | None) -> AsyncAnthropicVertex:
        """The plugin's client, once its credentials and project are known.

        Raises:
            GenkitError: FAILED_PRECONDITION for a ``context.secrets`` key, no
                usable credentials, or no project.
        """
        if api_key is not None:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    'A per-request API key from context.secrets cannot be used: '
                    'Anthropic(backend=GoogleCloud()) authenticates with Google Cloud credentials.'
                ),
            )
        await self._prepare()
        return self.client()

    async def list_model_ids(self) -> list[str]:
        """Nothing: Google Cloud has no Models API, so the built-in catalog is the whole list."""
        return []

    async def _prepare(self) -> None:
        """Load credentials and find the project, before the first client is built.

        With ADC, the credentials load once and their project fills in when
        none was passed or set, as on Cloud Run. The lookup can hit the
        metadata server, so it runs in a thread. A failure isn't cached, so the
        next call retries after the user logs in.
        """
        if self._ready:
            return
        if self._uses_adc:
            credentials, adc_project = await asyncio.to_thread(load_adc)
            self._options['credentials'] = credentials
            self._project = self._project or adc_project
        if self._project is None:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    'Anthropic(backend=GoogleCloud()) needs a Google Cloud project: pass GoogleCloud(project=...), '
                    'set GOOGLE_CLOUD_PROJECT, or run `gcloud config set project PROJECT_ID`'
                ),
            )
        self._ready = True
