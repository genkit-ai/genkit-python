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

"""Model Garden's OpenAI-compatible models on Google Cloud, as a backend for OpenAICompatible."""

import asyncio
import os
from dataclasses import dataclass
from typing import Any

from genkit import GenkitError

# Model Garden's open models are mostly regional; ModelGarden in genkit-vertexai used the same default.
DEFAULT_LOCATION = 'us-central1'
# The only namespace OpenAICompatible(backend=GoogleCloud()) registers under.
MODEL_GARDEN_NAME = 'modelgarden'
_SCOPES = ('https://www.googleapis.com/auth/cloud-platform',)
_LOGIN_FIX = (
    'run `gcloud auth application-default login` on your machine, '
    'or give the service account roles/aiplatform.user when running on Google Cloud'
)


@dataclass(frozen=True, kw_only=True)
class GoogleCloud:
    """Serve Model Garden's OpenAI-compatible models (Llama, Mistral and others) from Google Cloud.

    Pass it as ``OpenAICompatible(backend=GoogleCloud(...))``. Requests
    authenticate with application default credentials and bill the project,
    so there's no API key. Models register as ``modelgarden/<publisher>/<model>``.

    Example:
        ```python
        from genkit import Genkit
        from genkit_openai import GoogleCloud, OpenAICompatible

        # 1. Model Garden through your gcloud login
        garden = OpenAICompatible(backend=GoogleCloud(project='bistro-prod'))
        ai = Genkit(plugins=[garden])

        # 2. The ref for any Model Garden model on the OpenAI-compatible endpoint
        res = await ai.generate(model=garden.model('meta/llama-3.3-70b-instruct-maas'), prompt='Suggest a dish.')
        print(res.text)
        # => Grilled salmon with lemon and dill
        ```

    Attributes:
        project: Google Cloud project. Defaults to ``GCLOUD_PROJECT``, then
            ``GOOGLE_CLOUD_PROJECT``, then the credentials' project.
        location: Google Cloud location. Defaults to ``GOOGLE_CLOUD_LOCATION``,
            then ``us-central1``.
    """

    project: str | None = None
    location: str | None = None


def resolve_project(config: GoogleCloud) -> str | None:
    """The project from the argument or the environment; None means ask the credentials."""
    return config.project or os.environ.get('GCLOUD_PROJECT') or os.environ.get('GOOGLE_CLOUD_PROJECT') or None


def resolve_location(config: GoogleCloud) -> str:
    """The location from the argument or the environment, else ``us-central1``."""
    return config.location or os.environ.get('GOOGLE_CLOUD_LOCATION') or DEFAULT_LOCATION


def model_garden_base_url(*, project: str, location: str) -> str:
    """Model Garden's OpenAI-compatible endpoint; the SDK appends ``/chat/completions``."""
    host = 'aiplatform.googleapis.com' if location == 'global' else f'{location}-aiplatform.googleapis.com'
    return f'https://{host}/v1beta1/projects/{project}/locations/{location}/endpoints/openapi'


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
            "OpenAICompatible(backend=GoogleCloud()) needs the google-cloud extra: uv add 'genkit-openai[google-cloud]'"
        ) from e


def _auth_errors() -> tuple[type[Exception], type[Exception]]:
    # Optional dependency, checked by require_google_auth at setup.
    from google.auth.exceptions import DefaultCredentialsError, RefreshError  # noqa: PLC0415

    return DefaultCredentialsError, RefreshError


def _missing_credentials(e: Exception) -> GenkitError:
    return GenkitError(
        status='FAILED_PRECONDITION',
        message=f'OpenAICompatible(backend=GoogleCloud()) found no Google Cloud credentials: {_LOGIN_FIX}',
        cause=e,
    )


def _stale_credentials(e: Exception) -> GenkitError:
    return GenkitError(
        status='FAILED_PRECONDITION',
        message=f'OpenAICompatible(backend=GoogleCloud()) could not refresh Google Cloud credentials: {_LOGIN_FIX}',
        cause=e,
    )


def load_adc() -> tuple[Any, str | None]:  # noqa: ANN401 - google.auth credentials are untyped
    """Application default credentials, refreshed, and their project (blocking: may hit the metadata server).

    Refreshing here makes a missing or stale login fail on the first call
    with the command that fixes it, instead of as an unmapped google.auth
    error inside the SDK.

    Raises:
        GenkitError: FAILED_PRECONDITION when there are no credentials, or
            they need a fresh login.
    """
    # Optional dependency, checked by require_google_auth at setup.
    import google.auth  # noqa: PLC0415
    from google.auth.transport.requests import Request  # noqa: PLC0415

    default_error, refresh_error = _auth_errors()
    try:
        credentials, project = google.auth.default(scopes=list(_SCOPES))
    except default_error as e:
        raise _missing_credentials(e) from e
    try:
        credentials.refresh(Request())
    except refresh_error as e:
        raise _stale_credentials(e) from e
    return credentials, project


class AdcToken:
    """The bearer token for each request, refreshed in a thread when it has expired.

    Passed to ``AsyncOpenAI`` as ``api_key``; the SDK awaits it before every
    request, so a long-lived server keeps one client and never sends a dead
    token.
    """

    def __init__(self, credentials: Any) -> None:  # noqa: ANN401 - google.auth credentials are untyped
        """Hold credentials that ``load_adc`` already refreshed."""
        self._credentials = credentials

    async def __call__(self) -> str:
        """A valid token, refreshing first if needed.

        Raises:
            GenkitError: FAILED_PRECONDITION when the refresh is rejected.
        """
        if not self._credentials.valid:
            # Optional dependency, checked by require_google_auth at setup.
            from google.auth.transport.requests import Request  # noqa: PLC0415

            _, refresh_error = _auth_errors()
            try:
                await asyncio.to_thread(self._credentials.refresh, Request())
            except refresh_error as e:
                raise _stale_credentials(e) from e
        return self._credentials.token
