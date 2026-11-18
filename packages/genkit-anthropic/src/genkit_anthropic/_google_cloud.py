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

import os
from dataclasses import dataclass

# Every project that enables Claude gets quota on the global endpoint;
# regional endpoints need regional quota.
DEFAULT_LOCATION = 'global'


@dataclass(frozen=True, kw_only=True)
class GoogleCloud:
    """Serve Claude from Google Cloud, billed to a Google Cloud project.

    Pass it as ``Anthropic(backend=GoogleCloud(...))``. Requests authenticate
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


def adc_project() -> str | None:
    """The project of the application default credentials, or None (blocking: may hit the metadata server)."""
    # Optional dependency, checked by require_google_auth at setup.
    import google.auth  # noqa: PLC0415
    from google.auth.exceptions import DefaultCredentialsError  # noqa: PLC0415

    try:
        _, project = google.auth.default()
    except DefaultCredentialsError:
        return None
    return project
