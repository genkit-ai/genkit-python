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

"""Tests for Anthropic(backend=GoogleCloud(...)): Claude served from Google Cloud."""

import asyncio
import builtins
from dataclasses import FrozenInstanceError
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from anthropic import AsyncAnthropicVertex
from genkit_anthropic import Anthropic, GoogleCloud
from genkit_anthropic._google_cloud import load_adc
from genkit_anthropic._model_info import SUPPORTED_ANTHROPIC_MODELS
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from genkit import FinishReason, Genkit, GenkitError

_GCP_ENV = ('GCLOUD_PROJECT', 'GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION', 'ANTHROPIC_API_KEY')


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _GCP_ENV:
        monkeypatch.delenv(name, raising=False)


# Stands in for the machine's ADC so no test reads real credentials or calls Google.
FAKE_CREDENTIALS = MagicMock(name='adc-credentials')


@pytest.fixture(autouse=True)
def fake_adc() -> Any:  # noqa: ANN401 - the patched loader
    with patch('genkit_anthropic._google_cloud.load_adc', return_value=(FAKE_CREDENTIALS, None)) as adc:
        yield adc


def _client(plugin: Anthropic) -> Any:  # noqa: ANN401 - the SDK client under test
    async def _get() -> Any:  # noqa: ANN401
        return plugin._backend.client()

    return asyncio.run(_get())


def test_google_cloud_builds_vertex_client_with_project_and_location() -> None:
    """The plugin's client is AsyncAnthropicVertex for the configured project and location."""
    client = _client(Anthropic(backend=GoogleCloud(project='bistro-prod', location='us-east5')))

    assert isinstance(client, AsyncAnthropicVertex)
    assert client.project_id == 'bistro-prod'
    assert client.region == 'us-east5'


def test_google_cloud_reads_project_and_location_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """GCLOUD_PROJECT wins over GOOGLE_CLOUD_PROJECT; location falls back to GOOGLE_CLOUD_LOCATION."""
    monkeypatch.setenv('GCLOUD_PROJECT', 'from-gcloud')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'from-google-cloud')
    monkeypatch.setenv('GOOGLE_CLOUD_LOCATION', 'europe-west1')

    client = _client(Anthropic(backend=GoogleCloud()))

    assert client.project_id == 'from-gcloud'
    assert client.region == 'europe-west1'


def test_google_cloud_defaults_to_global_location() -> None:
    """With no location passed or set, Claude is served from the global endpoint."""
    client = _client(Anthropic(backend=GoogleCloud(project='bistro-prod')))

    assert client.region == 'global'


def test_google_cloud_rejects_api_key() -> None:
    """An API key can't authenticate a Google Cloud call, so passing one fails at setup."""
    with pytest.raises(ValueError, match='drop api_key'):
        Anthropic(backend=GoogleCloud(), api_key='sk-ant-test')


def test_backend_must_be_google_cloud() -> None:
    """A backend that isn't GoogleCloud fails at setup instead of silently using the Anthropic API."""
    with pytest.raises(TypeError, match='GoogleCloud'):
        Anthropic(backend=cast(Any, 'google_cloud'))


def test_google_cloud_is_frozen() -> None:
    """GoogleCloud is a frozen value, so one instance can be shared safely."""
    backend = GoogleCloud(project='bistro-prod')
    with pytest.raises(FrozenInstanceError):
        backend.project = 'other'  # type: ignore[misc]


def test_google_cloud_validates_client_options_against_vertex_client() -> None:
    """Vertex-only options pass; Anthropic-API-only options name the Vertex client in the error."""
    plugin = Anthropic(backend=GoogleCloud(project='p'), client_options={'access_token': 'tok', 'max_retries': 1})
    assert _client(plugin).max_retries == 1

    with pytest.raises(TypeError, match='AsyncAnthropicVertex accepts'):
        Anthropic(backend=GoogleCloud(project='p'), client_options={'auth_token': 'tok'})


@pytest.mark.parametrize('key', ['region', 'project_id'])
def test_google_cloud_reserves_region_and_project_id(key: str) -> None:
    """GoogleCloud(project=..., location=...) sets these, so client_options can't contradict them."""
    with pytest.raises(TypeError, match=f"unexpected client_options key '{key}'"):
        Anthropic(backend=GoogleCloud(project='p'), client_options={key: 'other'})


def test_google_cloud_accepts_v1_base_url() -> None:
    """Google Cloud endpoints end in /v1, so the direct-API /v1 guard doesn't apply."""
    client = _client(
        Anthropic(backend=GoogleCloud(project='p'), base_url='https://us-east5-aiplatform.googleapis.com/v1')
    )

    assert str(client.base_url).rstrip('/') == 'https://us-east5-aiplatform.googleapis.com/v1'


def test_google_cloud_without_google_auth_names_the_extra() -> None:
    """A missing google-auth fails on the constructor line with the install command."""
    real_import = builtins.__import__

    def _no_google_auth(name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if name == 'google.auth' or name.startswith('google.auth.'):
            raise ModuleNotFoundError(name=name)
        return real_import(name, *args, **kwargs)

    with (
        patch('builtins.__import__', _no_google_auth),
        pytest.raises(ImportError, match=r'genkit-anthropic\[google-cloud\]'),
    ):
        Anthropic(backend=GoogleCloud(project='p'))


@pytest.mark.asyncio
async def test_google_cloud_list_actions_uses_static_catalog_without_calling_api() -> None:
    """Google Cloud has no Models API, so listing returns the catalog and makes no call."""
    plugin = Anthropic(backend=GoogleCloud(project='p'))
    client = MagicMock()
    plugin._backend.client = client

    actions = await plugin.list_actions()
    again = await plugin.list_actions()

    client.assert_not_called()
    assert [a.name for a in actions] == [f'anthropic/{m}' for m in SUPPORTED_ANTHROPIC_MODELS]
    assert again is actions


def _vertex_response() -> MagicMock:
    response = MagicMock()
    response.content = [MagicMock(type='text', text='Grilled salmon')]
    response.usage = MagicMock(input_tokens=1, output_tokens=2)
    response.stop_reason = 'end_turn'
    return response


@pytest.mark.asyncio
async def test_google_cloud_generate_needs_no_api_key() -> None:
    """A Google Cloud plugin generates without ANTHROPIC_API_KEY; the request goes to the Vertex client."""
    plugin = Anthropic(backend=GoogleCloud(project='bistro-prod'))
    ai = Genkit(plugins=[plugin])
    with patch(
        'anthropic.resources.messages.AsyncMessages.create', AsyncMock(return_value=_vertex_response())
    ) as create:
        response = await ai.generate(model=Anthropic.claude('claude-sonnet-4-6'), prompt='Suggest a dish.')

    assert response.text == 'Grilled salmon'
    create.assert_awaited_once()
    assert create.call_args.kwargs['model'] == 'claude-sonnet-4-6'


@pytest.mark.asyncio
async def test_google_cloud_generate_takes_project_and_credentials_from_adc_once(fake_adc: MagicMock) -> None:
    """With no project set, ADC supplies it; the credentials load once and the client reuses them."""
    fake_adc.return_value = (FAKE_CREDENTIALS, 'adc-project')
    plugin = Anthropic(backend=GoogleCloud())
    ai = Genkit(plugins=[plugin])
    with patch('anthropic.resources.messages.AsyncMessages.create', AsyncMock(return_value=_vertex_response())):
        await ai.generate(model=Anthropic.claude('claude-sonnet-4-6'), prompt='One.')
        await ai.generate(model=Anthropic.claude('claude-sonnet-4-6'), prompt='Two.')

    fake_adc.assert_called_once()
    client = plugin._backend.client()
    assert isinstance(client, AsyncAnthropicVertex)
    assert client.project_id == 'adc-project'
    assert client.credentials is FAKE_CREDENTIALS


@pytest.mark.asyncio
async def test_google_cloud_explicit_project_beats_adc_project(fake_adc: MagicMock) -> None:
    """GoogleCloud(project=...) wins over the project the credentials carry."""
    fake_adc.return_value = (FAKE_CREDENTIALS, 'adc-project')
    plugin = Anthropic(backend=GoogleCloud(project='bistro-prod'))

    client = await plugin._backend.request_client(None)

    assert isinstance(client, AsyncAnthropicVertex)
    assert client.project_id == 'bistro-prod'


@pytest.mark.parametrize('own', [{'access_token': 'tok'}, {'credentials': MagicMock(name='service-account')}])
@pytest.mark.asyncio
async def test_google_cloud_own_credential_skips_adc(fake_adc: MagicMock, own: dict[str, Any]) -> None:
    """An access_token or credentials in client_options is used as is; ADC isn't loaded."""
    plugin = Anthropic(backend=GoogleCloud(project='bistro-prod'), client_options=own)

    await plugin._backend.request_client(None)

    fake_adc.assert_not_called()


@pytest.mark.asyncio
async def test_google_cloud_without_credentials_names_the_login_command(fake_adc: MagicMock) -> None:
    """No ADC fails FAILED_PRECONDITION with the gcloud command, not an opaque INTERNAL from inside the SDK."""
    fake_adc.side_effect = GenkitError(status='FAILED_PRECONDITION', message='no creds: gcloud auth ...')
    ai = Genkit(plugins=[Anthropic(backend=GoogleCloud(project='bistro-prod'))])
    with patch('anthropic.resources.messages.AsyncMessages.create', AsyncMock()) as create:
        response = await ai.generate(model=Anthropic.claude('claude-sonnet-4-6'), prompt='Suggest a dish.')

    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    create.assert_not_called()


@pytest.mark.asyncio
async def test_google_cloud_retries_credentials_after_a_failed_load(fake_adc: MagicMock) -> None:
    """A failed load isn't cached, so the next call works once the user has logged in."""
    fake_adc.side_effect = [
        GenkitError(status='FAILED_PRECONDITION', message='no creds'),
        (FAKE_CREDENTIALS, None),
    ]
    plugin = Anthropic(backend=GoogleCloud(project='bistro-prod'))

    with pytest.raises(GenkitError):
        await plugin._backend.request_client(None)
    client = await plugin._backend.request_client(None)

    assert client.credentials is FAKE_CREDENTIALS
    assert fake_adc.call_count == 2


def test_load_adc_refreshes_and_returns_project() -> None:
    """load_adc asks for the cloud-platform scope and refreshes now, so a stale login fails on the first call."""
    credentials = MagicMock(name='user-credentials')
    with patch('google.auth.default', return_value=(credentials, 'gcloud-project')) as default:
        assert load_adc() == (credentials, 'gcloud-project')

    assert default.call_args.kwargs['scopes'] == ['https://www.googleapis.com/auth/cloud-platform']
    credentials.refresh.assert_called_once()


@pytest.mark.parametrize(
    ('failure', 'match'),
    [
        (DefaultCredentialsError('Your default credentials were not found.'), 'found no Google Cloud credentials'),
        (RefreshError('Reauthentication is needed.'), 'could not refresh Google Cloud credentials'),
    ],
    ids=['no-adc', 'stale-login'],
)
def test_load_adc_failures_name_the_fix(failure: Exception, match: str) -> None:
    """A missing or stale login is a setup problem, reported with the command that fixes it."""
    credentials = MagicMock(name='user-credentials')
    if isinstance(failure, RefreshError):
        credentials.refresh.side_effect = failure
        default = patch('google.auth.default', return_value=(credentials, None))
    else:
        default = patch('google.auth.default', side_effect=failure)

    with default, pytest.raises(GenkitError) as exc_info:
        load_adc()

    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert match in str(exc_info.value)
    assert 'gcloud auth application-default login' in str(exc_info.value)


@pytest.mark.asyncio
async def test_google_cloud_generate_without_any_project_fails_precondition() -> None:
    """No project anywhere is a setup problem, reported as FAILED_PRECONDITION naming the fix."""
    plugin = Anthropic(backend=GoogleCloud())
    ai = Genkit(plugins=[plugin])
    response = await ai.generate(model=Anthropic.claude('claude-sonnet-4-6'), prompt='Suggest a dish.')

    assert response.finish_reason == FinishReason.FAILED
    assert 'GOOGLE_CLOUD_PROJECT' in str(response.finish_message)


@pytest.mark.asyncio
async def test_google_cloud_rejects_secrets_key_before_looking_up_project() -> None:
    """A context.secrets key can't bill Google Cloud, so the call fails without an ADC lookup or a request."""
    plugin = Anthropic(backend=GoogleCloud())
    ai = Genkit(plugins=[plugin])
    with (
        patch('genkit_anthropic._google_cloud.load_adc') as adc,
        patch('anthropic.resources.messages.AsyncMessages.create', AsyncMock()) as create,
    ):
        response = await ai.generate(
            model=Anthropic.claude('claude-sonnet-4-6'),
            prompt='Suggest a dish.',
            context={'secrets': {'api_key': 'tenant-key'}},
        )

    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert 'Google Cloud credentials' in response.error.message
    assert 'tenant-key' not in response.error.message
    adc.assert_not_called()
    create.assert_not_called()


@pytest.mark.parametrize(
    ('config', 'betas'),
    [
        ({'apiVersion': 'beta'}, None),
        ({'taskBudget': 20000}, ['task-budgets-2026-03-13']),
    ],
    ids=['beta-surface-no-defaults', 'task-budget-brings-its-header'],
)
@pytest.mark.asyncio
async def test_google_cloud_sends_only_the_betas_a_field_needs(config: dict[str, Any], betas: list[str] | None) -> None:
    """The Anthropic API's default betas stay off Google Cloud; task_budget still brings its own header."""
    plugin = Anthropic(backend=GoogleCloud(project='bistro-prod'))
    ai = Genkit(plugins=[plugin])
    # The Vertex client has its own beta resource, so stub this loop's client rather than an SDK class.
    client = plugin._backend.client()
    with patch.object(client.beta.messages, 'create', AsyncMock(return_value=_vertex_response())) as create:
        await ai.generate(model=Anthropic.claude('claude-sonnet-4-6'), prompt='Suggest a dish.', config=config)

    create.assert_awaited_once()
    assert create.call_args.kwargs.get('betas') == betas
