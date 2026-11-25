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

"""OpenAICompatible(backend=GoogleCloud()): Model Garden's OpenAI-compatible models, against a fake endpoint."""

import builtins
import json
from collections.abc import Iterator
from typing import Any, cast
from unittest.mock import MagicMock, patch

import httpx
import pytest
from genkit_openai import GoogleCloud, OpenAI, OpenAICompatible
from genkit_openai._google_cloud import AdcToken, load_adc, model_garden_base_url
from google.auth.exceptions import DefaultCredentialsError, RefreshError

from genkit import Genkit, GenkitError

LLAMA = 'meta/llama-3.3-70b-instruct-maas'
_ENV = ('GCLOUD_PROJECT', 'GOOGLE_CLOUD_PROJECT', 'GOOGLE_CLOUD_LOCATION', 'OPENAI_API_KEY')


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _ENV:
        monkeypatch.delenv(name, raising=False)


FAKE_TOKEN = 'adc-token'  # noqa: S105 - test double, not a credential


def _credentials() -> MagicMock:
    credentials = MagicMock(name='adc-credentials')
    credentials.valid = True
    credentials.token = FAKE_TOKEN
    return credentials


@pytest.fixture(autouse=True)
def fake_adc() -> Iterator[MagicMock]:
    """Stands in for the machine's ADC so no test reads real credentials or calls Google."""
    with patch('genkit_openai._compatible.load_adc', return_value=(_credentials(), None)) as adc:
        yield adc


class FakeGarden:
    """Answers chat completions like Model Garden's OpenAI-compatible endpoint, and records requests."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': body['model'],
                'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': 'Grilled salmon'}, 'finish_reason': 'stop'}
                ],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            },
        )


def _garden(server: FakeGarden, **config: Any) -> OpenAICompatible:  # noqa: ANN401
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return OpenAICompatible(backend=GoogleCloud(**config), client_options={'http_client': http_client})


@pytest.mark.asyncio
async def test_generate_goes_to_model_gardens_endpoint_with_the_adc_token() -> None:
    """`modelgarden/<publisher>/<model>` posts to the project's openapi endpoint with a Google bearer token."""
    server = FakeGarden()
    garden = _garden(server, project='bistro-prod', location='us-east5')
    ai = Genkit(plugins=[garden])

    response = await ai.generate(model=garden.model(LLAMA), prompt='Suggest a dish.')

    assert response.text == 'Grilled salmon'
    request = server.requests[-1]
    assert str(request.url) == (
        'https://us-east5-aiplatform.googleapis.com/v1beta1/projects/bistro-prod/locations/us-east5/endpoints/openapi'
        '/chat/completions'
    )
    assert request.headers['authorization'] == 'Bearer adc-token'
    assert json.loads(request.content)['model'] == LLAMA


def test_refs_are_modelgarden_publisher_model() -> None:
    """The ref helper gives the same refs genkit-vertexai's ModelGarden used for these models."""
    garden = OpenAICompatible(backend=GoogleCloud(project='p'))

    assert garden.name == 'modelgarden'
    assert garden.model(LLAMA).name == f'modelgarden/{LLAMA}'


@pytest.mark.parametrize(
    ('location', 'url'),
    [
        (
            'us-central1',
            'https://us-central1-aiplatform.googleapis.com/v1beta1/projects/p/locations/us-central1/endpoints/openapi',
        ),
        ('global', 'https://aiplatform.googleapis.com/v1beta1/projects/p/locations/global/endpoints/openapi'),
    ],
)
def test_endpoint_url_per_location(location: str, url: str) -> None:
    """Regional endpoints carry the region in the host; the global one doesn't."""
    assert model_garden_base_url(project='p', location=location) == url


@pytest.mark.asyncio
async def test_location_defaults_to_us_central1_and_project_to_adc(fake_adc: MagicMock) -> None:
    """With nothing set, the project comes from the credentials and the location is us-central1."""
    fake_adc.return_value = (_credentials(), 'adc-project')
    server = FakeGarden()
    ai = Genkit(plugins=[_garden(server)])

    await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')
    await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')

    fake_adc.assert_called_once()
    assert server.requests[-1].url.host == 'us-central1-aiplatform.googleapis.com'
    assert '/projects/adc-project/' in server.requests[-1].url.path


@pytest.mark.asyncio
async def test_env_project_and_location(monkeypatch: pytest.MonkeyPatch) -> None:
    """GCLOUD_PROJECT wins over GOOGLE_CLOUD_PROJECT; GOOGLE_CLOUD_LOCATION sets the region."""
    monkeypatch.setenv('GCLOUD_PROJECT', 'from-gcloud')
    monkeypatch.setenv('GOOGLE_CLOUD_PROJECT', 'from-google-cloud')
    monkeypatch.setenv('GOOGLE_CLOUD_LOCATION', 'europe-west4')
    server = FakeGarden()
    ai = Genkit(plugins=[_garden(server)])

    await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')

    assert str(server.requests[-1].url).startswith(
        'https://europe-west4-aiplatform.googleapis.com/v1beta1/projects/from-gcloud/locations/europe-west4/'
    )


@pytest.mark.asyncio
async def test_no_project_anywhere_fails_with_the_fix(fake_adc: MagicMock) -> None:
    """No project is a setup problem: FAILED_PRECONDITION naming three ways to set one."""
    fake_adc.return_value = (_credentials(), None)
    server = FakeGarden()
    ai = Genkit(plugins=[_garden(server)])

    response = await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')

    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert 'GOOGLE_CLOUD_PROJECT' in response.error.message
    assert server.requests == []


@pytest.mark.asyncio
async def test_missing_credentials_fail_and_are_retried_after_login(fake_adc: MagicMock) -> None:
    """A failed credential load surfaces as FAILED_PRECONDITION and isn't cached."""
    fake_adc.side_effect = [GenkitError(status='FAILED_PRECONDITION', message='no creds'), (_credentials(), None)]
    server = FakeGarden()
    ai = Genkit(plugins=[_garden(server, project='p')])

    first = await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')
    second = await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')

    assert first.error is not None and first.error.status == 'FAILED_PRECONDITION'
    assert second.text == 'Grilled salmon'
    assert fake_adc.call_count == 2


@pytest.mark.asyncio
async def test_secrets_key_is_refused_before_loading_credentials(fake_adc: MagicMock) -> None:
    """A context.secrets key can't bill a Google Cloud project, so the call fails without touching ADC."""
    server = FakeGarden()
    ai = Genkit(plugins=[_garden(server, project='p')])

    response = await ai.generate(
        model=f'modelgarden/{LLAMA}', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}}
    )

    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert 'Google Cloud credentials' in response.error.message
    assert 'sk-tenant' not in response.error.message
    fake_adc.assert_not_called()
    assert server.requests == []


@pytest.mark.asyncio
async def test_lists_nothing_and_calls_no_models_endpoint() -> None:
    """Model Garden has no /models, so the Dev UI lists nothing for it and no request goes out."""
    server = FakeGarden()
    garden = _garden(server, project='p')

    assert await garden.list_actions() == []
    assert server.requests == []


@pytest.mark.asyncio
async def test_runs_beside_openai_and_a_named_compatible_server() -> None:
    """modelgarden/, openai/ and a named server's namespace don't collide."""
    garden_server = FakeGarden()
    ai = Genkit(
        plugins=[
            OpenAI(api_key='sk-openai'),
            OpenAICompatible(name='openrouter', base_url='https://openrouter.ai/api/v1', api_key='sk-or'),
            _garden(garden_server, project='p'),
        ]
    )

    await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')

    assert json.loads(garden_server.requests[-1].content)['model'] == LLAMA
    assert await ai.lookup_model('openrouter/meta-llama/llama-3.3-70b-instruct') is not None
    assert await ai.lookup_model('openai/gpt-4o') is not None


@pytest.mark.asyncio
async def test_base_url_overrides_the_endpoint() -> None:
    """base_url replaces the derived endpoint, e.g. for Private Service Connect; auth stays Google's."""
    server = FakeGarden()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    garden = OpenAICompatible(
        backend=GoogleCloud(project='p'),
        base_url='https://psc.example.internal/v1',
        client_options={'http_client': http_client},
    )
    ai = Genkit(plugins=[garden])

    await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')

    assert str(server.requests[-1].url) == 'https://psc.example.internal/v1/chat/completions'
    assert server.requests[-1].headers['authorization'] == 'Bearer adc-token'


@pytest.mark.parametrize('name', [None, 'modelgarden'])
def test_name_may_be_left_out_or_modelgarden(name: str | None) -> None:
    """With GoogleCloud the namespace is always modelgarden; saying so explicitly is fine."""
    garden = OpenAICompatible(backend=GoogleCloud(project='p'), name=cast(Any, name))

    assert garden.name == 'modelgarden'


def test_other_name_with_google_cloud_raises() -> None:
    """Any other name fails on the constructor line, naming the fix."""
    with pytest.raises(ValueError, match="registers under 'modelgarden'; drop name='vertex'"):
        OpenAICompatible(backend=GoogleCloud(project='p'), name=cast(Any, 'vertex'))


def test_api_key_with_google_cloud_raises() -> None:
    """An API key can't authenticate a Google Cloud call, so passing one fails at setup."""
    with pytest.raises(ValueError, match='drop api_key'):
        OpenAICompatible(backend=GoogleCloud(project='p'), **cast(Any, {'api_key': 'sk-x'}))


def test_backend_must_be_google_cloud() -> None:
    """A backend that isn't GoogleCloud fails at setup."""
    with pytest.raises(TypeError, match='GoogleCloud'):
        OpenAICompatible(backend=cast(Any, 'google_cloud'))


def test_without_backend_name_and_base_url_are_still_required() -> None:
    """The generic plugin has no default name or server, as in JS and Go."""
    with pytest.raises(ValueError, match='non-empty string'):
        OpenAICompatible(**cast(Any, {'base_url': 'https://openrouter.ai/api/v1'}))
    with pytest.raises(ValueError, match='needs base_url'):
        OpenAICompatible(**cast(Any, {'name': 'openrouter'}))


def test_without_google_auth_names_the_extra() -> None:
    """A missing google-auth fails on the constructor line with the install command."""
    real_import = builtins.__import__

    def _no_google_auth(name: str, *args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        if name == 'google.auth' or name.startswith('google.auth.'):
            raise ModuleNotFoundError(name=name)
        return real_import(name, *args, **kwargs)

    with (
        patch('builtins.__import__', _no_google_auth),
        pytest.raises(ImportError, match=r'genkit-openai\[google-cloud\]'),
    ):
        OpenAICompatible(backend=GoogleCloud(project='p'))


def test_load_adc_refreshes_and_returns_project() -> None:
    """load_adc asks for the cloud-platform scope and refreshes now, so a stale login fails on the first call."""
    credentials = MagicMock(name='user-credentials')
    with patch('google.auth.default', return_value=(credentials, 'gcloud-project')) as default:
        assert load_adc() == (credentials, 'gcloud-project')

    assert default.call_args.kwargs['scopes'] == ['https://www.googleapis.com/auth/cloud-platform']
    credentials.refresh.assert_called_once()


@pytest.mark.parametrize('stale', [False, True], ids=['no-adc', 'stale-login'])
def test_load_adc_failures_name_the_login_command(stale: bool) -> None:
    """A missing or stale login is a setup problem, reported with the command that fixes it."""
    credentials = MagicMock(name='user-credentials')
    if stale:
        credentials.refresh.side_effect = RefreshError('Reauthentication is needed.')
        default = patch('google.auth.default', return_value=(credentials, None))
    else:
        default = patch('google.auth.default', side_effect=DefaultCredentialsError('not found'))

    with default, pytest.raises(GenkitError) as exc_info:
        load_adc()

    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'gcloud auth application-default login' in str(exc_info.value)


@pytest.mark.asyncio
async def test_token_refreshes_only_when_expired() -> None:
    """The token function refreshes an expired credential and reuses a valid one."""
    credentials = MagicMock(name='adc-credentials')
    credentials.valid = False
    credentials.token = 'fresh'
    token = AdcToken(credentials)

    assert await token() == 'fresh'
    credentials.refresh.assert_called_once()

    credentials.valid = True
    assert await token() == 'fresh'
    credentials.refresh.assert_called_once()


@pytest.mark.asyncio
async def test_token_refresh_rejected_names_the_login_command() -> None:
    """A refresh rejected mid-run fails the call with FAILED_PRECONDITION and the fix."""
    credentials = MagicMock(name='adc-credentials')
    credentials.valid = False
    credentials.refresh.side_effect = RefreshError('Reauthentication is needed.')

    with pytest.raises(GenkitError) as exc_info:
        await AdcToken(credentials)()

    assert exc_info.value.status == 'FAILED_PRECONDITION'
    assert 'gcloud auth application-default login' in str(exc_info.value)


@pytest.mark.asyncio
async def test_explicit_project_beats_the_credentials_project(fake_adc: MagicMock) -> None:
    """GoogleCloud(project=...) wins over the project the credentials carry."""
    fake_adc.return_value = (_credentials(), 'adc-project')
    server = FakeGarden()
    ai = Genkit(plugins=[_garden(server, project='bistro-prod')])

    await ai.generate(model=f'modelgarden/{LLAMA}', prompt='hi')

    assert '/projects/bistro-prod/' in server.requests[-1].url.path
