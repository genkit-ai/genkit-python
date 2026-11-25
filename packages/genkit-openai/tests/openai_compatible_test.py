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

"""OpenAICompatible against a fake OpenAI-compatible server, driven through ai.generate and ai.embed."""

import json
from typing import Any

import httpx
import pytest
from genkit_openai import OpenAI, OpenAICompatible, OpenAIConfig

from genkit import Genkit, GenkitError
from genkit.plugin_api import ActionKind

BASE_URL = 'https://router.example.com/api/v1'


class FakeServer:
    """Answers chat, embeddings and /models like an OpenAI-compatible server, and records requests."""

    def __init__(self, *, model_ids: list[str] | None = None) -> None:
        self.requests: list[httpx.Request] = []
        self._model_ids = model_ids

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path
        if path.endswith('/models'):
            if self._model_ids is None:
                return httpx.Response(404, json={'error': {'message': 'Not found', 'type': 'not_found'}})
            data = [{'id': i, 'object': 'model', 'created': 0, 'owned_by': 'x'} for i in self._model_ids]
            return httpx.Response(200, json={'object': 'list', 'data': data})
        body = json.loads(request.content)
        if path.endswith('/embeddings'):
            data = [{'object': 'embedding', 'index': i, 'embedding': [0.1, 0.2]} for i, _ in enumerate(body['input'])]
            return httpx.Response(
                200,
                json={
                    'object': 'list',
                    'data': data,
                    'model': body['model'],
                    'usage': {'prompt_tokens': 1, 'total_tokens': 1},
                },
            )
        return httpx.Response(
            200,
            json={
                'id': 'chatcmpl-1',
                'object': 'chat.completion',
                'created': 0,
                'model': body['model'],
                'choices': [
                    {'index': 0, 'message': {'role': 'assistant', 'content': 'Rise & Shine'}, 'finish_reason': 'stop'}
                ],
                'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
            },
        )

    def body(self, index: int = -1) -> dict[str, Any]:
        return json.loads(self.requests[index].content)

    def auth(self, index: int = -1) -> str | None:
        return self.requests[index].headers.get('authorization')


def _plugin(server: FakeServer, *, name: str = 'router', **kwargs: Any) -> OpenAICompatible:  # noqa: ANN401
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return OpenAICompatible(name=name, base_url=BASE_URL, client_options={'http_client': http_client}, **kwargs)


@pytest.mark.parametrize(
    'server_id',
    ['meta-llama/llama-3.3-70b-instruct', 'openai/gpt-4o', 'google/gemini-2.5-pro'],
    ids=['vendor-slash', 'openai-prefixed', 'pro-suffix'],
)
@pytest.mark.asyncio
async def test_generate_sends_the_server_id_as_written(server_id: str) -> None:
    """`<name>/<id>` sends `<id>` untouched, including ids that start with openai/ or end in -pro."""
    server = FakeServer()
    router = _plugin(server, api_key='sk-router')
    ai = Genkit(plugins=[router])

    by_string = await ai.generate(model=f'router/{server_id}', prompt='Name a bakery.')
    by_ref = await ai.generate(model=router.model(server_id), prompt='Name a bakery.')

    assert by_string.text == by_ref.text == 'Rise & Shine'
    assert [server.body(0)['model'], server.body(1)['model']] == [server_id, server_id]
    assert str(server.requests[0].url) == f'{BASE_URL}/chat/completions'
    assert server.auth() == 'Bearer sk-router'


@pytest.mark.asyncio
async def test_model_ref_carries_typed_config() -> None:
    """`router.model(id, config=...)` is an OpenAIConfig ref; its config reaches the request."""
    server = FakeServer()
    router = _plugin(server, api_key='k')
    ai = Genkit(plugins=[router])

    await ai.generate(model=router.model('qwen/qwen3', config=OpenAIConfig(temperature=0.2)), prompt='hi')

    assert router.model('qwen/qwen3').config_schema is OpenAIConfig
    assert server.body()['temperature'] == 0.2


@pytest.mark.asyncio
async def test_runs_beside_openai_and_another_instance() -> None:
    """Each instance has its own namespace, so OpenAI() and two compatible servers coexist."""
    router_server, local_server = FakeServer(), FakeServer()
    ai = Genkit(
        plugins=[
            OpenAI(api_key='sk-openai'),
            _plugin(router_server, name='router', api_key='sk-router'),
            _plugin(local_server, name='local'),
        ]
    )

    await ai.generate(model='router/openai/gpt-4o', prompt='hi')
    await ai.generate(model='local/qwen3-8b', prompt='hi')

    assert router_server.body()['model'] == 'openai/gpt-4o'
    assert local_server.body()['model'] == 'qwen3-8b'
    assert await ai.lookup_model('openai/gpt-4o') is not None


@pytest.mark.asyncio
async def test_registers_no_built_in_catalog() -> None:
    """No OpenAI models show up for another server: init registers nothing."""
    assert await _plugin(FakeServer(), api_key='k').init() == []


@pytest.mark.asyncio
async def test_list_actions_lists_the_servers_models_without_openai_filters() -> None:
    """The Dev UI lists /models under the plugin's name; embed ids as embedders; -pro ids aren't dropped."""
    server = FakeServer(model_ids=['google/gemini-2.5-pro', 'openai/gpt-4o', 'nomic/nomic-embed-text', ''])
    router = _plugin(server, api_key='k')

    actions = await router.list_actions()
    again = await router.list_actions()

    assert [(a.name, a.kind) for a in actions] == [
        ('router/google/gemini-2.5-pro', ActionKind.MODEL),
        ('router/openai/gpt-4o', ActionKind.MODEL),
        ('router/nomic/nomic-embed-text', ActionKind.EMBEDDER),
    ]
    assert again is actions


@pytest.mark.asyncio
async def test_list_actions_on_a_server_without_models_lists_nothing_and_retries() -> None:
    """A 404 from /models lists nothing instead of raising, and isn't cached."""
    server = FakeServer(model_ids=None)
    router = _plugin(server, api_key='k')

    assert await router.list_actions() == []
    assert await router.list_actions() == []
    assert len([r for r in server.requests if r.url.path.endswith('/models')]) == 2


@pytest.mark.asyncio
async def test_no_key_never_sends_openai_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A keyless local server never gets OPENAI_API_KEY, even when it's set."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-openai-secret')
    server = FakeServer()
    ai = Genkit(plugins=[_plugin(server, name='local')])

    await ai.generate(model='local/qwen3-8b', prompt='hi')

    assert 'sk-openai-secret' not in (server.auth() or '')


@pytest.mark.asyncio
async def test_async_api_key_is_awaited_before_each_request() -> None:
    """A token function runs before every call, so an expiring token stays fresh."""
    tokens = iter(['tok-1', 'tok-2'])

    async def token() -> str:
        return next(tokens)

    server = FakeServer()
    ai = Genkit(plugins=[_plugin(server, api_key=token)])

    await ai.generate(model='router/meta/llama-3.3-70b-instruct-maas', prompt='hi')
    await ai.generate(model='router/meta/llama-3.3-70b-instruct-maas', prompt='hi')

    assert [server.auth(0), server.auth(1)] == ['Bearer tok-1', 'Bearer tok-2']


@pytest.mark.asyncio
async def test_secrets_key_replaces_the_plugin_key_for_that_call() -> None:
    """`context={'secrets': {'api_key': ...}}` authenticates one call; the next goes back to the plugin key."""
    server = FakeServer()
    ai = Genkit(plugins=[_plugin(server, api_key='sk-plugin')])

    await ai.generate(model='router/x', prompt='hi', context={'secrets': {'api_key': 'sk-tenant'}})
    await ai.generate(model='router/x', prompt='hi')

    assert [server.auth(0), server.auth(1)] == ['Bearer sk-tenant', 'Bearer sk-plugin']


@pytest.mark.asyncio
async def test_embed_sends_the_server_id_and_dimensions() -> None:
    """`ai.embed(embedder='router/<id>')` resolves by kind, with the id and dimensions sent as given."""
    server = FakeServer()
    ai = Genkit(plugins=[_plugin(server, api_key='k')])

    result = await ai.embed(
        embedder='router/nomic/nomic-embed-text', content='Mushroom Risotto', config={'dimensions': 2}
    )

    assert result[0].embedding == [0.1, 0.2]
    assert server.body()['model'] == 'nomic/nomic-embed-text'
    assert server.body()['dimensions'] == 2
    assert str(server.requests[-1].url) == f'{BASE_URL}/embeddings'


@pytest.mark.parametrize('bad_name', ['', 'open/router'])
def test_name_must_be_a_single_segment(bad_name: str) -> None:
    """The name is the namespace, so it can't be empty or contain the separator."""
    with pytest.raises(ValueError, match='non-empty string without'):
        OpenAICompatible(name=bad_name, base_url=BASE_URL)


def test_base_url_is_required() -> None:
    """Without a server there's nothing to be compatible with; OpenAI() is for OpenAI's API."""
    with pytest.raises(ValueError, match='needs base_url'):
        OpenAICompatible(name='router', base_url='')


def test_unknown_client_option_names_this_plugin() -> None:
    """A bad client_options key fails on the constructor line, naming OpenAICompatible."""
    with pytest.raises(TypeError, match="OpenAICompatible got an unexpected client_options key 'region'"):
        OpenAICompatible(name='router', base_url=BASE_URL, client_options={'region': 'us'})


def test_model_ref_rejects_an_empty_name() -> None:
    """An empty id would mint `router/`, which resolves to nothing."""
    router = OpenAICompatible(name='router', base_url=BASE_URL)
    with pytest.raises(GenkitError, match='model name must be a non-empty string'):
        router.model('')


@pytest.mark.asyncio
async def test_mistral_host_gets_mistrals_embedding_field_name() -> None:
    """Pointed at Mistral's own host, dimensions go out as output_dimension, which Mistral requires."""
    server = FakeServer()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    gateway = OpenAICompatible(
        name='mistral-direct',
        base_url='https://api.mistral.ai/v1',
        api_key='k',
        client_options={'http_client': http_client},
    )
    ai = Genkit(plugins=[gateway])

    await ai.embed(embedder='mistral-direct/mistral-embed', content='Mushroom Risotto', config={'dimensions': 2})

    assert server.body()['output_dimension'] == 2
    assert 'dimensions' not in server.body()
