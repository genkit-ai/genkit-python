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

"""What OpenAI(base_url='https://api.mistral.ai/v1') sends: the body Mistral's closed schema accepts."""

import base64
import json
import struct
from typing import Any

import httpx
import pytest
from genkit_openai import OpenAI
from genkit_openai._models._mistral import mistral_chat_body
from pydantic import BaseModel

from genkit import FinishReason, Genkit, Part

MISTRAL_URL = 'https://api.mistral.ai/v1'
CODESTRAL_URL = 'https://codestral.mistral.ai/v1'
PLUGIN_KEY = 'mistral-key'

# The fields Mistral's ChatCompletionRequest and EmbeddingRequest declare
# (https://docs.mistral.ai/openapi.yaml, as mirrored by the mistralai SDK). Mistral
# answers any other top-level field with a 422, and so does the fake below.
_CHAT_FIELDS = frozenset({
    'model',
    'messages',
    'temperature',
    'top_p',
    'max_tokens',
    'stream',
    'stop',
    'random_seed',
    'metadata',
    'response_format',
    'tools',
    'tool_choice',
    'presence_penalty',
    'frequency_penalty',
    'min_tokens',
    'repetition_penalty',
    'top_k',
    'logprobs',
    'top_logprobs',
    'prompt_logprobs',
    'top_prompt_logprobs',
    'n',
    'prediction',
    'parallel_tool_calls',
    'reasoning_effort',
    'prompt_mode',
    'guardrails',
    'prompt_cache_key',
    'service_tier',
    'safe_prompt',
})
_EMBEDDING_FIELDS = frozenset({'model', 'input', 'metadata', 'output_dimension', 'output_dtype', 'encoding_format'})
_MISTRAL_HOSTS = frozenset({'api.mistral.ai', 'codestral.mistral.ai'})

# A Magistral reply: reasoning in a thinking chunk, then the answer, in a content list.
_THINKING = {'type': 'thinking', 'thinking': [{'type': 'text', 'text': 'Two loaves, one each.'}]}


def _content(model: str, text: str) -> Any:  # noqa: ANN401
    if model.startswith('magistral'):
        return [_THINKING, {'type': 'text', 'text': text}]
    return text


def _completion(model: str) -> dict[str, Any]:
    return {
        'id': 'cmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': model,
        'choices': [
            {
                'index': 0,
                'finish_reason': 'stop',
                'message': {'role': 'assistant', 'content': _content(model, 'bonjour')},
            }
        ],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
    }


def _stream_body(model: str) -> bytes:
    # Mistral sends usage on the last chunk, beside finish_reason, without stream_options.
    first: Any = [_THINKING] if model.startswith('magistral') else 'bon'
    second: Any = [{'type': 'text', 'text': 'bonjour'}] if model.startswith('magistral') else 'jour'
    chunks = [
        {
            'id': 'cmpl-1',
            'object': 'chat.completion.chunk',
            'created': 0,
            'model': model,
            'choices': [{'index': 0, 'delta': {'role': 'assistant', 'content': first}, 'finish_reason': None}],
        },
        {
            'id': 'cmpl-1',
            'object': 'chat.completion.chunk',
            'created': 0,
            'model': model,
            'choices': [{'index': 0, 'delta': {'content': second}, 'finish_reason': 'stop'}],
            'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
        },
    ]
    return b''.join(f'data: {json.dumps(c)}\n\n'.encode() for c in chunks) + b'data: [DONE]\n\n'


def _embeddings(body: dict[str, Any]) -> dict[str, Any]:
    vector = [0.5, 0.25]
    embedding: Any = vector
    if body.get('encoding_format') == 'base64':
        embedding = base64.b64encode(struct.pack(f'<{len(vector)}f', *vector)).decode()
    return {
        'object': 'list',
        'model': body['model'],
        'data': [{'object': 'embedding', 'index': 0, 'embedding': embedding}],
        'usage': {'prompt_tokens': 1, 'total_tokens': 1},
    }


def _extra_forbidden(fields: list[str]) -> httpx.Response:
    detail = [
        {'type': 'extra_forbidden', 'loc': ['body', f], 'msg': 'Extra inputs are not permitted', 'input': None}
        for f in fields
    ]
    return httpx.Response(422, json={'object': 'error', 'message': {'detail': detail}, 'type': 'invalid_request_error'})


class _Server:
    """A fake OpenAI-compatible endpoint that records every request and, on Mistral's hosts, enforces its schema."""

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        embeddings = request.url.path.endswith('/embeddings')
        if request.url.host in _MISTRAL_HOSTS:
            allowed = _EMBEDDING_FIELDS if embeddings else _CHAT_FIELDS
            if unknown := sorted(set(body) - allowed):
                return _extra_forbidden(unknown)
        if embeddings:
            return httpx.Response(200, json=_embeddings(body))
        if body.get('stream'):
            return httpx.Response(
                200, content=_stream_body(body['model']), headers={'content-type': 'text/event-stream'}
            )
        return httpx.Response(200, json=_completion(body['model']))

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def server() -> _Server:
    return _Server()


def _ai(server: _Server, base_url: str | None) -> Genkit:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return Genkit(
        plugins=[
            OpenAI(
                api_key=PLUGIN_KEY,
                base_url=base_url,
                client_options={'http_client': http_client, 'max_retries': 0},
            )
        ]
    )


@pytest.fixture
def mistral(server: _Server) -> Genkit:
    return _ai(server, MISTRAL_URL)


@pytest.fixture
def openai(server: _Server) -> Genkit:
    return _ai(server, None)


@pytest.mark.asyncio
async def test_mistral_generate_goes_to_mistral_with_the_plugin_key(mistral: Genkit, server: _Server) -> None:
    """`ai.generate(model='openai/mistral-small-latest')` posts to Mistral's chat endpoint with the plugin's key."""
    response = await mistral.generate(model='openai/mistral-small-latest', prompt='salut')

    assert response.text == 'bonjour'
    [request] = server.requests
    assert str(request.url) == f'{MISTRAL_URL}/chat/completions'
    assert request.headers['authorization'] == f'Bearer {PLUGIN_KEY}'
    assert server.bodies()[0]['model'] == 'mistral-small-latest'


@pytest.mark.asyncio
async def test_mistral_stream_sends_no_stream_options_and_reads_usage(mistral: Genkit, server: _Server) -> None:
    """A streamed call to Mistral sends no `stream_options`, and usage still comes back from the last chunk."""
    stream = mistral.generate_stream(model='openai/mistral-small-latest', prompt='salut')
    chunks = [chunk.text async for chunk in stream.stream]
    response = await stream.response

    assert chunks == ['bon', 'jour']
    assert response.text == 'bonjour'
    assert response.usage is not None
    assert (response.usage.input_tokens, response.usage.output_tokens) == (3, 2)
    [body] = server.bodies()
    assert body['stream'] is True
    assert 'stream_options' not in body


@pytest.mark.asyncio
async def test_mistral_stream_drops_stream_options_from_config(mistral: Genkit, server: _Server) -> None:
    """`config={'stream_options': {...}}` is not sent to Mistral, which has no such field."""
    stream = mistral.generate_stream(
        model='openai/mistral-small-latest',
        prompt='salut',
        config={'stream_options': {'include_usage': True}},
    )
    await stream.response

    [body] = server.bodies()
    assert 'stream_options' not in body


@pytest.mark.parametrize(
    'config',
    [
        pytest.param({'max_output_tokens': 50}, id='chat'),
        pytest.param({'max_output_tokens': 50, 'reasoning_effort': 'high'}, id='reasoning_effort'),
    ],
)
@pytest.mark.asyncio
async def test_mistral_max_output_tokens_goes_out_as_max_tokens(
    mistral: Genkit, server: _Server, config: dict[str, Any]
) -> None:
    """`max_output_tokens` reaches Mistral as `max_tokens`, even with `reasoning_effort` set."""
    await mistral.generate(model='openai/magistral-medium-latest', prompt='salut', config=config)

    [body] = server.bodies()
    caps_sent = {k: body[k] for k in ('max_tokens', 'max_completion_tokens') if k in body}
    assert caps_sent == {'max_tokens': 50}
    assert body.get('reasoning_effort') == config.get('reasoning_effort')


@pytest.mark.asyncio
async def test_mistral_seed_goes_out_as_random_seed(mistral: Genkit, server: _Server) -> None:
    """`config={'seed': 7}` reaches Mistral as `random_seed`, beside the other settings."""
    await mistral.generate(
        model='openai/mistral-small-latest',
        prompt='salut',
        config={'seed': 7, 'temperature': 0.2, 'extra': {'safe_prompt': True}},
    )

    [body] = server.bodies()
    assert body['random_seed'] == 7
    assert 'seed' not in body
    assert body['temperature'] == 0.2
    assert body['safe_prompt'] is True


@pytest.mark.asyncio
async def test_mistral_random_seed_in_extra_wins_over_seed(mistral: Genkit, server: _Server) -> None:
    """`extra={'random_seed': 9}` replaces `seed=7`, as any `extra` key replaces a setting of the same name."""
    await mistral.generate(
        model='openai/mistral-small-latest',
        prompt='salut',
        config={'seed': 7, 'extra': {'random_seed': 9}},
    )

    [body] = server.bodies()
    assert body['random_seed'] == 9
    assert 'seed' not in body


@pytest.mark.asyncio
async def test_mistral_tenant_key_call_is_rewritten_too(mistral: Genkit, server: _Server) -> None:
    """A call with a `context.secrets` key still goes to Mistral with Mistral's field names."""
    stream = mistral.generate_stream(
        model='openai/mistral-small-latest',
        prompt='salut',
        config={'seed': 7},
        context={'secrets': {'api_key': 'tenant-key'}},
    )
    await stream.response

    [request] = server.requests
    assert request.headers['authorization'] == 'Bearer tenant-key'
    body = json.loads(request.content)
    assert 'stream_options' not in body
    assert body['random_seed'] == 7
    assert 'seed' not in body


@pytest.mark.asyncio
async def test_mistral_embed_dimensions_goes_out_as_output_dimension(mistral: Genkit, server: _Server) -> None:
    """`config={'dimensions': 256}` reaches Mistral's embeddings endpoint as `output_dimension`."""
    response = await mistral.embed(
        embedder='openai/codestral-embed', content='def f(): ...', config={'dimensions': 256}
    )

    assert response[0].embedding == [0.5, 0.25]
    [request] = server.requests
    assert str(request.url) == f'{MISTRAL_URL}/embeddings'
    body = json.loads(request.content)
    assert body['output_dimension'] == 256
    assert 'dimensions' not in body


@pytest.mark.asyncio
async def test_mistral_embed_float_encoding_still_goes_out(mistral: Genkit, server: _Server) -> None:
    """`encodingFormat='float'` with `dimensions` sends both, `dimensions` under Mistral's name."""
    await mistral.embed(
        embedder='openai/codestral-embed',
        content='def f(): ...',
        config={'dimensions': 256, 'encodingFormat': 'float'},
    )

    [body] = server.bodies()
    assert body['output_dimension'] == 256
    assert body['encoding_format'] == 'float'
    assert 'dimensions' not in body


@pytest.mark.asyncio
async def test_openai_base_url_keeps_openai_field_names(openai: Genkit, server: _Server) -> None:
    """Without Mistral's base URL the plugin still sends `stream_options`, `seed`, and `dimensions`."""
    stream = openai.generate_stream(model='openai/gpt-4o', prompt='hi', config={'seed': 7})
    await stream.response
    await openai.embed(embedder='openai/text-embedding-3-small', content='hi', config={'dimensions': 256})

    chat, embed = server.bodies()
    assert chat['stream_options'] == {'include_usage': True}
    assert chat['seed'] == 7
    assert 'random_seed' not in chat
    assert embed['dimensions'] == 256
    assert 'output_dimension' not in embed


@pytest.mark.asyncio
async def test_codestral_host_gets_mistral_field_names(server: _Server) -> None:
    """OpenAI(base_url='https://codestral.mistral.ai/v1') sends Mistral's names too."""
    ai = _ai(server, CODESTRAL_URL)

    await ai.generate(model='openai/codestral-latest', prompt='def add(a, b):', config={'seed': 7})

    [request] = server.requests
    assert str(request.url) == f'{CODESTRAL_URL}/chat/completions'
    body = json.loads(request.content)
    assert body['random_seed'] == 7
    assert 'seed' not in body


@pytest.mark.asyncio
async def test_openai_base_url_env_pointing_at_mistral_gets_mistral_field_names(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OPENAI_BASE_URL=https://api.mistral.ai/v1` with no `base_url` argument still sends Mistral's names."""
    monkeypatch.setenv('OPENAI_BASE_URL', MISTRAL_URL)
    ai = _ai(server, None)

    stream = ai.generate_stream(model='openai/mistral-small-latest', prompt='salut', config={'seed': 7})
    await stream.response

    [request] = server.requests
    assert request.url.host == 'api.mistral.ai'
    body = json.loads(request.content)
    assert 'stream_options' not in body
    assert body['random_seed'] == 7


@pytest.mark.asyncio
async def test_mistral_generate_drops_stream_options_from_config(mistral: Genkit, server: _Server) -> None:
    """`config={'stream_options': {...}}` on a non-streamed call is not sent to Mistral either."""
    await mistral.generate(
        model='openai/mistral-small-latest', prompt='salut', config={'stream_options': {'include_usage': True}}
    )

    [body] = server.bodies()
    assert 'stream_options' not in body


@pytest.mark.asyncio
async def test_mistral_top_k_goes_out(mistral: Genkit, server: _Server) -> None:
    """`config={'top_k': 5}` reaches Mistral, which declares `top_k`; OpenAI never gets it."""
    await mistral.generate(model='openai/mistral-small-latest', prompt='salut', config={'top_k': 5})

    [body] = server.bodies()
    assert body['top_k'] == 5


@pytest.mark.asyncio
async def test_mistral_magistral_reply_splits_reasoning_from_text(mistral: Genkit, server: _Server) -> None:
    """A Magistral reply's `thinking` chunk comes back as a reasoning part, its `text` chunk as the text."""
    response = await mistral.generate(model='openai/magistral-medium-latest', prompt='Bread for two?')

    assert response.text == 'bonjour'
    assert response.message is not None
    assert [p.reasoning for p in response.message.content if p.reasoning] == ['Two loaves, one each.']


@pytest.mark.asyncio
async def test_mistral_magistral_stream_splits_reasoning_from_text(mistral: Genkit, server: _Server) -> None:
    """Streamed `thinking` chunks arrive as reasoning parts, and the final response keeps reasoning before text."""
    stream = mistral.generate_stream(model='openai/magistral-medium-latest', prompt='Bread for two?')
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    assert [c.text for c in chunks] == ['', 'bonjour']
    assert chunks[0].content[0].reasoning == 'Two loaves, one each.'
    assert response.text == 'bonjour'
    assert response.message is not None
    assert response.message.content[0].reasoning == 'Two loaves, one each.'


@pytest.mark.asyncio
async def test_mistral_tools_pass_the_schema(mistral: Genkit, server: _Server) -> None:
    """A call with tools and `tool_choice` gets through Mistral's schema with the tool definitions."""

    @mistral.tool(name='lookup_menu')
    async def lookup_menu(dish: str) -> str:
        return f'{dish}: in stock'

    response = await mistral.generate(
        model='openai/mistral-small-latest',
        prompt='Is the baguette in stock?',
        tools=['lookup_menu'],
        tool_choice='auto',
    )

    assert response.text == 'bonjour'
    [body] = server.bodies()
    assert [t['function']['name'] for t in body['tools']] == ['lookup_menu']
    assert body['tool_choice'] == 'auto'


class _Loaf(BaseModel):
    name: str
    grams: int


@pytest.mark.asyncio
async def test_mistral_output_schema_goes_out_as_json_schema(mistral: Genkit, server: _Server) -> None:
    """`output_schema=` reaches Mistral as a `json_schema` response format."""
    await mistral.generate(model='openai/mistral-small-latest', prompt='Describe a loaf.', output_schema=_Loaf)

    [body] = server.bodies()
    assert body['response_format']['type'] == 'json_schema'
    assert set(body['response_format']['json_schema']['schema']['properties']) == {'name', 'grams'}


@pytest.mark.asyncio
async def test_mistral_image_input_goes_out_as_image_url(mistral: Genkit, server: _Server) -> None:
    """An image part reaches Mistral as an `image_url` content block."""
    await mistral.generate(
        model='openai/pixtral-large-latest',
        prompt=[Part.from_text('What bread is this?'), Part.from_media('https://example.com/loaf.png', 'image/png')],
    )

    [body] = server.bodies()
    blocks = body['messages'][-1]['content']
    assert {'type': 'image_url', 'image_url': {'url': 'https://example.com/loaf.png'}} in blocks


@pytest.mark.asyncio
async def test_mistral_rejects_an_openai_only_setting(mistral: Genkit, server: _Server) -> None:
    """An OpenAI-only setting such as `verbosity` is sent as given, and Mistral's 422 fails the turn."""
    response = await mistral.generate(model='openai/mistral-small-latest', prompt='salut', config={'verbosity': 'low'})

    assert response.finish_reason == FinishReason.FAILED
    assert 'verbosity' in str(response.error)
    [body] = server.bodies()
    assert body['verbosity'] == 'low'


@pytest.mark.asyncio
async def test_mistral_embed_without_options_sends_only_mistral_fields(mistral: Genkit, server: _Server) -> None:
    """An embed call with no options sends `model`, `input`, and the SDK's default `encoding_format`."""
    response = await mistral.embed(embedder='openai/mistral-embed', content='croissant')

    assert response[0].embedding == [0.5, 0.25]
    [body] = server.bodies()
    assert set(body) == {'model', 'input', 'encoding_format'}


def test_mistral_chat_body_leaves_its_input_alone() -> None:
    """`mistral_chat_body` returns a rewritten copy; the OpenAI kwargs it was given are unchanged."""
    body = {
        'model': 'mistral-small-latest',
        'seed': 7,
        'max_completion_tokens': 50,
        'stream_options': {'include_usage': True},
        'extra_body': {'safe_prompt': True},
    }
    before = json.loads(json.dumps(body))

    out = mistral_chat_body(body, top_k=5)

    assert body == before
    assert out == {
        'model': 'mistral-small-latest',
        'max_tokens': 50,
        'extra_body': {'random_seed': 7, 'top_k': 5, 'safe_prompt': True},
    }
