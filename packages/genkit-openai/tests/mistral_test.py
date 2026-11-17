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
from typing import Any, Literal, cast

import httpx
import pytest
from genkit_openai import OpenAI
from genkit_openai._models._mistral import mistral_chat_body
from pydantic import BaseModel

from genkit import FinishReason, Genkit, Message, Part, Role

MISTRAL_URL = 'https://api.mistral.ai/v1'
CODESTRAL_URL = 'https://codestral.mistral.ai/v1'
# A gateway that forwards to Mistral, so the fake enforces Mistral's schema on it too.
GATEWAY_URL = 'https://llm-gateway.example.com/v1'
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
_MISTRAL_HOSTS = frozenset({
    'api.mistral.ai',
    'api.eu.mistral.ai',
    'api.us.mistral.ai',
    'codestral.mistral.ai',
    'llm-gateway.example.com',
})
# Chunk types a message's content list may hold, and the keys a thinking chunk
# may carry (ContentChunk and ThinkChunk in the mistralai SDK).
_CHUNK_TYPES = frozenset({
    'text',
    'image_url',
    'reference',
    'file',
    'input_audio',
    'document_url',
    'resource',
    'resource_link',
    'thinking',
})
_THINKING_KEYS = frozenset({'type', 'thinking', 'signature', 'closed'})

SIGNATURE = 'sig-abc'
REASONING = 'Two loaves, one each.'


def _thinking(text: str, *, signature: str | None = None) -> dict[str, Any]:
    chunk: dict[str, Any] = {'type': 'thinking', 'thinking': [{'type': 'text', 'text': text}]}
    if signature is not None:
        chunk['signature'] = signature
    return chunk


def _completion(body: dict[str, Any], finish_reason: str, content: Any = None) -> dict[str, Any]:  # noqa: ANN401
    # With reasoning_effort set, Mistral puts the reasoning in a thinking chunk
    # ahead of the answer, in a content list.
    if content is None and body.get('reasoning_effort'):
        content = [_thinking(REASONING, signature=SIGNATURE), {'type': 'text', 'text': 'bonjour'}]
    elif content is None:
        content = 'bonjour'
    return {
        'id': 'cmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': body['model'],
        'choices': [{'index': 0, 'finish_reason': finish_reason, 'message': {'role': 'assistant', 'content': content}}],
        'usage': {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5},
    }


def _stream_body(body: dict[str, Any], deltas: list[Any] | None = None) -> bytes:
    # Mistral sends usage on the last chunk, beside finish_reason, without stream_options.
    # With reasoning_effort set, deltas go: a list with a thinking chunk, a list
    # with the closing thinking chunk and the first text chunk, then strings.
    if deltas is None and not body.get('reasoning_effort'):
        deltas = ['bon', 'jour']
    elif deltas is None:
        deltas = [
            [_thinking('Two loaves,')],
            [_thinking(' one each.', signature=SIGNATURE), {'type': 'text', 'text': 'bon'}],
            'jour',
        ]
    chunks = []
    for i, delta in enumerate(deltas):
        last = i == len(deltas) - 1
        chunk: dict[str, Any] = {
            'id': 'cmpl-1',
            'object': 'chat.completion.chunk',
            'created': 0,
            'model': body['model'],
            'choices': [{'index': 0, 'delta': {'content': delta}, 'finish_reason': 'stop' if last else None}],
        }
        if last:
            chunk['usage'] = {'prompt_tokens': 3, 'completion_tokens': 2, 'total_tokens': 5}
        chunks.append(chunk)
    return b''.join(f'data: {json.dumps(c)}\n\n'.encode() for c in chunks) + b'data: [DONE]\n\n'


def _bad_chunks(body: dict[str, Any]) -> list[str]:
    bad: list[str] = []
    for i, message in enumerate(body.get('messages', [])):
        content = message.get('content')
        if not isinstance(content, list):
            continue
        for j, chunk in enumerate(content):
            if chunk.get('type') not in _CHUNK_TYPES:
                bad.append(f'messages.{i}.content.{j}.type')
            elif chunk['type'] == 'thinking' and set(chunk) - _THINKING_KEYS:
                bad.append(f'messages.{i}.content.{j}')
    return bad


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
        self.finish_reason = 'stop'
        # Replaces the reply's content, or the streamed deltas, when set.
        self.content: Any = None
        self.deltas: list[Any] | None = None

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        body = json.loads(request.content)
        embeddings = request.url.path.endswith('/embeddings')
        if request.url.host in _MISTRAL_HOSTS:
            allowed = _EMBEDDING_FIELDS if embeddings else _CHAT_FIELDS
            if unknown := sorted(set(body) - allowed) + _bad_chunks(body):
                return _extra_forbidden(unknown)
        if embeddings:
            return httpx.Response(200, json=_embeddings(body))
        if body.get('stream'):
            stream = _stream_body(body, self.deltas)
            return httpx.Response(200, content=stream, headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json=_completion(body, self.finish_reason, self.content))

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(r.content) for r in self.requests]


@pytest.fixture
def server() -> _Server:
    return _Server()


def _ai(
    server: _Server,
    base_url: str | None,
    *,
    api_key: str | None = PLUGIN_KEY,
    provider: Literal['mistral'] | None = None,
) -> Genkit:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return Genkit(
        plugins=[
            OpenAI(
                api_key=api_key,
                base_url=base_url,
                client_options={'http_client': http_client, 'max_retries': 0},
                provider=provider,
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
    await mistral.generate(model='openai/mistral-small-latest', prompt='salut', config=config)

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
        config={'seed': 7, 'temperature': 0.2, 'extra': {'min_tokens': 5}},
    )

    [body] = server.bodies()
    assert body['random_seed'] == 7
    assert 'seed' not in body
    assert body['temperature'] == 0.2
    assert body['min_tokens'] == 5


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
async def test_mistral_refuses_top_k(mistral: Genkit, server: _Server) -> None:
    """`config={'top_k': 5}` fails the turn before the request goes out: Mistral's models reject `top_k`."""
    response = await mistral.generate(model='openai/mistral-small-latest', prompt='salut', config={'top_k': 5})

    assert response.finish_reason == FinishReason.FAILED
    assert 'top_k' in str(response.error)
    assert server.bodies() == []


_REASONING_CONFIG = {'reasoning_effort': 'high'}


@pytest.mark.asyncio
async def test_mistral_reasoning_reply_splits_reasoning_from_text(mistral: Genkit, server: _Server) -> None:
    """With `reasoning_effort` set, the reply's `thinking` chunk is a reasoning part and its `text` chunk the text."""
    response = await mistral.generate(
        model='openai/mistral-small-latest', prompt='Bread for two?', config=_REASONING_CONFIG
    )

    assert response.text == 'bonjour'
    assert response.message is not None
    [reasoning] = [p for p in response.message.content if p.reasoning]
    assert reasoning.reasoning == REASONING
    assert reasoning.metadata == {'reasoningSource': 'mistral', 'thoughtSignature': SIGNATURE}


@pytest.mark.asyncio
async def test_mistral_reasoning_stream_splits_reasoning_from_text(mistral: Genkit, server: _Server) -> None:
    """Streamed thinking chunks arrive as reasoning parts, through all three delta shapes Mistral sends."""
    stream = mistral.generate_stream(
        model='openai/mistral-small-latest', prompt='Bread for two?', config=_REASONING_CONFIG
    )
    chunks = [chunk async for chunk in stream.stream]
    response = await stream.response

    assert [c.text for c in chunks] == ['', 'bon', 'jour']
    assert ''.join(p.reasoning for c in chunks for p in c.content if p.reasoning) == REASONING
    assert response.text == 'bonjour'
    assert response.message is not None
    assert ''.join(p.reasoning for p in response.message.content if p.reasoning) == REASONING


@pytest.mark.parametrize('streamed', [False, True], ids=['generate', 'stream'])
@pytest.mark.asyncio
async def test_mistral_replays_reasoning_as_a_thinking_chunk(mistral: Genkit, server: _Server, streamed: bool) -> None:
    """The next turn sends the reply's reasoning back as one leading `thinking` chunk, signature included."""
    if streamed:
        stream = mistral.generate_stream(
            model='openai/mistral-small-latest', prompt='Bread for two?', config=_REASONING_CONFIG
        )
        first = await stream.response
    else:
        first = await mistral.generate(
            model='openai/mistral-small-latest', prompt='Bread for two?', config=_REASONING_CONFIG
        )

    second = await mistral.generate(model='openai/mistral-small-latest', messages=first.messages, prompt='And three?')

    assert second.finish_reason == FinishReason.STOP
    body = server.bodies()[-1]
    [assistant] = [m for m in body['messages'] if m['role'] == 'assistant']
    assert assistant['content'] == [
        {'type': 'thinking', 'thinking': [{'type': 'text', 'text': REASONING}], 'signature': SIGNATURE},
        {'type': 'text', 'text': 'bonjour'},
    ]


@pytest.mark.asyncio
async def test_openai_host_does_not_replay_reasoning(openai: Genkit, server: _Server) -> None:
    """On any other host a model message's reasoning is still left out of the next turn."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Bread for two?')]),
        Message(role=Role.MODEL, content=[Part.from_reasoning(REASONING), Part.from_text('bonjour')]),
    ]

    await openai.generate(model='openai/gpt-4o', messages=history, prompt='And three?')

    [body] = server.bodies()
    [assistant] = [m for m in body['messages'] if m['role'] == 'assistant']
    assert assistant['content'] == 'bonjour'


@pytest.mark.asyncio
async def test_mistral_model_length_finish_is_length(mistral: Genkit, server: _Server) -> None:
    """Mistral's `model_length` finish reason comes back as LENGTH."""
    server.finish_reason = 'model_length'

    response = await mistral.generate(model='openai/mistral-small-latest', prompt='salut')

    assert response.finish_reason == FinishReason.LENGTH


@pytest.mark.parametrize('host', ['api.eu.mistral.ai', 'api.us.mistral.ai'])
@pytest.mark.asyncio
async def test_mistral_regional_hosts_get_mistral_field_names(server: _Server, host: str) -> None:
    """The EU and US endpoints get the same rewrite as the global one."""
    ai = _ai(server, f'https://{host}/v1')

    stream = ai.generate_stream(model='openai/mistral-small-latest', prompt='salut', config={'seed': 7})
    await stream.response

    [request] = server.requests
    assert request.url.host == host
    body = json.loads(request.content)
    assert 'stream_options' not in body
    assert body['random_seed'] == 7


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
        'extra_body': {'min_tokens': 5},
    }
    before = json.loads(json.dumps(body))

    out = mistral_chat_body(body)

    assert body == before
    assert out == {
        'model': 'mistral-small-latest',
        'max_tokens': 50,
        'extra_body': {'random_seed': 7, 'min_tokens': 5},
    }


def _mistral_reasoning(text: str, signature: str | None = SIGNATURE) -> Part:
    """A reasoning part as the plugin reads it from a Mistral thinking chunk."""
    metadata: dict[str, Any] = {'reasoningSource': 'mistral'}
    if signature is not None:
        metadata['thoughtSignature'] = signature
    return Part.from_reasoning(text, metadata=metadata)


def _assistant_messages(body: dict[str, Any]) -> list[dict[str, Any]]:
    return [m for m in body['messages'] if m['role'] == 'assistant']


@pytest.mark.asyncio
async def test_mistral_replays_only_mistral_reasoning_from_a_mixed_history(mistral: Genkit, server: _Server) -> None:
    """Reasoning another model left in the history, signed or not, is not sent to Mistral; Mistral's own is."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Bread for two?')]),
        # A Gemini or Claude turn: its reasoning carries a signature under the same key.
        Message(
            role=Role.MODEL,
            content=[
                Part.from_reasoning('Gemini thinks.', metadata={'thoughtSignature': 'gemini-sig'}),
                Part.from_text('Two loaves.'),
            ],
        ),
        Message(role=Role.USER, content=[Part.from_text('And three?')]),
        Message(role=Role.MODEL, content=[Part.from_reasoning('Unsigned, unmarked.'), Part.from_text('Three.')]),
        Message(role=Role.USER, content=[Part.from_text('And four?')]),
        Message(role=Role.MODEL, content=[_mistral_reasoning(REASONING), Part.from_text('Four.')]),
    ]

    response = await mistral.generate(model='openai/mistral-small-latest', messages=history, prompt='And five?')

    assert response.finish_reason == FinishReason.STOP
    [body] = server.bodies()
    gemini, unmarked, own = _assistant_messages(body)
    assert gemini['content'] == 'Two loaves.'
    assert unmarked['content'] == 'Three.'
    assert own['content'] == [
        {'type': 'thinking', 'thinking': [{'type': 'text', 'text': REASONING}], 'signature': SIGNATURE},
        {'type': 'text', 'text': 'Four.'},
    ]


@pytest.mark.asyncio
async def test_mistral_reasoning_replays_after_a_json_round_trip(mistral: Genkit, server: _Server) -> None:
    """A history saved as JSON and loaded back still replays Mistral's reasoning, signature included."""
    first = await mistral.generate(
        model='openai/mistral-small-latest', prompt='Bread for two?', config=_REASONING_CONFIG
    )
    saved = json.dumps([m.model_dump(mode='json', by_alias=True, exclude_none=True) for m in first.messages])
    history = [Message.model_validate(m) for m in json.loads(saved)]

    await mistral.generate(model='openai/mistral-small-latest', messages=history, prompt='And three?')

    [assistant] = _assistant_messages(server.bodies()[-1])
    assert assistant['content'][0] == {
        'type': 'thinking',
        'thinking': [{'type': 'text', 'text': REASONING}],
        'signature': SIGNATURE,
    }


@pytest.mark.asyncio
async def test_mistral_replays_reasoning_onto_a_tool_call_message_with_no_content(
    mistral: Genkit, server: _Server
) -> None:
    """A turn with reasoning and only tool calls goes back as one assistant message: thinking chunk and calls."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Is the baguette in stock?')]),
        Message(
            role=Role.MODEL,
            content=[
                _mistral_reasoning(REASONING),
                Part.from_tool_request(name='lookup_menu', input={'dish': 'baguette'}, ref='call-1'),
            ],
        ),
        Message(
            role=Role.TOOL,
            content=[Part.from_tool_response(name='lookup_menu', output='baguette: in stock', ref='call-1')],
        ),
    ]

    response = await mistral.generate(model='openai/mistral-small-latest', messages=history)

    assert response.finish_reason == FinishReason.STOP
    [body] = server.bodies()
    [assistant] = _assistant_messages(body)
    assert assistant['content'] == [
        {'type': 'thinking', 'thinking': [{'type': 'text', 'text': REASONING}], 'signature': SIGNATURE}
    ]
    assert [c['id'] for c in assistant['tool_calls']] == ['call-1']


@pytest.mark.asyncio
async def test_mistral_replays_a_reasoning_only_turn(mistral: Genkit, server: _Server) -> None:
    """A turn that is only reasoning goes back as an assistant message holding just the thinking chunk."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Bread for two?')]),
        Message(role=Role.MODEL, content=[_mistral_reasoning(REASONING, signature=None)]),
    ]

    await mistral.generate(model='openai/mistral-small-latest', messages=history, prompt='Well?')

    [assistant] = _assistant_messages(server.bodies()[0])
    assert assistant['content'] == [{'type': 'thinking', 'thinking': [{'type': 'text', 'text': REASONING}]}]


@pytest.mark.parametrize('streamed', [False, True], ids=['generate', 'stream'])
@pytest.mark.asyncio
async def test_openai_host_reads_list_content(openai: Genkit, server: _Server, streamed: bool) -> None:
    """A non-Mistral host that answers with a list of text chunks gives back their text, with no reasoning part."""
    server.content = [{'type': 'text', 'text': 'bon'}, {'type': 'text', 'text': 'jour'}]
    server.deltas = [[{'type': 'text', 'text': 'bon'}], 'jour']

    if streamed:
        stream = openai.generate_stream(model='openai/gpt-4o', prompt='hi')
        chunks = [chunk.text async for chunk in stream.stream]
        response = await stream.response
        assert chunks == ['bon', 'jour']
    else:
        response = await openai.generate(model='openai/gpt-4o', prompt='hi')

    assert response.text == 'bonjour'
    assert response.message is not None
    assert [p for p in response.message.content if p.reasoning is not None] == []


@pytest.mark.parametrize('streamed', [False, True], ids=['generate', 'stream'])
@pytest.mark.asyncio
async def test_mistral_renames_openai_fields_in_extra(mistral: Genkit, server: _Server, streamed: bool) -> None:
    """`extra` with `seed`, `max_completion_tokens` or `stream_options` goes out under Mistral's names, not as a 422."""
    config = {'extra': {'seed': 1, 'max_completion_tokens': 50, 'stream_options': {'include_usage': True}}}
    if streamed:
        stream = mistral.generate_stream(model='openai/mistral-small-latest', prompt='salut', config=config)
        response = await stream.response
    else:
        response = await mistral.generate(model='openai/mistral-small-latest', prompt='salut', config=config)

    assert response.finish_reason == FinishReason.STOP
    [body] = server.bodies()
    assert body['random_seed'] == 1
    assert body['max_tokens'] == 50
    assert not {'seed', 'max_completion_tokens', 'stream_options'} & set(body)


def test_mistral_chat_body_prefers_mistral_names_in_extra() -> None:
    """With both names in `extra`, Mistral's own one is kept; `extra` still beats the config's `seed`."""
    out = mistral_chat_body({
        'seed': 7,
        'extra_body': {'seed': 1, 'random_seed': 2, 'max_completion_tokens': 50, 'max_tokens': 60},
    })

    assert out == {'extra_body': {'random_seed': 2, 'max_tokens': 60}}


@pytest.mark.asyncio
async def test_provider_mistral_sends_mistral_names_through_a_gateway(server: _Server) -> None:
    """`OpenAI(base_url=<gateway>, provider='mistral')` sends Mistral's names to a host that isn't Mistral's."""
    ai = _ai(server, GATEWAY_URL, provider='mistral')

    stream = ai.generate_stream(
        model='openai/mistral-small-latest', prompt='Bread for two?', config={'seed': 7, **_REASONING_CONFIG}
    )
    first = await stream.response
    await ai.generate(model='openai/mistral-small-latest', messages=first.messages, prompt='And three?')
    await ai.embed(embedder='openai/mistral-embed', content='croissant', config={'dimensions': 256})

    streamed, replay, embed = server.requests
    assert {r.url.host for r in server.requests} == {'llm-gateway.example.com'}
    streamed_body = json.loads(streamed.content)
    assert 'stream_options' not in streamed_body
    assert streamed_body['random_seed'] == 7
    [assistant] = _assistant_messages(json.loads(replay.content))
    assert assistant['content'][0]['type'] == 'thinking'
    assert json.loads(embed.content)['output_dimension'] == 256


@pytest.mark.asyncio
async def test_gateway_without_provider_keeps_openai_names(server: _Server) -> None:
    """Without `provider='mistral'` a gateway host gets OpenAI's names (and the fake Mistral behind it 422s)."""
    ai = _ai(server, GATEWAY_URL)

    response = await ai.generate(model='openai/mistral-small-latest', prompt='salut', config={'seed': 7})

    assert response.finish_reason == FinishReason.FAILED
    assert server.bodies()[0]['seed'] == 7


def test_unknown_provider_is_refused() -> None:
    """`provider=` takes only `'mistral'`."""
    with pytest.raises(ValueError, match='mistral'):
        OpenAI(api_key=PLUGIN_KEY, provider=cast(Any, 'cohere'))


@pytest.mark.parametrize(
    ('base_url', 'provider'),
    [
        pytest.param(MISTRAL_URL, None, id='mistral-host'),
        pytest.param(GATEWAY_URL, 'mistral', id='provider'),
    ],
)
@pytest.mark.asyncio
async def test_mistral_without_api_key_uses_mistral_api_key_env(
    server: _Server, monkeypatch: pytest.MonkeyPatch, base_url: str, provider: Literal['mistral'] | None
) -> None:
    """With no `api_key=`, a Mistral target sends MISTRAL_API_KEY, never OPENAI_API_KEY."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-openai')
    monkeypatch.setenv('MISTRAL_API_KEY', 'mistral-env')
    ai = _ai(server, base_url, api_key=None, provider=provider)

    await ai.generate(model='openai/mistral-small-latest', prompt='salut')
    await ai.embed(embedder='openai/mistral-embed', content='croissant')

    assert [r.headers['authorization'] for r in server.requests] == ['Bearer mistral-env'] * 2


@pytest.mark.asyncio
async def test_mistral_explicit_api_key_beats_mistral_api_key_env(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`api_key=` wins over MISTRAL_API_KEY."""
    monkeypatch.setenv('MISTRAL_API_KEY', 'mistral-env')
    ai = _ai(server, MISTRAL_URL)

    await ai.generate(model='openai/mistral-small-latest', prompt='salut')

    assert server.requests[0].headers['authorization'] == f'Bearer {PLUGIN_KEY}'


@pytest.mark.asyncio
async def test_mistral_without_any_key_does_not_send_openai_api_key(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With only OPENAI_API_KEY set, a Mistral call fails naming MISTRAL_API_KEY and sends nothing."""
    monkeypatch.setenv('OPENAI_API_KEY', 'sk-openai')
    monkeypatch.delenv('MISTRAL_API_KEY', raising=False)
    ai = _ai(server, MISTRAL_URL, api_key=None)

    response = await ai.generate(model='openai/mistral-small-latest', prompt='salut')

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert 'MISTRAL_API_KEY' in str(response.finish_message)
    assert server.requests == []


def _signed_thinking(text: str, signature: str) -> dict[str, Any]:
    return {'type': 'thinking', 'thinking': [{'type': 'text', 'text': text}], 'signature': signature}


@pytest.mark.asyncio
async def test_mistral_replays_each_reasoning_part_as_its_own_thinking_chunk(mistral: Genkit, server: _Server) -> None:
    """Each Mistral reasoning part goes back as its own chunk with its own signature; others are left out."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Bread for two?')]),
        Message(
            role=Role.MODEL,
            content=[
                _mistral_reasoning('First, count.', signature='sig-1'),
                Part.from_reasoning('Gemini thinks.', metadata={'thoughtSignature': 'gemini-sig'}),
                Part.from_reasoning('Unmarked.'),
                _mistral_reasoning('', signature=None),
                _mistral_reasoning('Unsigned middle.', signature=None),
                _mistral_reasoning('Then, answer.', signature='sig-2'),
                Part.from_text('Two loaves.'),
            ],
        ),
    ]

    await mistral.generate(model='openai/mistral-small-latest', messages=history, prompt='And three?')

    [assistant] = _assistant_messages(server.bodies()[0])
    assert assistant['content'] == [
        _signed_thinking('First, count.', 'sig-1'),
        {'type': 'thinking', 'thinking': [{'type': 'text', 'text': 'Unsigned middle.'}]},
        _signed_thinking('Then, answer.', 'sig-2'),
        {'type': 'text', 'text': 'Two loaves.'},
    ]


@pytest.mark.asyncio
async def test_mistral_sends_no_thinking_chunk_when_reasoning_parts_are_empty(mistral: Genkit, server: _Server) -> None:
    """Mistral reasoning parts with neither text nor a signature leave the turn as plain text."""
    history = [
        Message(role=Role.USER, content=[Part.from_text('Bread for two?')]),
        Message(role=Role.MODEL, content=[_mistral_reasoning('', signature=None), Part.from_text('Two loaves.')]),
    ]

    await mistral.generate(model='openai/mistral-small-latest', messages=history, prompt='And three?')

    [assistant] = _assistant_messages(server.bodies()[0])
    assert assistant['content'] == 'Two loaves.'


@pytest.mark.asyncio
async def test_mistral_reply_with_two_thinking_chunks_gives_two_reasoning_parts(
    mistral: Genkit, server: _Server
) -> None:
    """Two thinking chunks in a reply come back as two reasoning parts, each with its own signature, and replay so."""
    server.content = [
        _signed_thinking('First, count.', 'sig-1'),
        _signed_thinking('Then, answer.', 'sig-2'),
        {'type': 'text', 'text': 'Two loaves.'},
    ]

    first = await mistral.generate(model='openai/mistral-small-latest', prompt='Bread for two?')
    await mistral.generate(model='openai/mistral-small-latest', messages=first.messages, prompt='And three?')

    assert first.message is not None
    assert [(p.reasoning, p.metadata) for p in first.message.content if p.reasoning is not None] == [
        ('First, count.', {'reasoningSource': 'mistral', 'thoughtSignature': 'sig-1'}),
        ('Then, answer.', {'reasoningSource': 'mistral', 'thoughtSignature': 'sig-2'}),
    ]
    [assistant] = _assistant_messages(server.bodies()[-1])
    assert assistant['content'] == server.content


@pytest.mark.asyncio
async def test_mistral_stream_joins_a_thinking_chunks_fragments_into_one_part(mistral: Genkit, server: _Server) -> None:
    """A streamed thinking chunk's fragments end up as one reasoning part, with the signature, in the final message."""
    stream = mistral.generate_stream(
        model='openai/mistral-small-latest', prompt='Bread for two?', config=_REASONING_CONFIG
    )
    response = await stream.response

    assert response.message is not None
    assert [(p.reasoning, p.metadata) for p in response.message.content if p.reasoning is not None] == [
        (REASONING, {'reasoningSource': 'mistral', 'thoughtSignature': SIGNATURE})
    ]


@pytest.mark.asyncio
async def test_provider_mistral_without_base_url_goes_to_mistral(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`OpenAI(provider='mistral')` with no base_url sends to api.mistral.ai, not the SDK's api.openai.com."""
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://api.openai.com/v1')
    ai = _ai(server, None, provider='mistral')

    await ai.generate(model='openai/mistral-small-latest', prompt='salut', config={'seed': 7})

    [request] = server.requests
    assert str(request.url) == f'{MISTRAL_URL}/chat/completions'
    assert json.loads(request.content)['random_seed'] == 7


@pytest.mark.asyncio
async def test_provider_mistral_keeps_base_url_from_client_options(server: _Server) -> None:
    """A base_url given in client_options still wins over the Mistral default."""
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    ai = Genkit(
        plugins=[
            OpenAI(
                api_key=PLUGIN_KEY,
                client_options={'http_client': http_client, 'max_retries': 0, 'base_url': GATEWAY_URL},
                provider='mistral',
            )
        ]
    )

    await ai.generate(model='openai/mistral-small-latest', prompt='salut')

    assert server.requests[0].url.host == 'llm-gateway.example.com'
