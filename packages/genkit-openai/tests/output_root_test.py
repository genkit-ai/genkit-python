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

"""Tests for wrapping non-object output schemas for OpenAI and unwrapping the reply."""

import json
from enum import Enum
from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest
from genkit_openai import OpenAI
from genkit_openai._models._model import OpenAIModel, _unwrap_output
from genkit_openai._models._output_root import OutputStreamUnwrapper, has_object_root, unwrap_output_text

from genkit import Genkit, Message, Part, Role
from genkit.model import ModelRequest, OutputConfig, ToolRequest

ARRAY_SCHEMA: dict[str, Any] = {
    'type': 'array',
    'items': {'type': 'object', 'properties': {'name': {'type': 'string'}}, 'required': ['name']},
}


class Color(str, Enum):
    """A colour, which Pydantic emits as a string enum schema."""

    RED = 'red'
    BLUE = 'blue'


def _run(deltas: list[str]) -> str:
    unwrapper = OutputStreamUnwrapper()
    return ''.join(unwrapper.push(d) for d in deltas) + unwrapper.flush()


def _json_request(schema: dict[str, Any]) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Hi')])],
        output=OutputConfig(format='json', json_schema=schema),
    )


def _sent_schema(model: OpenAIModel, request: ModelRequest) -> dict[str, Any]:
    response_format = model._get_response_format(request)
    assert response_format is not None
    return response_format['json_schema']['schema']


# Unwrapping text


@pytest.mark.parametrize(
    ('text', 'expected'),
    [
        ('{"value":[{"name":"a"}]}', '[{"name":"a"}]'),
        ('{"value":42}', '42'),
        ('{"value":"red"}', '"red"'),
        ('{"value":0}', '0'),
        ('{"value":false}', 'false'),
        ('{"value":""}', '""'),
        ('{"value":null}', 'null'),
        ('{ "value" : [1, 2] }\n', '[1, 2]'),
        ('{"value":{"a":"}"}}', '{"a":"}"}'),
    ],
)
def test_unwrap_output_text_returns_the_value_as_sent(text: str, expected: str) -> None:
    """The wrapped value comes back as the text it was sent as, falsy values included."""
    assert unwrap_output_text(text) == expected


@pytest.mark.parametrize('text', ['[1,2]', '{"other":1}', '{"value":1,"extra":2}', 'not json', ''])
def test_unwrap_output_text_passes_through_anything_else(text: str) -> None:
    """Only exactly {"value": ...} is unwrapped; a bare value from a host that ignored the schema is kept."""
    assert unwrap_output_text(text) == text


def test_unwrap_output_keeps_reasoning_and_leaves_a_tool_call_preamble_alone() -> None:
    """Text is unwrapped and other parts kept, unless the reply calls a tool."""
    answer = _unwrap_output(
        Message(role=Role.MODEL, content=[Part.from_reasoning('thinking'), Part.from_text('{"value":[1]}')])
    )
    assert [p.reasoning for p in answer.content] == ['thinking', None]
    assert answer.content[1].text == '[1]'

    preamble = Message(
        role=Role.MODEL,
        content=[Part.from_text('{"value":1}'), Part(tool_request=ToolRequest(name='lookup', input={}))],
    )
    assert _unwrap_output(preamble) is preamble


# Streaming


FULL = '{"value":[{"name":"a"},{"name":"b"}]}'


def test_stream_unwrapper_gives_the_same_result_however_the_stream_is_split() -> None:
    """Every two-way split, and one character at a time, yield the bare value."""
    assert _run([FULL]) == '[{"name":"a"},{"name":"b"}]'
    assert _run(list(FULL)) == '[{"name":"a"},{"name":"b"}]'
    for i in range(1, len(FULL)):
        assert _run([FULL[:i], FULL[i:]]) == '[{"name":"a"},{"name":"b"}]', f'split at {i}'


def test_stream_unwrapper_handles_a_spaced_opening_split_anywhere() -> None:
    """Whitespace around the opening's tokens is dropped wherever the stream splits."""
    spaced = '{ "value" :  [1, 2] }\n'
    for i in range(1, len(spaced)):
        assert _run([spaced[:i], spaced[i:]]) == '[1, 2]', f'split at {i}'
    assert _run(list(spaced)) == '[1, 2]'


def test_stream_unwrapper_never_emits_the_closing_brace_early() -> None:
    """A trailing brace is held until it is known to belong to the value."""
    unwrapper = OutputStreamUnwrapper()
    assert unwrapper.push('{"value":{"a":1}') == '{"a":1'
    assert unwrapper.push('}') == ''
    assert unwrapper.flush() == '}'


@pytest.mark.parametrize('text', ['[1,2]', '{"valuable":1}', '{"other":1}', '42'])
def test_stream_unwrapper_passes_through_output_that_is_not_a_wrapper(text: str) -> None:
    """A reply that never matches the opening streams unchanged."""
    assert _run(list(text)) == text


def test_stream_unwrapper_pass_through_returns_held_text_and_stops_unwrapping() -> None:
    """Once a tool call arrives, held text is released and the rest streams as is."""
    unwrapper = OutputStreamUnwrapper()
    assert unwrapper.push('{"val') == ''
    assert unwrapper.pass_through() == '{"val'
    assert unwrapper.push('ue":1}') == 'ue":1}'
    assert unwrapper.flush() == ''


# The response format


def test_response_format_wraps_a_non_object_root_for_an_object_only_host() -> None:
    """The schema goes out under a required value property, closed for strict mode."""
    model = OpenAIModel(model='gpt-4o', client=MagicMock(), object_schema_root=True)
    schema = _sent_schema(model, _json_request(ARRAY_SCHEMA))
    assert schema['type'] == 'object'
    assert schema['required'] == ['value']
    assert schema['additionalProperties'] is False
    assert schema['properties']['value']['type'] == 'array'
    assert schema['properties']['value']['items']['additionalProperties'] is False


def test_response_format_keeps_defs_at_the_wrapper_root() -> None:
    """References into $defs still resolve once the schema is wrapped."""
    model = OpenAIModel(model='gpt-4o', client=MagicMock(), object_schema_root=True)
    node = {
        'type': 'object',
        'properties': {'children': {'type': 'array', 'items': {'$ref': '#/$defs/Node'}}},
        'required': ['children'],
    }
    schema = _sent_schema(
        model, _json_request({'type': 'array', 'items': {'$ref': '#/$defs/Node'}, '$defs': {'Node': node}})
    )
    assert schema['properties']['value'] == {'type': 'array', 'items': {'$ref': '#/$defs/Node'}}
    assert schema['$defs']['Node']['properties']['children']['items'] == {'$ref': '#/$defs/Node'}


@pytest.mark.parametrize(
    'schema',
    [
        {'type': 'object', 'properties': {'a': {'type': 'string'}}, 'required': ['a']},
        # A root $ref to an object, as Pydantic emits for a recursive model.
        {'$ref': '#/$defs/Node', '$defs': {'Node': {'type': 'object', 'properties': {}}}},
    ],
)
def test_response_format_sends_an_object_root_unwrapped(schema: dict[str, Any]) -> None:
    """An object root, directly or through a root $ref, is not wrapped."""
    model = OpenAIModel(model='gpt-4o', client=MagicMock(), object_schema_root=True)
    request = _json_request(schema)
    assert not model._wraps_output(request)
    assert 'value' not in _sent_schema(model, request).get('properties', {})


_OBJ: dict[str, Any] = {'type': 'object'}


@pytest.mark.parametrize(
    ('schema', 'expected'),
    [
        ({'type': 'object', 'properties': {}}, True),
        ({'properties': {'a': {'type': 'string'}}}, False),
        (ARRAY_SCHEMA, False),
        ({'type': 'string', 'enum': ['a', 'b']}, False),
        ({'anyOf': [_OBJ, {'type': 'string'}]}, False),
        ({'type': ['object', 'null']}, False),
        ({'$ref': '#/definitions/Foo', 'definitions': {'Foo': _OBJ}}, True),
        ({'$ref': '#/$defs/Foo', '$defs': {'Foo': _OBJ}}, True),
        ({'$schema': 'http://json-schema.org/draft-07/schema#', '$ref': '#/$defs/Foo', '$defs': {'Foo': _OBJ}}, True),
        ({'$ref': '#/$defs/A', '$defs': {'A': {'$ref': '#/$defs/B'}, 'B': _OBJ}}, True),
        # OpenAI does not resolve escaped names or allow keywords beside $ref.
        ({'$ref': '#/$defs/a~1b', '$defs': {'a/b': _OBJ}}, False),
        ({'$ref': '#/$defs/Foo', 'description': 'd', '$defs': {'Foo': _OBJ}}, False),
        ({'$ref': '#/$defs/A', '$defs': {'A': {'$ref': '#/$defs/B', 'title': 'A'}, 'B': _OBJ}}, False),
        ({'$ref': '#/$defs/Foo', '$defs': {'Foo': {'type': 'array'}}}, False),
        ({'$ref': '#/$defs/Missing', '$defs': {}}, False),
        ({'$ref': '#/$defs/A', '$defs': {'A': {'$ref': '#/$defs/A'}}}, False),
        ({'$ref': '#/$defs/Foo/properties/a', '$defs': {'Foo': _OBJ}}, False),
    ],
)
def test_has_object_root(schema: dict[str, Any], expected: bool) -> None:
    """Only an object, directly or through bare root refs to definitions, counts."""
    assert has_object_root(schema) is expected


def test_response_format_wraps_a_root_ref_with_sibling_keywords() -> None:
    """OpenAI rejects keywords beside a root $ref, so such a root is wrapped."""
    model = OpenAIModel(model='gpt-4o', client=MagicMock(), object_schema_root=True)
    schema = {'$ref': '#/$defs/Foo', 'description': 'A person', '$defs': {'Foo': {'type': 'object', 'properties': {}}}}
    request = _json_request(schema)
    assert model._wraps_output(request)
    # The SDK's strict pass inlines a $ref that has siblings.
    value = _sent_schema(model, request)['properties']['value']
    assert value['type'] == 'object'
    assert value['description'] == 'A person'


def test_response_format_wraps_a_root_ref_to_an_enum() -> None:
    """A root $ref is judged by its target, so an enum behind one is wrapped."""
    model = OpenAIModel(model='gpt-4o', client=MagicMock(), object_schema_root=True)
    schema = {'$ref': '#/$defs/Color', '$defs': {'Color': {'type': 'string', 'enum': ['red', 'blue']}}}
    sent = _sent_schema(model, _json_request(schema))
    assert sent['properties']['value'] == {'$ref': '#/$defs/Color'}
    assert sent['$defs'] == schema['$defs']


def test_response_format_is_unchanged_without_the_setting() -> None:
    """Other users of OpenAIModel (Model Garden) send the schema as before."""
    model = OpenAIModel(model='meta/llama', client=MagicMock())
    request = _json_request(ARRAY_SCHEMA)
    assert not model._wraps_output(request)
    assert _sent_schema(model, request)['type'] == 'array'


def test_deepseek_is_not_wrapped() -> None:
    """DeepSeek takes json_object with the schema in the prompt, so there is nothing to wrap."""
    model = OpenAIModel(model='deepseek-chat', client=MagicMock(), object_schema_root=True)
    request = _json_request(ARRAY_SCHEMA)
    assert not model._wraps_output(request)
    assert model._get_response_format(request) == {'type': 'json_object'}


# Through the plugin


def _completion(content: str) -> dict[str, Any]:
    return {
        'id': 'chatcmpl-1',
        'object': 'chat.completion',
        'created': 0,
        'model': 'gpt-4o',
        'choices': [{'index': 0, 'finish_reason': 'stop', 'message': {'role': 'assistant', 'content': content}}],
        'usage': {'prompt_tokens': 1, 'completion_tokens': 1, 'total_tokens': 2},
    }


def _delta(delta: dict[str, Any], finish_reason: str | None = None) -> str:
    return json.dumps({
        'id': 'chatcmpl-1',
        'object': 'chat.completion.chunk',
        'created': 0,
        'model': 'gpt-4o',
        'choices': [{'index': 0, 'delta': delta, 'finish_reason': finish_reason}],
    })


class _Server:
    """A fake OpenAI endpoint that answers every request the same way and records each body."""

    def __init__(self, content: str = '', frames: list[str] | None = None) -> None:
        self._content = content
        self._frames = frames
        self.bodies: list[dict[str, Any]] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if body.get('stream'):
            frames = self._frames or [_delta({'content': self._content}, 'stop')]
            sse = ''.join(f'data: {f}\n\n' for f in frames) + 'data: [DONE]\n\n'
            return httpx.Response(200, text=sse, headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json=_completion(self._content))

    def sent_schema(self) -> dict[str, Any]:
        return self.bodies[-1]['response_format']['json_schema']['schema']


def _genkit(server: _Server, **options: Any) -> Genkit:  # noqa: ANN401
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(server.handler))
    return Genkit(
        plugins=[OpenAI(api_key='sk-test', client_options={'http_client': http_client, 'max_retries': 0, **options})]
    )


@pytest.mark.asyncio
async def test_plugin_sends_a_wrapped_list_and_returns_the_bare_value() -> None:
    """A list schema goes out wrapped and the caller gets the list, with its text as sent."""
    server = _Server('{"value":[{"name":"a"},{"name":"b"}]}')
    response = await _genkit(server).generate(model='openai/gpt-4o', prompt='Two people.', output_schema=ARRAY_SCHEMA)

    assert server.sent_schema()['required'] == ['value']
    assert response.text == '[{"name":"a"},{"name":"b"}]'
    assert response.output == [{'name': 'a'}, {'name': 'b'}]


@pytest.mark.asyncio
@pytest.mark.parametrize(('schema', 'reply', 'expected'), [(int, '{"value":0}', 0), (bool, '{"value":false}', False)])
async def test_plugin_returns_a_falsy_scalar(schema: type, reply: str, expected: object) -> None:
    """0 and false come back as values, not as missing output."""
    response = await _genkit(_Server(reply)).generate(model='openai/gpt-4o', prompt='?', output_schema=schema)
    assert response.output == expected


@pytest.mark.asyncio
async def test_plugin_wraps_a_pydantic_enum() -> None:
    """An Enum output type is a string schema, so it is wrapped too."""
    server = _Server('{"value":"blue"}')
    response = await _genkit(server).generate(model='openai/gpt-4o', prompt='Sky?', output_schema=Color)

    assert server.sent_schema()['properties']['value']['enum'] == ['red', 'blue']
    assert response.output == Color.BLUE


@pytest.mark.asyncio
async def test_plugin_streams_unwrapped_chunks_and_an_unwrapped_message() -> None:
    """Chunks read as the caller's value and the final message matches them."""
    frames = [_delta({'content': FULL[i : i + 5]}) for i in range(0, len(FULL), 5)]
    frames.append(_delta({}, 'stop'))
    server = _Server(frames=frames)

    stream = _genkit(server).generate_stream(model='openai/gpt-4o', prompt='Two people.', output_schema=ARRAY_SCHEMA)
    streamed = ''.join([chunk.text async for chunk in stream.stream])
    response = await stream.response

    assert server.sent_schema()['required'] == ['value']
    assert streamed == '[{"name":"a"},{"name":"b"}]'
    assert response.text == '[{"name":"a"},{"name":"b"}]'
    assert response.output == [{'name': 'a'}, {'name': 'b'}]


@pytest.mark.asyncio
async def test_plugin_streams_a_tool_call_preamble_unchanged() -> None:
    """Text before a tool call is not the output, so it streams and stays as sent."""
    tool_call = {
        'index': 0,
        'id': 'call_1',
        'type': 'function',
        'function': {'name': 'lookup', 'arguments': '{"dish":"pho"}'},
    }
    server = _Server(
        frames=[_delta({'content': '{"val'}), _delta({'tool_calls': [tool_call]}), _delta({}, 'tool_calls')]
    )
    ai = _genkit(server)

    @ai.tool(name='lookup')
    async def lookup(dish: str) -> str:
        return dish

    stream = ai.generate_stream(
        model='openai/gpt-4o',
        prompt='Two people.',
        output_schema=ARRAY_SCHEMA,
        tools=['lookup'],
        return_tool_requests=True,
    )
    streamed = ''.join([chunk.text async for chunk in stream.stream])
    response = await stream.response

    assert streamed == '{"val'
    assert response.message is not None
    assert response.message.content[0].text == '{"val'
    assert [p.tool_request.name for p in response.tool_requests if p.tool_request] == ['lookup']


@pytest.mark.asyncio
async def test_plugin_sends_an_object_root_unchanged() -> None:
    """Object schemas send the same request as before."""
    schema = {'type': 'object', 'properties': {'a': {'type': 'string'}}, 'required': ['a']}
    server = _Server('{"a":"x"}')
    response = await _genkit(server).generate(model='openai/gpt-4o', prompt='An object.', output_schema=schema)

    assert 'value' not in server.sent_schema()['properties']
    assert response.output == {'a': 'x'}


@pytest.mark.asyncio
async def test_plugin_pointed_at_another_host_still_wraps() -> None:
    """Gateways in front of OpenAI reject the same roots, so base_url doesn't turn wrapping off."""
    server = _Server('{"value":[{"name":"a"}]}')
    ai = _genkit(server, base_url='https://gateway.example.com/v1')
    response = await ai.generate(model='openai/gpt-4o', prompt='One person.', output_schema=ARRAY_SCHEMA)

    assert server.sent_schema()['required'] == ['value']
    assert response.output == [{'name': 'a'}]
