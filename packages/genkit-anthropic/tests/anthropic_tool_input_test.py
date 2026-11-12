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

"""Tools whose input isn't an object, driven through ``ai.generate``.

Claude only accepts object tool inputs, so ``city: str`` is declared as
``{'input': <schema>}``. Each test runs the real Anthropic SDK against a fake
HTTP transport that answers with a tool call, then text.
"""

import json
from typing import Any

import httpx
import pytest
from genkit_anthropic import Anthropic
from pydantic import BaseModel

from genkit import Genkit, Message, Part, Role
from genkit.model import ToolRequest, ToolResponse

MODEL = 'anthropic/claude-sonnet-4-6'


def _message(content: list[dict[str, Any]], stop_reason: str) -> dict[str, Any]:
    return {
        'id': 'msg_1',
        'type': 'message',
        'role': 'assistant',
        'model': 'claude-sonnet-4-6',
        'content': content,
        'stop_reason': stop_reason,
        'stop_sequence': None,
        'usage': {'input_tokens': 1, 'output_tokens': 1},
    }


def _tool_use(name: str, tool_input: dict[str, Any]) -> dict[str, Any]:
    return {'type': 'tool_use', 'id': 'toolu_1', 'name': name, 'input': tool_input}


def _sse(content: list[dict[str, Any]], stop_reason: str) -> str:
    events: list[dict[str, Any]] = [
        {'type': 'message_start', 'message': {**_message([], stop_reason), 'stop_reason': None}}
    ]
    for index, block in enumerate(content):
        if block['type'] == 'tool_use':
            events += [
                {'type': 'content_block_start', 'index': index, 'content_block': {**block, 'input': {}}},
                {
                    'type': 'content_block_delta',
                    'index': index,
                    'delta': {'type': 'input_json_delta', 'partial_json': json.dumps(block['input'])},
                },
            ]
        else:
            events += [
                {'type': 'content_block_start', 'index': index, 'content_block': {'type': 'text', 'text': ''}},
                {'type': 'content_block_delta', 'index': index, 'delta': {'type': 'text_delta', 'text': block['text']}},
            ]
        events.append({'type': 'content_block_stop', 'index': index})
    events += [
        {
            'type': 'message_delta',
            'delta': {'stop_reason': stop_reason, 'stop_sequence': None},
            'usage': {'output_tokens': 1},
        },
        {'type': 'message_stop'},
    ]
    return ''.join(f'event: {e["type"]}\ndata: {json.dumps(e)}\n\n' for e in events)


class FakeClaudeApi:
    """Answers the first call with ``first_turn`` blocks and every later call with text."""

    def __init__(self, first_turn: list[dict[str, Any]] | None = None) -> None:
        self.bodies: list[dict[str, Any]] = []
        self._first_turn = first_turn

    def handler(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        self.bodies.append(body)
        if self._first_turn is not None and len(self.bodies) == 1:
            content, stop_reason = self._first_turn, 'tool_use'
        else:
            content, stop_reason = [{'type': 'text', 'text': 'done'}], 'end_turn'
        if body.get('stream'):
            return httpx.Response(200, text=_sse(content, stop_reason), headers={'content-type': 'text/event-stream'})
        return httpx.Response(200, json=_message(content, stop_reason))

    def tool_schema(self, name: str) -> dict[str, Any]:
        return next(t['input_schema'] for t in self.bodies[0]['tools'] if t['name'] == name)


def _genkit(api: FakeClaudeApi) -> Genkit:
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(api.handler))
    return Genkit(plugins=[Anthropic(api_key='key', client_options={'http_client': http_client})])


class BookingRequest(BaseModel):
    name: str
    party_size: int


@pytest.mark.asyncio
async def test_scalar_tool_reaches_claude_as_object_with_input_property() -> None:
    """A `city: str` tool is declared to Claude as a required `input` string."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    @ai.tool()
    async def get_weather(city: str) -> str:
        return 'sunny'

    await ai.generate(model=MODEL, prompt='weather?', tools=[get_weather])

    assert api.tool_schema('get_weather') == {
        'type': 'object',
        'properties': {'input': {'type': 'string'}},
        'required': ['input'],
    }


@pytest.mark.asyncio
async def test_list_arg_tool_reaches_claude_wrapped_like_scalar() -> None:
    """A `cities: list[str]` tool is declared under a required `input` too."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    @ai.tool()
    async def get_forecasts(cities: list[str]) -> str:
        return 'sunny'

    await ai.generate(model=MODEL, prompt='weather?', tools=[get_forecasts])

    schema = api.tool_schema('get_forecasts')
    assert schema['type'] == 'object'
    assert schema['properties']['input']['type'] == 'array'
    assert schema['required'] == ['input']


@pytest.mark.asyncio
async def test_optional_scalar_tool_input_is_not_required() -> None:
    """A `city: str | None` tool is declared under `input` with no `required`, so Claude may leave it out."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    @ai.tool()
    async def get_weather(city: str | None) -> str:
        return 'sunny'

    await ai.generate(model=MODEL, prompt='weather?', tools=[get_weather])

    schema = api.tool_schema('get_weather')
    assert schema['type'] == 'object'
    assert 'input' in schema['properties']
    assert 'required' not in schema


@pytest.mark.asyncio
async def test_scalar_tool_call_from_claude_runs_tool_with_bare_value() -> None:
    """Claude calls with `{'input': 'Paris'}` and the tool runs with `'Paris'`."""
    api = FakeClaudeApi(first_turn=[_tool_use('get_weather', {'input': 'Paris'})])
    ai = _genkit(api)
    seen: list[str] = []

    @ai.tool()
    async def get_weather(city: str) -> str:
        seen.append(city)
        return 'sunny'

    response = await ai.generate(model=MODEL, prompt='weather?', tools=[get_weather])

    assert seen == ['Paris']
    assert response.text == 'done'


@pytest.mark.asyncio
async def test_streamed_scalar_tool_call_runs_tool_with_bare_value() -> None:
    """Through `generate_stream`, Claude's `{'input': 'Paris'}` also runs the tool with `'Paris'`."""
    api = FakeClaudeApi(first_turn=[_tool_use('get_weather', {'input': 'Paris'})])
    ai = _genkit(api)
    seen: list[str] = []

    @ai.tool()
    async def get_weather(city: str) -> str:
        seen.append(city)
        return 'sunny'

    result = ai.generate_stream(model=MODEL, prompt='weather?', tools=[get_weather])
    async for _ in result.stream:
        pass
    response = await result.response

    assert seen == ['Paris']
    assert response.text == 'done'


@pytest.mark.asyncio
async def test_scalar_tool_history_is_sent_back_wrapped() -> None:
    """On turn 2, the earlier `get_weather('Paris')` call goes back to Claude as `{'input': 'Paris'}`."""
    api = FakeClaudeApi(first_turn=[_tool_use('get_weather', {'input': 'Paris'})])
    ai = _genkit(api)

    @ai.tool()
    async def get_weather(city: str) -> str:
        return 'sunny'

    await ai.generate(model=MODEL, prompt='weather?', tools=[get_weather])

    tool_use = next(b for m in api.bodies[1]['messages'] for b in m['content'] if b['type'] == 'tool_use')
    assert tool_use['input'] == {'input': 'Paris'}


@pytest.mark.asyncio
async def test_history_call_to_tool_not_in_request_is_sent_unchanged() -> None:
    """A past call to a tool this request doesn't pass goes back with its stored input as is."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    await ai.generate(
        model=MODEL,
        messages=[
            Message(role=Role.USER, content=[Part.from_text('weather?')]),
            Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(ref='toolu_1', name='get_weather', input='Paris'))],
            ),
            Message(
                role=Role.TOOL,
                content=[Part(tool_response=ToolResponse(ref='toolu_1', name='get_weather', output='sunny'))],
            ),
        ],
    )

    tool_use = next(b for m in api.bodies[0]['messages'] for b in m['content'] if b['type'] == 'tool_use')
    assert tool_use['input'] == 'Paris'


@pytest.mark.asyncio
async def test_object_tool_schema_and_input_pass_through_unchanged() -> None:
    """A Pydantic-arg tool is declared with its own object schema and runs with Claude's fields as is."""
    api = FakeClaudeApi(first_turn=[_tool_use('book_table', {'name': 'Ann', 'party_size': 2})])
    ai = _genkit(api)
    seen: list[BookingRequest] = []

    @ai.tool()
    async def book_table(req: BookingRequest) -> str:
        seen.append(req)
        return 'booked'

    await ai.generate(model=MODEL, prompt='book', tools=[book_table])

    schema = api.tool_schema('book_table')
    assert schema['type'] == 'object'
    assert 'input' not in schema['properties']
    assert set(schema['properties']) == {'name', 'party_size'}
    assert seen == [BookingRequest(name='Ann', party_size=2)]
    tool_use = next(b for m in api.bodies[1]['messages'] for b in m['content'] if b['type'] == 'tool_use')
    assert tool_use['input'] == {'name': 'Ann', 'party_size': 2}


@pytest.mark.asyncio
async def test_zero_arg_tool_still_gets_empty_object_schema() -> None:
    """`get_specials()` is declared as an empty object, not wrapped under `input`."""
    api = FakeClaudeApi()
    ai = _genkit(api)

    @ai.tool()
    async def get_specials() -> str:
        return 'risotto'

    await ai.generate(model=MODEL, prompt='specials?', tools=[get_specials])

    assert api.tool_schema('get_specials') == {'type': 'object', 'properties': {}}
