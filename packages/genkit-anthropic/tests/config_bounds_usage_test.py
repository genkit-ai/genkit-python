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

"""AnthropicConfig sampling bounds and cache usage mapping."""

import json
from typing import Any, Literal
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genkit_anthropic import Anthropic, AnthropicConfig
from genkit_anthropic._models import AnthropicModel
from genkit_anthropic._utils import build_cache_usage
from pydantic import ValidationError

from genkit import Genkit, GenkitError, Message, Part, Role
from genkit.model import GenerationUsage, ModelRequest


@pytest.mark.parametrize(
    ('field', 'value'),
    [('temperature', 1.5), ('temperature', -0.1), ('top_p', 1.1), ('top_p', -0.5)],
)
def test_out_of_range_sampling_values_fail_validation(field: str, value: float) -> None:
    """Anthropic accepts temperature and top_p in [0, 1]; anything else fails before a request."""
    with pytest.raises(ValidationError, match=f'{field} must be between 0.0 and 1.0'):
        AnthropicConfig.model_validate({field: value})


@pytest.mark.parametrize('value', [0.0, 0.7, 1.0])
def test_in_range_sampling_values_pass(value: float) -> None:
    """Bounds are inclusive."""
    config = AnthropicConfig(temperature=value, top_p=value)

    assert (config.temperature, config.top_p) == (value, value)


def test_bounds_show_in_the_dev_ui_schema() -> None:
    """The JSON schema the Dev UI reads advertises the range."""
    props = AnthropicConfig.model_json_schema()['properties']

    assert (props['temperature']['minimum'], props['temperature']['maximum']) == (0.0, 1.0)
    assert (props['topP']['minimum'], props['topP']['maximum']) == (0.0, 1.0)


def test_cache_tokens_map_to_inclusive_input_and_shared_usage_fields() -> None:
    """Cache reads and writes are added into input_tokens/total_tokens and fill shared cache fields."""
    usage = build_cache_usage(
        input_tokens=40,
        output_tokens=20,
        basic_usage=GenerationUsage(),
        cache_creation_input_tokens=10,
        cache_read_input_tokens=50,
    )

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (100, 20, 120)
    assert (usage.cached_content_tokens, usage.cache_write_tokens) == (50, 10)
    assert usage.custom == {'cache_creation_input_tokens': 10, 'cache_read_input_tokens': 50}


def test_no_cache_tokens_leaves_the_fields_unset() -> None:
    """A response without cache activity reports None, not 0."""
    usage = build_cache_usage(input_tokens=40, output_tokens=20, basic_usage=GenerationUsage())

    assert (usage.input_tokens, usage.output_tokens, usage.total_tokens) == (40, 20, 60)
    assert (usage.cached_content_tokens, usage.cache_write_tokens, usage.custom) == (None, None, None)


@pytest.mark.asyncio
async def test_generate_rejects_out_of_range_sampling_before_calling_client() -> None:
    """AnthropicModel.generate raises INVALID_ARGUMENT before sending the request."""
    mock_client = MagicMock()
    mock_client.messages.create = AsyncMock()
    model = AnthropicModel(model_name='claude-sonnet-4-5', client=mock_client)
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])],
        config={'temperature': 1.4},
    )

    with pytest.raises(GenkitError, match='temperature must be between 0.0 and 1.0') as exc_info:
        await model.generate(request)

    assert exc_info.value.status == 'INVALID_ARGUMENT'
    mock_client.messages.create.assert_not_called()


def _build_genkit_with_usage(
    *,
    start_usage: dict[str, int],
    delta_usage: dict[str, int],
    final_usage: dict[str, int],
    text: str = 'Smoked Salmon Tartine',
) -> Genkit:
    """Wire a Genkit Anthropic plugin against a mock HTTP transport serving JSON and SSE."""

    def handler(request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        if body.get('stream'):
            events = [
                (
                    'message_start',
                    {
                        'type': 'message_start',
                        'message': {
                            'id': 'msg_1',
                            'type': 'message',
                            'role': 'assistant',
                            'model': 'claude-sonnet-4-6',
                            'content': [],
                            'stop_reason': None,
                            'stop_sequence': None,
                            'usage': start_usage,
                        },
                    },
                ),
                (
                    'content_block_start',
                    {'type': 'content_block_start', 'index': 0, 'content_block': {'type': 'text', 'text': ''}},
                ),
                (
                    'content_block_delta',
                    {'type': 'content_block_delta', 'index': 0, 'delta': {'type': 'text_delta', 'text': text}},
                ),
                ('content_block_stop', {'type': 'content_block_stop', 'index': 0}),
                (
                    'message_delta',
                    {
                        'type': 'message_delta',
                        'delta': {'stop_reason': 'end_turn', 'stop_sequence': None},
                        'usage': delta_usage,
                    },
                ),
                ('message_stop', {'type': 'message_stop'}),
            ]
            sse = ''.join(f'event: {name}\ndata: {json.dumps(data)}\n\n' for name, data in events)
            return httpx.Response(200, text=sse, headers={'content-type': 'text/event-stream'})

        return httpx.Response(
            200,
            json={
                'id': 'msg_1',
                'type': 'message',
                'role': 'assistant',
                'model': 'claude-sonnet-4-6',
                'content': [{'type': 'text', 'text': text}],
                'stop_reason': 'end_turn',
                'stop_sequence': None,
                'usage': final_usage,
            },
        )

    http_client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return Genkit(plugins=[Anthropic(api_key='test-key', client_options={'http_client': http_client})])


@pytest.mark.parametrize('api_version', ['stable', 'beta'])
@pytest.mark.asyncio
async def test_streaming_and_non_streaming_produce_identical_cache_usage(
    api_version: Literal['stable', 'beta'],
) -> None:
    """ai.generate and ai.generate_stream compute identical inclusive GenerationUsage."""
    ai = _build_genkit_with_usage(
        start_usage={
            'input_tokens': 40,
            'output_tokens': 1,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 1834,
        },
        delta_usage={'output_tokens': 20},
        final_usage={
            'input_tokens': 40,
            'output_tokens': 20,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 1834,
        },
    )

    non_stream = await ai.generate(
        model='anthropic/claude-sonnet-4-6',
        prompt='Suggest a dish.',
        config={'api_version': api_version},
    )
    stream_res = ai.generate_stream(
        model='anthropic/claude-sonnet-4-6',
        prompt='Suggest a dish.',
        config={'api_version': api_version},
    )
    chunks = [chunk.text async for chunk in stream_res.stream]
    streamed = await stream_res.response

    assert ''.join(chunks) == 'Smoked Salmon Tartine'
    assert non_stream.usage is not None
    assert streamed.usage == non_stream.usage
    assert (non_stream.usage.input_tokens, non_stream.usage.output_tokens, non_stream.usage.total_tokens) == (
        1884,
        20,
        1904,
    )
    assert (non_stream.usage.cached_content_tokens, non_stream.usage.cache_write_tokens) == (1834, 10)
    assert non_stream.usage.custom == {'cache_creation_input_tokens': 10, 'cache_read_input_tokens': 1834}


@pytest.mark.asyncio
async def test_streaming_message_delta_cumulative_cache_usage_updates_totals() -> None:
    """Cumulative input/cache counts emitted on message_delta produce the same inclusive usage."""
    ai = _build_genkit_with_usage(
        start_usage={
            'input_tokens': 25,
            'output_tokens': 1,
            'cache_creation_input_tokens': 5,
            'cache_read_input_tokens': 1000,
        },
        delta_usage={
            'input_tokens': 40,
            'output_tokens': 20,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 1834,
        },
        final_usage={
            'input_tokens': 40,
            'output_tokens': 20,
            'cache_creation_input_tokens': 10,
            'cache_read_input_tokens': 1834,
        },
    )

    non_stream = await ai.generate(model='anthropic/claude-sonnet-4-6', prompt='Suggest a dish.')
    streamed = await ai.generate_stream(model='anthropic/claude-sonnet-4-6', prompt='Suggest a dish.').response

    assert streamed.usage == non_stream.usage
    assert streamed.usage is not None
    assert (streamed.usage.input_tokens, streamed.usage.output_tokens, streamed.usage.total_tokens) == (1884, 20, 1904)
    assert (streamed.usage.cached_content_tokens, streamed.usage.cache_write_tokens) == (1834, 10)


@pytest.mark.asyncio
async def test_streaming_and_non_streaming_without_cache_leave_cache_fields_unset() -> None:
    """Uncached calls leave cached_content_tokens, cache_write_tokens, and custom unset in both modes."""
    ai = _build_genkit_with_usage(
        start_usage={'input_tokens': 40, 'output_tokens': 1},
        delta_usage={'output_tokens': 20},
        final_usage={'input_tokens': 40, 'output_tokens': 20},
    )

    non_stream = await ai.generate(model='anthropic/claude-sonnet-4-6', prompt='Suggest a dish.')
    streamed = await ai.generate_stream(model='anthropic/claude-sonnet-4-6', prompt='Suggest a dish.').response

    assert streamed.usage == non_stream.usage
    assert streamed.usage is not None
    assert (streamed.usage.input_tokens, streamed.usage.output_tokens, streamed.usage.total_tokens) == (40, 20, 60)
    assert (streamed.usage.cached_content_tokens, streamed.usage.cache_write_tokens, streamed.usage.custom) == (
        None,
        None,
        None,
    )
