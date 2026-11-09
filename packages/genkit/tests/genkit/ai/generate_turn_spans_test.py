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

"""The trace of a tool-calling ai.generate: one generate span per model turn."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from genkit import Genkit, Message, ModelResponse
from genkit._core._error import GenkitError
from genkit._core._telemetry._attrs import Attr
from genkit._core._telemetry._http import ActiveSpan
from genkit._core._typing import FinishReason, Role
from genkit.model import ModelContext, ModelRequest


def _tool_call(*names: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message.model_validate({
            'role': 'model',
            'content': [
                {'toolRequest': {'ref': f'r{i}', 'name': name, 'input': {'city': 'Paris'}}}
                for i, name in enumerate(names)
            ],
        }),
    )


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message.model_validate({'role': 'model', 'content': [{'text': text}]}),
    )


def _fake_model(ai: Genkit, turns: list[ModelResponse | Exception]) -> None:
    script = list(turns)

    async def fake(_: ModelRequest, __: ModelContext) -> ModelResponse:
        turn = script.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    ai.define_model(name='fake', fn=fake)


def _weather_tools(ai: Genkit) -> None:
    @ai.tool(name='get_weather')
    async def get_weather(_: dict) -> str:
        return '21C'

    @ai.tool(name='get_time')
    async def get_time(_: dict) -> str:
        return 'noon'


def _tree(spans: Sequence[ActiveSpan]) -> list[tuple[str, str | None, int]]:
    """(name, parent name, depth) in start order; the outer generate is depth 0."""
    by_id = {s.span_id: s for s in spans}

    def depth(s: ActiveSpan) -> int:
        d = 0
        while s.parent_span_id in by_id:
            s = by_id[s.parent_span_id]
            d += 1
        return d

    ordered = sorted(spans, key=lambda s: s.start_time_unix_nano)
    return [(s.name, by_id[s.parent_span_id].name if s.parent_span_id in by_id else None, depth(s)) for s in ordered]


def _generate_spans(spans: Sequence[ActiveSpan]) -> list[ActiveSpan]:
    """Generate spans outermost first."""
    found = [s for s in spans if s.name == 'generate']
    for s in found:
        assert s.attributes[Attr.TYPE] == 'util'
    return sorted(found, key=lambda s: s.start_time_unix_nano)


@pytest.mark.asyncio
async def test_generate_with_one_tool_call_nests_the_second_model_call_in_its_own_generate(exporter) -> None:
    """A model that calls one tool then answers shows turn 2 as a generate nested under the first."""
    ai = Genkit()
    _weather_tools(ai)
    _fake_model(ai, [_tool_call('get_weather'), _answer('done')])

    response = await ai.generate(model='fake', prompt='weather in Paris', tools=['get_weather'])

    assert response.text == 'done'
    assert response.finish_reason == FinishReason.STOP
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    assert _tree(exporter.get_finished_spans()) == [
        ('generate', None, 0),
        ('fake', 'generate', 1),
        ('get_weather', 'generate', 1),
        ('generate', 'generate', 1),
        ('fake', 'generate', 2),
    ]
    outer, inner = _generate_spans(exporter.get_finished_spans())
    assert [s.attributes[Attr.STATE] for s in (outer, inner)] == ['success', 'success']


@pytest.mark.asyncio
async def test_generate_with_two_tools_in_one_turn_lists_both_tools_before_the_nested_generate(exporter) -> None:
    """Two tools called in one turn both sit beside turn 1's model call, before turn 2's generate."""
    ai = Genkit()
    _weather_tools(ai)
    _fake_model(ai, [_tool_call('get_weather', 'get_time'), _answer('done')])

    response = await ai.generate(model='fake', prompt='weather in Paris', tools=['get_weather', 'get_time'])

    assert response.text == 'done'
    assert response.finish_reason == FinishReason.STOP
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL, Role.MODEL]
    tree = _tree(exporter.get_finished_spans())
    assert tree[:2] == [('generate', None, 0), ('fake', 'generate', 1)]
    assert sorted(tree[2:4]) == [('get_time', 'generate', 1), ('get_weather', 'generate', 1)]
    assert tree[4:] == [('generate', 'generate', 1), ('fake', 'generate', 2)]
    outer, inner = _generate_spans(exporter.get_finished_spans())
    assert [s.attributes[Attr.STATE] for s in (outer, inner)] == ['success', 'success']


@pytest.mark.asyncio
async def test_generate_with_three_turns_nests_one_generate_per_turn(exporter) -> None:
    """Every further tool turn nests one more generate under the previous turn's generate."""
    ai = Genkit()
    _weather_tools(ai)
    _fake_model(ai, [_tool_call('get_weather'), _tool_call('get_time'), _answer('done')])

    response = await ai.generate(model='fake', prompt='weather in Paris', tools=['get_weather', 'get_time'])

    assert response.text == 'done'
    assert response.finish_reason == FinishReason.STOP
    assert [m.role for m in response.messages] == [
        Role.USER,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
        Role.TOOL,
        Role.MODEL,
    ]
    assert _tree(exporter.get_finished_spans()) == [
        ('generate', None, 0),
        ('fake', 'generate', 1),
        ('get_weather', 'generate', 1),
        ('generate', 'generate', 1),
        ('fake', 'generate', 2),
        ('get_time', 'generate', 2),
        ('generate', 'generate', 2),
        ('fake', 'generate', 3),
    ]
    spans = _generate_spans(exporter.get_finished_spans())
    assert [s.attributes[Attr.STATE] for s in spans] == ['success', 'success', 'success']


@pytest.mark.asyncio
async def test_generate_hitting_max_turns_shows_one_generate_per_turn_it_ran(exporter) -> None:
    """Hitting max_turns leaves one generate per model call made, and no generate for the turn it refused."""
    ai = Genkit()
    _weather_tools(ai)
    _fake_model(ai, [_tool_call('get_weather'), _tool_call('get_weather')])

    response = await ai.generate(model='fake', prompt='weather in Paris', tools=['get_weather'], max_turns=1)

    assert response.text == ''
    assert response.finish_reason == FinishReason.ABORTED
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tree(exporter.get_finished_spans()) == [
        ('generate', None, 0),
        ('fake', 'generate', 1),
        ('get_weather', 'generate', 1),
        ('generate', 'generate', 1),
        ('fake', 'generate', 2),
    ]
    outer, inner = _generate_spans(exporter.get_finished_spans())
    assert [s.attributes[Attr.STATE] for s in (outer, inner)] == ['error', 'error']


@pytest.mark.asyncio
async def test_model_fails_on_second_turn_inner_generate_is_red_and_outer_shows_the_cause(exporter) -> None:
    """A model failure on turn 2 marks turn 2's generate failed and the outer generate failed too."""
    ai = Genkit()
    _weather_tools(ai)
    _fake_model(ai, [_tool_call('get_weather'), GenkitError(status='UNAVAILABLE', message='model is down')])

    response = await ai.generate(model='fake', prompt='weather in Paris', tools=['get_weather'])

    assert response.text == ''
    assert response.finish_reason == FinishReason.FAILED
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL, Role.TOOL]
    assert _tree(exporter.get_finished_spans()) == [
        ('generate', None, 0),
        ('fake', 'generate', 1),
        ('get_weather', 'generate', 1),
        ('generate', 'generate', 1),
        ('fake', 'generate', 2),
    ]
    outer, inner = _generate_spans(exporter.get_finished_spans())
    assert [s.attributes[Attr.STATE] for s in (outer, inner)] == ['error', 'error']


@pytest.mark.asyncio
async def test_generate_without_tools_has_a_single_generate_span(exporter) -> None:
    """A plain answer with no tool calls is one generate wrapping one model call."""
    ai = Genkit()
    _fake_model(ai, [_answer('hello')])

    response = await ai.generate(model='fake', prompt='hi')

    assert response.text == 'hello'
    assert response.finish_reason == FinishReason.STOP
    assert [m.role for m in response.messages] == [Role.USER, Role.MODEL]
    assert _tree(exporter.get_finished_spans()) == [
        ('generate', None, 0),
        ('fake', 'generate', 1),
    ]
    (outer,) = _generate_spans(exporter.get_finished_spans())
    assert outer.attributes[Attr.STATE] == 'success'
