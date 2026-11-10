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

"""What the Dev UI trace shows when a tool or model fails, or a tool interrupts, inside ai.generate."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest

from genkit import Genkit, Interrupt, Message, ModelResponse, Part, PublicError
from genkit._core._model import ModelRequest
from genkit._core._telemetry._attrs import Attr, metadata_key
from genkit._core._telemetry._http import ActiveSpan
from genkit._core._typing import FinishReason, Role, ToolRequest
from genkit.model import ModelContext

INTERRUPT_ATTR = metadata_key('interrupt')


def _span(spans: Sequence[ActiveSpan], name: str) -> ActiveSpan:
    matches = [s for s in spans if s.name == name]
    assert len(matches) == 1, f'expected one span named {name!r} in {[s.name for s in spans]}'
    return matches[0]


def _generate_spans(spans: Sequence[ActiveSpan]) -> list[ActiveSpan]:
    """The generate span for the call plus one nested generate span per follow-up turn."""
    return [s for s in spans if s.name == 'generate']


def _calls_tool(name: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name=name, ref=f'ref-{name}', input={}))],
        ),
    )


def _says(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


def _fake_model(ai: Genkit, turns: list[ModelResponse | Exception]) -> None:
    """Model `fake` plays `turns` in order; an exception in the list is raised on that turn."""

    async def model(_request: ModelRequest, _ctx: ModelContext) -> ModelResponse:
        turn = turns.pop(0)
        if isinstance(turn, Exception):
            raise turn
        return turn

    ai.define_model(name='fake', fn=model)


def _define_tools(ai: Genkit) -> None:
    @ai.tool(name='get_weather')
    async def get_weather() -> str:
        return '72F'

    @ai.tool(name='explode')
    async def explode() -> str:
        raise RuntimeError('tool exploded')

    @ai.tool(name='find_order')
    async def find_order() -> str:
        raise PublicError('NOT_FOUND', 'no such order')

    @ai.tool(name='approve_refund')
    async def approve_refund() -> str:
        raise Interrupt({'reason': 'approve'})

    ai.define_interrupt('ask_user')


async def _run_flow(ai: Genkit, *, tools: list[str]) -> tuple[ModelResponse, str]:
    seen: list[ModelResponse] = []

    @ai.flow(name='trip_flow')
    async def trip_flow(_: str) -> str:
        resp = await ai.generate(model='fake', prompt='plan a trip', tools=tools)
        seen.append(resp)
        return str(resp.finish_reason)

    out = await trip_flow('go')
    return seen[0], out


@pytest.mark.asyncio
async def test_tool_raises_in_generate_flow_stays_green_and_generate_span_shows_the_cause(exporter) -> None:
    """A tool that raises in ai.generate puts its message on the generate span; the flow that returned stays green."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('explode')])

    resp, out = await _run_flow(ai, tools=['explode'])

    spans = exporter.get_finished_spans()
    flow, generate = _span(spans, 'trip_flow'), _span(spans, 'generate')
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert (generate.attributes[Attr.STATE], generate.attributes[Attr.ERROR], generate.status_message) == (
        'error',
        'tool exploded',
        'tool exploded',
    )
    assert Attr.IS_FAILURE_SOURCE not in generate.attributes
    assert resp.finish_reason == FinishReason.FAILED
    assert resp.error is not None and resp.error.message == 'internal error'
    assert out == 'failed'


@pytest.mark.asyncio
async def test_tool_raises_after_a_completed_tool_round_generate_span_shows_the_cause(exporter) -> None:
    """A tool that raises on the second turn shows its message on that turn's generate span and the outer one."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('get_weather'), _calls_tool('explode')])

    resp, out = await _run_flow(ai, tools=['get_weather', 'explode'])

    spans = exporter.get_finished_spans()
    flow, generates = _span(spans, 'trip_flow'), _generate_spans(spans)
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert [(g.attributes[Attr.STATE], g.attributes[Attr.ERROR], g.status_message) for g in generates] == [
        ('error', 'tool exploded', 'tool exploded')
    ] * 2
    assert all(Attr.IS_FAILURE_SOURCE not in g.attributes for g in generates)
    assert resp.finish_reason == FinishReason.FAILED
    assert resp.error is not None and resp.error.message == 'internal error'
    assert out == 'failed'


@pytest.mark.asyncio
async def test_model_raises_in_generate_generate_span_shows_the_model_error(exporter) -> None:
    """A model that raises shows its message on the generate span; the model span is where it started."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [RuntimeError('fake model failure')])

    resp, out = await _run_flow(ai, tools=[])

    spans = exporter.get_finished_spans()
    flow, generate, model = _span(spans, 'trip_flow'), _span(spans, 'generate'), _span(spans, 'fake')
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert (generate.attributes[Attr.STATE], generate.attributes[Attr.ERROR], generate.status_message) == (
        'error',
        'fake model failure',
        'fake model failure',
    )
    assert Attr.IS_FAILURE_SOURCE not in generate.attributes
    assert model.attributes.get(Attr.IS_FAILURE_SOURCE) is True
    assert resp.finish_reason == FinishReason.FAILED
    assert resp.error is not None and resp.error.message == 'internal error'
    assert out == 'failed'


@pytest.mark.asyncio
async def test_model_raises_after_a_completed_tool_round_generate_span_shows_the_model_error(exporter) -> None:
    """A model that raises after a tool round shows its message on that turn's generate span and the outer one."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('get_weather'), RuntimeError('fake model failure')])

    resp, out = await _run_flow(ai, tools=['get_weather'])

    spans = exporter.get_finished_spans()
    flow, generates = _span(spans, 'trip_flow'), _generate_spans(spans)
    failed_models = [s for s in spans if s.name == 'fake' and s.attributes[Attr.STATE] == 'error']
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert [(g.attributes[Attr.STATE], g.attributes[Attr.ERROR], g.status_message) for g in generates] == [
        ('error', 'fake model failure', 'fake model failure')
    ] * 2
    assert all(Attr.IS_FAILURE_SOURCE not in g.attributes for g in generates)
    assert [s.attributes.get(Attr.IS_FAILURE_SOURCE) for s in failed_models] == [True]
    assert resp.finish_reason == FinishReason.FAILED
    assert resp.error is not None and resp.error.message == 'internal error'
    assert out == 'failed'


@pytest.mark.asyncio
async def test_tool_raises_public_error_generate_span_and_response_show_the_same_message(exporter) -> None:
    """A tool's PublicError message is both on the generate span and on resp.error."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('find_order')])

    resp, out = await _run_flow(ai, tools=['find_order'])

    spans = exporter.get_finished_spans()
    flow, generate = _span(spans, 'trip_flow'), _span(spans, 'generate')
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert (generate.attributes[Attr.STATE], generate.attributes[Attr.ERROR], generate.status_message) == (
        'error',
        'no such order',
        'no such order',
    )
    assert Attr.IS_FAILURE_SOURCE not in generate.attributes
    assert resp.finish_reason == FinishReason.FAILED
    assert resp.error is not None and resp.error.message == 'no such order'
    assert out == 'failed'


@pytest.mark.asyncio
async def test_failing_tool_span_is_marked_as_where_the_failure_started(exporter) -> None:
    """Only the tool span that raised is marked as where the failure started, not generate or the flow."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('explode')])

    resp, out = await _run_flow(ai, tools=['explode'])

    spans = exporter.get_finished_spans()
    flow, generate, tool = _span(spans, 'trip_flow'), _span(spans, 'generate'), _span(spans, 'explode')
    assert (tool.attributes[Attr.STATE], tool.status_message) == ('error', 'tool exploded')
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['explode']
    assert Attr.IS_FAILURE_SOURCE not in generate.attributes
    assert Attr.IS_FAILURE_SOURCE not in flow.attributes
    assert resp.finish_reason == FinishReason.FAILED
    assert resp.error is not None and resp.error.message == 'internal error'
    assert out == 'failed'


@pytest.mark.asyncio
async def test_flow_that_raises_is_red_and_marked_as_where_the_failure_started(exporter) -> None:
    """A flow that raises is red, carries its message, and is marked as where the failure started."""
    ai = Genkit()

    @ai.flow(name='boom_flow')
    async def boom_flow(_: str) -> str:
        raise RuntimeError('boom')

    with pytest.raises(RuntimeError, match='boom'):
        await boom_flow('go')

    flow = _span(exporter.get_finished_spans(), 'boom_flow')
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('error', 'boom')
    assert flow.status_code == 2
    assert flow.attributes.get(Attr.IS_FAILURE_SOURCE) is True
    assert flow.attributes.get(Attr.IS_ROOT) is True


@pytest.mark.asyncio
async def test_interrupted_tool_without_metadata_shows_as_interrupted(exporter) -> None:
    """An interrupt with no metadata still marks the tool span interrupted; generate and the flow stay green."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('ask_user')])

    resp, out = await _run_flow(ai, tools=['ask_user'])

    spans = exporter.get_finished_spans()
    flow, generate, tool = _span(spans, 'trip_flow'), _span(spans, 'generate'), _span(spans, 'ask_user')
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert (generate.attributes[Attr.STATE], generate.status_message) == ('success', None)
    assert (tool.attributes[Attr.STATE], tool.status_code) == ('error', 2)
    assert json.loads(str(tool.attributes[INTERRUPT_ATTR])) is True
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['ask_user']
    assert resp.finish_reason == FinishReason.INTERRUPTED
    assert resp.error is None
    assert out == 'interrupted'


@pytest.mark.asyncio
async def test_interrupted_tool_with_metadata_keeps_its_metadata_on_the_span(exporter) -> None:
    """Interrupt({'reason': 'approve'}) puts that metadata on the tool span; generate and the flow stay green."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('approve_refund')])

    resp, out = await _run_flow(ai, tools=['approve_refund'])

    spans = exporter.get_finished_spans()
    flow, generate, tool = _span(spans, 'trip_flow'), _span(spans, 'generate'), _span(spans, 'approve_refund')
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert (generate.attributes[Attr.STATE], generate.status_message) == ('success', None)
    assert (tool.attributes[Attr.STATE], tool.status_code) == ('error', 2)
    assert json.loads(str(tool.attributes[INTERRUPT_ATTR])) == {'reason': 'approve'}
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['approve_refund']
    assert resp.finish_reason == FinishReason.INTERRUPTED
    assert resp.error is None
    assert out == 'interrupted'


@pytest.mark.asyncio
async def test_successful_flow_marks_only_the_root_span_as_root(exporter) -> None:
    """A flow that calls a tool and answers marks only itself as the root, and nothing as a failure."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('get_weather'), _says('sunny')])

    resp, out = await _run_flow(ai, tools=['get_weather'])

    spans = exporter.get_finished_spans()
    flow, generates = _span(spans, 'trip_flow'), _generate_spans(spans)
    assert (flow.attributes[Attr.STATE], flow.status_message) == ('success', None)
    assert [(g.attributes[Attr.STATE], g.status_message) for g in generates] == [('success', None)] * 2
    assert [s.name for s in spans if s.attributes.get(Attr.IS_ROOT)] == ['trip_flow']
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == []
    assert resp.finish_reason == FinishReason.STOP
    assert resp.error is None
    assert out == 'stop'


@pytest.mark.asyncio
async def test_loop_failure_without_failing_child_marks_innermost_generate_as_failure_source(exporter) -> None:
    """When generate hits max_turns with no failing child, the innermost generate turn is the failure source."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('get_weather'), _calls_tool('get_weather')])

    @ai.flow(name='capped_flow')
    async def capped_flow(_: str) -> str:
        resp = await ai.generate(model='fake', prompt='plan a trip', tools=['get_weather'], max_turns=1)
        return str(resp.finish_reason)

    assert await capped_flow('go') == 'aborted'

    spans = exporter.get_finished_spans()
    generates = _generate_spans(spans)
    assert len(generates) == 2
    outer, inner = generates[0], generates[1]
    assert inner.parent_span_id == outer.span_id
    expected_msg = 'Exceeded maximum tool call iterations (1)'
    assert [(g.attributes[Attr.STATE], g.status_message) for g in generates] == [('error', expected_msg)] * 2
    assert [s.span_id for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == [inner.span_id]


@pytest.mark.asyncio
async def test_parent_flow_wrapping_child_error_without_from_keeps_child_as_failure_source(exporter) -> None:
    """Implicit exception chaining (`raise ...` inside `except:` without `from`) keeps the child as culprit."""
    ai = Genkit()

    @ai.flow(name='inner_flow')
    async def inner_flow(_: str) -> str:
        raise ValueError('bad kitchen input')

    @ai.flow(name='outer_flow')
    async def outer_flow(x: str) -> str:
        try:
            return await inner_flow(x)
        except ValueError:
            raise RuntimeError('outer wrapper')  # noqa: B904

    with pytest.raises(RuntimeError, match='outer wrapper'):
        await outer_flow('go')

    spans = exporter.get_finished_spans()
    outer, inner = _span(spans, 'outer_flow'), _span(spans, 'inner_flow')
    assert (inner.attributes[Attr.STATE], inner.status_message) == ('error', 'bad kitchen input')
    assert (outer.attributes[Attr.STATE], outer.status_message) == ('error', 'outer wrapper')
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['inner_flow']
