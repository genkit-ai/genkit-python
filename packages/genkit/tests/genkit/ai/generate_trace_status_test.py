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

from genkit import Genkit, GenkitError, Interrupt, Message, ModelResponse, Operation, Part, PublicError
from genkit._core._model import ModelRequest
from genkit._core._telemetry._attrs import Attr, metadata_key
from genkit._core._telemetry._http import ActiveSpan
from genkit._core._typing import FinishReason, Role, ToolRequest
from genkit.exp import Genkit as ExpGenkit
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
async def test_parent_flow_raising_new_error_in_except_without_from_claims_failure_source(exporter) -> None:
    """Implicit `__context__` is not a cause, so a new error raised in `except:` without `from` claims the parent."""
    ai = Genkit()

    @ai.flow(name='check_allergens')
    async def check_allergens(_: str) -> str:
        raise ValueError('peanut flagged in pad thai')

    @ai.flow(name='place_order')
    async def place_order(x: str) -> str:
        try:
            return await check_allergens(x)
        except ValueError:
            raise RuntimeError('order rejected')  # noqa: B904

    with pytest.raises(RuntimeError, match='order rejected'):
        await place_order('table-7')

    spans = exporter.get_finished_spans()
    outer, inner = _span(spans, 'place_order'), _span(spans, 'check_allergens')
    assert (inner.attributes[Attr.STATE], inner.status_message) == ('error', 'peanut flagged in pad thai')
    assert (outer.attributes[Attr.STATE], outer.status_message) == ('error', 'order rejected')
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['place_order', 'check_allergens']


@pytest.mark.asyncio
async def test_parent_flow_wrapping_child_error_with_from_keeps_child_as_failure_source(exporter) -> None:
    """`raise ... from e` is an explicit cause, so the child stays the only failure source."""
    ai = Genkit()

    @ai.flow(name='check_allergens')
    async def check_allergens(_: str) -> str:
        raise ValueError('peanut flagged in pad thai')

    @ai.flow(name='place_order')
    async def place_order(x: str) -> str:
        try:
            return await check_allergens(x)
        except ValueError as e:
            raise RuntimeError('order rejected') from e

    with pytest.raises(RuntimeError, match='order rejected'):
        await place_order('table-7')

    spans = exporter.get_finished_spans()
    outer, inner = _span(spans, 'place_order'), _span(spans, 'check_allergens')
    assert (inner.attributes[Attr.STATE], inner.status_message) == ('error', 'peanut flagged in pad thai')
    assert (outer.attributes[Attr.STATE], outer.status_message) == ('error', 'order rejected')
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['check_allergens']


@pytest.mark.asyncio
async def test_flow_raising_after_interrupted_generate_marks_the_flow_as_failure_source(exporter) -> None:
    """A flow that raises its own error after generate paused on an Interrupt claims isFailureSource too."""
    ai = Genkit()
    _define_tools(ai)
    _fake_model(ai, [_calls_tool('approve_refund')])

    @ai.flow(name='refund_flow')
    async def refund_flow(_: str) -> str:
        resp = await ai.generate(model='fake', prompt='issue refund', tools=['approve_refund'])
        if resp.finish_reason == FinishReason.INTERRUPTED:
            raise ValueError('approval required')
        return resp.text

    @ai.flow(name='checkout_flow')
    async def checkout_flow(x: str) -> str:
        return await refund_flow(x)

    with pytest.raises(ValueError, match='approval required'):
        await checkout_flow('order-42')

    spans = exporter.get_finished_spans()
    checkout, refund, generate, tool = (
        _span(spans, 'checkout_flow'),
        _span(spans, 'refund_flow'),
        _span(spans, 'generate'),
        _span(spans, 'approve_refund'),
    )
    assert (generate.attributes[Attr.STATE], generate.status_message) == ('success', None)
    assert json.loads(str(tool.attributes[INTERRUPT_ATTR])) == {'reason': 'approve'}
    assert (refund.attributes[Attr.STATE], refund.status_message) == ('error', 'approval required')
    assert (checkout.attributes[Attr.STATE], checkout.status_message) == ('error', 'approval required')
    # Both the interrupted tool and refund_flow (which authored ValueError) are marked;
    # checkout_flow only re-raised refund_flow's marked ValueError, so it is not.
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['refund_flow', 'approve_refund']


@pytest.mark.asyncio
async def test_flow_raising_new_error_after_caught_child_failure_marks_flow_as_failure_source(exporter) -> None:
    """A flow that catches a child failure and later raises an unchained error claims isFailureSource."""
    ai = Genkit()

    @ai.flow(name='primary_kitchen')
    async def primary_kitchen(_: str) -> str:
        raise ValueError('oven offline')

    @ai.flow(name='dispatch_order')
    async def dispatch_order(x: str) -> str:
        kitchen_failed = False
        try:
            return await primary_kitchen(x)
        except ValueError:
            kitchen_failed = True
        if kitchen_failed:
            raise RuntimeError('no backup kitchen available')
        return 'ok'

    with pytest.raises(RuntimeError, match='no backup kitchen available'):
        await dispatch_order('table-7')

    spans = exporter.get_finished_spans()
    outer, inner = _span(spans, 'dispatch_order'), _span(spans, 'primary_kitchen')
    assert (inner.attributes[Attr.STATE], inner.status_message) == ('error', 'oven offline')
    assert (outer.attributes[Attr.STATE], outer.status_message) == ('error', 'no backup kitchen available')
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['dispatch_order', 'primary_kitchen']


@pytest.mark.asyncio
async def test_flow_raising_with_from_none_marks_flow_as_new_failure_source(exporter) -> None:
    """Explicit `raise ... from None` severs the child chain so the parent claims its new error."""
    ai = Genkit()

    @ai.flow(name='inner_flow')
    async def inner_flow(_: str) -> str:
        raise ValueError('bad kitchen input')

    @ai.flow(name='outer_flow')
    async def outer_flow(x: str) -> str:
        try:
            return await inner_flow(x)
        except ValueError:
            raise RuntimeError('clean break') from None

    with pytest.raises(RuntimeError, match='clean break'):
        await outer_flow('go')

    spans = exporter.get_finished_spans()
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['outer_flow', 'inner_flow']


@pytest.mark.asyncio
async def test_generate_operation_re_raising_boxed_failure_keeps_model_as_sole_failure_source(exporter) -> None:
    """generate_operation unboxing response.error does not make the enclosing flow reclaim isFailureSource."""
    ai = Genkit()

    async def start(_request: ModelRequest, _ctx: ModelContext) -> Operation[ModelResponse]:
        raise GenkitError(status='UNAVAILABLE', message='renderer quota exhausted')

    async def check(op: Operation[ModelResponse], _ctx: ModelContext) -> Operation[ModelResponse]:
        return op

    ai.define_background_model(name='video_renderer', start=start, check=check)

    @ai.flow(name='promo_flow')
    async def promo_flow(_: str) -> str:
        op = await ai.generate_operation(model='video_renderer', prompt='render promo')
        return op.id

    with pytest.raises(GenkitError, match='renderer quota exhausted'):
        await promo_flow('go')

    spans = exporter.get_finished_spans()
    flow = _span(spans, 'promo_flow')
    assert flow.attributes[Attr.STATE] == 'error'
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['video_renderer']


@pytest.mark.asyncio
async def test_prompt_agent_turn_re_raising_boxed_failure_keeps_model_as_sole_failure_source(exporter) -> None:
    """A prompt-backed agent turn unboxing response.error turns runTurn-1 red without reclaiming isFailureSource."""
    ai = ExpGenkit()
    _fake_model(ai, [RuntimeError('kitchen printer jammed')])
    waiter = ai.define_agent(name='waiter', model='fake', system='Take the order.')

    reply = await waiter.chat().send('two espressos')

    assert reply.error is not None
    spans = exporter.get_finished_spans()
    turn_span = _span(spans, 'runTurn-1')
    assert turn_span.attributes[Attr.STATE] == 'error'
    assert [s.name for s in spans if s.attributes.get(Attr.IS_FAILURE_SOURCE)] == ['fake']
