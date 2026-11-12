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

"""The TraceData the Dev UI poster sends to ``/api/traces``, and what the Traces tab reads from it."""

from __future__ import annotations

import json
from collections.abc import Generator
from typing import Any

import pytest

from genkit import Genkit, GenkitError, Interrupt, Message, ModelResponse, Part
from genkit._core._error import _StreamingCallbackError
from genkit._core._model import ModelRequest
from genkit._core._telemetry._http import (
    ActiveSpan,
    DirectHttpInstrumentation,
    collector_traces_url,
    encode_traces,
)
from genkit._core._typing import FinishReason, Role, ToolRequest
from genkit.model import ModelContext
from genkit.telemetry import SpanContext, configure_instrumentation, reset_instrumentation, run_in_new_span


class SnapshotSink:
    """Encodes at export time, like CollectorHttpSink, so the start export keeps its in-progress shape."""

    def __init__(self) -> None:
        self.posts: list[dict[str, Any]] = []

    def export_spans(self, spans: list[ActiveSpan]) -> None:
        self.posts.extend(json.loads(json.dumps(trace)) for trace in encode_traces(spans))

    def flush(self) -> None:
        return

    def shutdown(self) -> None:
        return

    def exports(self, name: str) -> list[dict[str, Any]]:
        """Every export of the span called ``name``, start first."""
        return [span for post in self.posts for span in post['spans'].values() if span['displayName'] == name]

    def final(self, name: str) -> dict[str, Any]:
        (*_, last) = self.exports(name)
        return last

    def trace_posts(self, name: str) -> list[dict[str, Any]]:
        return [post for post in self.posts if any(s['displayName'] == name for s in post['spans'].values())]


@pytest.fixture
def sink() -> Generator[SnapshotSink, None, None]:
    recorder = SnapshotSink()
    reset_instrumentation()
    configure_instrumentation(DirectHttpInstrumentation(recorder))  # type: ignore[arg-type]
    try:
        yield recorder
    finally:
        reset_instrumentation()


def exception_event(span: dict[str, Any]) -> dict[str, Any] | None:
    events = [e for e in span['timeEvents']['timeEvent'] if e['annotation']['description'] == 'exception']
    return events[0]['annotation']['attributes'] if events else None


def dev_ui_heading(span: dict[str, Any]) -> str:
    """The error heading genkit-ui's span details shows for a non-model span.

    Mirrors ``errorMessage`` in span-details.component.ts and
    ``isSpanInterrupted`` in utils/trace.ts.
    """
    exception = exception_event(span) or {}
    if exception.get('exception.message'):
        return str(exception['exception.message'])
    attrs = span['attributes']
    interrupted = attrs.get('genkit:metadata:subtype') == 'tool' and bool(attrs.get('genkit:metadata:interrupt'))
    return 'Interrupted' if interrupted else 'Error'


def calls_tool(name: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name=name, ref=f'ref-{name}', input={}))],
        ),
    )


def scripted_model(ai: Genkit, turns: list[ModelResponse]) -> None:
    async def model(_request: ModelRequest, _ctx: ModelContext) -> ModelResponse:
        return turns.pop(0)

    ai.define_model(name='kitchen-model', fn=model)


def test_traces_url_targets_api_traces() -> None:
    """Spans go to /api/traces; /api/otlp drops span events."""
    assert collector_traces_url('http://127.0.0.1:4033') == 'http://127.0.0.1:4033/api/traces'
    assert collector_traces_url('http://127.0.0.1:4033/') == 'http://127.0.0.1:4033/api/traces'


@pytest.mark.asyncio
async def test_successful_span_encodes_as_span_data(sink: SnapshotSink) -> None:
    """A finished span carries the SpanData fields the collector schema requires, in milliseconds."""

    async def plate_order(_span: SpanContext) -> str:
        return 'plated'

    await run_in_new_span('plateOrder', plate_order, input={'order': 1042})

    span = sink.final('plateOrder')
    assert set(span) == {
        'spanId',
        'traceId',
        'startTime',
        'endTime',
        'attributes',
        'displayName',
        'links',
        'instrumentationLibrary',
        'spanKind',
        'sameProcessAsParentSpan',
        'status',
        'timeEvents',
    }
    assert span['instrumentationLibrary'] == {'name': 'genkit-tracer', 'version': 'v1'}
    assert span['spanKind'] == 'INTERNAL'
    assert span['status'] == {'code': 1}
    assert span['timeEvents'] == {'timeEvent': []}
    assert span['attributes']['genkit:state'] == 'success'
    assert span['attributes']['genkit:output'] == '"plated"'
    # milliseconds since the epoch, not nanoseconds
    assert 1e12 < span['startTime'] <= span['endTime'] < 1e14


@pytest.mark.asyncio
async def test_start_export_is_in_progress_and_end_export_closes_the_trace(sink: SnapshotSink) -> None:
    """The start export has endTime 0 and no trace endTime; the end export sets both."""

    async def fire_ticket(_span: SpanContext) -> str:
        return 'fired'

    await run_in_new_span('fireTicket', fire_ticket)

    started, ended = sink.trace_posts('fireTicket')
    (start_span,) = started['spans'].values()
    assert start_span['endTime'] == 0
    assert start_span['status'] == {'code': 0}
    assert 'genkit:state' not in start_span['attributes']
    assert started['displayName'] == 'fireTicket'
    assert started['startTime'] == start_span['startTime']
    assert 'endTime' not in started

    (end_span,) = ended['spans'].values()
    assert end_span['spanId'] == start_span['spanId']
    assert ended['endTime'] == end_span['endTime'] > 0


@pytest.mark.asyncio
async def test_child_span_links_to_parent_and_leaves_trace_fields_to_the_root(sink: SnapshotSink) -> None:
    """A child carries parentSpanId and the root's traceId; only the root sets the trace's name and times."""

    async def check_pantry(_span: SpanContext) -> str:
        return 'stocked'

    async def place_order(_span: SpanContext) -> str:
        return await run_in_new_span('checkPantry', check_pantry)

    await run_in_new_span('placeOrder', place_order)

    order, pantry = sink.final('placeOrder'), sink.final('checkPantry')
    assert 'parentSpanId' not in order
    assert pantry['parentSpanId'] == order['spanId']
    assert pantry['traceId'] == order['traceId']
    for post in sink.trace_posts('checkPantry'):
        assert set(post) == {'traceId', 'spans'}


@pytest.mark.asyncio
async def test_attributes_keep_their_json_types(sink: SnapshotSink) -> None:
    """Attributes encode as a flat map: str, bool, int and float stay typed; structured input is a JSON string."""

    async def weigh_flour(span: SpanContext) -> None:
        span.set_metadata({'station': 'pastry'})

    await run_in_new_span(
        'weighFlour',
        weigh_flour,
        input={'grams': 500, 'sifted': True},
        attributes={'kitchen.open': True, 'kitchen.covers': 48, 'kitchen.load': 0.75, 'kitchen.nan': float('nan')},
    )

    attrs = sink.final('weighFlour')['attributes']
    assert attrs['kitchen.open'] is True
    assert attrs['kitchen.covers'] == 48
    assert attrs['kitchen.load'] == 0.75
    # NaN isn't JSON; a bare NaN would make the collector reject the whole trace.
    assert attrs['kitchen.nan'] == 'nan'
    assert attrs['genkit:metadata:station'] == 'pastry'
    assert json.loads(attrs['genkit:input']) == {'grams': 500, 'sifted': True}
    assert attrs['genkit:isRoot'] is True


@pytest.mark.asyncio
async def test_failed_span_records_exception_event_with_type_message_and_stacktrace(sink: SnapshotSink) -> None:
    """A raised error lands as an exception event the Traces tab reads for its heading and stack box."""

    async def check_allergens(_span: SpanContext) -> None:
        raise ValueError('allergen db unavailable')

    with pytest.raises(ValueError):
        await run_in_new_span('checkAllergens', check_allergens)

    start, end = sink.exports('checkAllergens')
    assert exception_event(start) is None
    assert end['status'] == {'code': 2, 'message': 'allergen db unavailable'}
    (event,) = end['timeEvents']['timeEvent']
    assert event['time'] == end['endTime']
    attrs = event['annotation']['attributes']
    assert attrs['exception.type'] == 'ValueError'
    assert attrs['exception.message'] == 'allergen db unavailable'
    assert attrs['exception.stacktrace'].startswith('Traceback (most recent call last):')
    assert 'in check_allergens' in attrs['exception.stacktrace']
    assert attrs['exception.stacktrace'].rstrip().endswith('ValueError: allergen db unavailable')
    assert dev_ui_heading(end) == 'allergen db unavailable'


@pytest.mark.asyncio
async def test_error_propagating_through_a_flow_fails_both_spans_with_the_message(sink: SnapshotSink) -> None:
    """A tool error that escapes the flow shows its message on the tool span and on the flow span."""
    ai = Genkit()

    @ai.tool()
    async def check_allergens(dish: str) -> str:
        raise ValueError('allergen db unavailable')

    @ai.flow()
    async def take_order(dish: str) -> str:
        return await check_allergens(dish)

    with pytest.raises(ValueError):
        await take_order('pad thai')

    tool, flow = sink.final('check_allergens'), sink.final('take_order')
    assert dev_ui_heading(tool) == dev_ui_heading(flow) == 'allergen db unavailable'
    assert tool['parentSpanId'] == flow['spanId']
    assert tool['attributes']['genkit:isFailureSource'] is True
    assert 'genkit:isFailureSource' not in flow['attributes']


@pytest.mark.asyncio
async def test_tool_stack_starts_at_the_user_frame(sink: SnapshotSink) -> None:
    """The stack box leads with the tool's own line, not the action runtime's frames."""
    ai = Genkit()

    @ai.tool()
    async def check_allergens(dish: str) -> str:
        raise ValueError('allergen db unavailable')

    with pytest.raises(ValueError):
        await check_allergens('pad thai')

    attrs = exception_event(sink.final('check_allergens'))
    assert attrs is not None
    frames = [line for line in attrs['exception.stacktrace'].splitlines() if line.strip().startswith('File ')]
    assert frames, attrs['exception.stacktrace']
    assert __file__ in frames[0] and 'in check_allergens' in frames[0]
    assert 'run_in_new_span' not in attrs['exception.stacktrace']


@pytest.mark.asyncio
async def test_genkit_error_message_is_the_full_string(sink: SnapshotSink) -> None:
    """exception.message is str(err), status prefix and cause included, the same text as status.message."""

    async def reserve_table(_span: SpanContext) -> None:
        raise GenkitError(status='UNAVAILABLE', message='booking service down', cause=TimeoutError('after 5s'))

    with pytest.raises(GenkitError):
        await run_in_new_span('reserveTable', reserve_table)

    span = sink.final('reserveTable')
    attrs = exception_event(span)
    assert attrs is not None
    assert attrs['exception.type'] == 'genkit._core._error.GenkitError'
    assert attrs['exception.message'] == 'UNAVAILABLE: booking service down: after 5s'
    assert span['status']['message'] == attrs['exception.message']
    assert span['attributes']['genkit:error'] == 'booking service down'


@pytest.mark.asyncio
async def test_empty_message_leaves_exception_message_out(sink: SnapshotSink) -> None:
    """A bare TimeoutError() still records its type and stack, without a blank heading."""

    async def wait_for_oven(_span: SpanContext) -> None:
        raise TimeoutError

    with pytest.raises(TimeoutError):
        await run_in_new_span('waitForOven', wait_for_oven)

    attrs = exception_event(sink.final('waitForOven'))
    assert attrs is not None
    assert attrs['exception.type'] == 'TimeoutError'
    assert 'exception.message' not in attrs
    assert 'TimeoutError' in attrs['exception.stacktrace']


@pytest.mark.asyncio
async def test_span_failed_without_raising_titles_with_the_message(sink: SnapshotSink) -> None:
    """set_state('error', message=...) adds an exception event, so the heading is the message, not "Error"."""

    async def plate_order(span: SpanContext) -> str:
        span.set_state('error', message='allergen db unavailable')
        return 'returned failed'

    await run_in_new_span('plateOrder', plate_order)

    span = sink.final('plateOrder')
    assert exception_event(span) == {'exception.message': 'allergen db unavailable'}
    assert span['status'] == {'code': 2, 'message': 'allergen db unavailable'}
    assert dev_ui_heading(span) == 'allergen db unavailable'


@pytest.mark.asyncio
async def test_raised_error_replaces_the_set_state_event(sink: SnapshotSink) -> None:
    """A body that marks itself failed and then raises shows the raised error, with its stack."""

    async def plate_order(span: SpanContext) -> None:
        span.set_state('error', message='plating failed')
        raise ValueError('oven offline')

    with pytest.raises(ValueError):
        await run_in_new_span('plateOrder', plate_order)

    span = sink.final('plateOrder')
    (event,) = span['timeEvents']['timeEvent']
    attrs = event['annotation']['attributes']
    assert (attrs['exception.type'], attrs['exception.message']) == ('ValueError', 'oven offline')
    assert 'in plate_order' in attrs['exception.stacktrace']


@pytest.mark.asyncio
async def test_set_state_without_message_adds_no_event(sink: SnapshotSink) -> None:
    """With no message there's nothing to title the span with; it stays "Error"."""

    async def plate_order(span: SpanContext) -> None:
        span.set_state('error')

    await run_in_new_span('plateOrder', plate_order)

    span = sink.final('plateOrder')
    assert exception_event(span) is None
    assert dev_ui_heading(span) == 'Error'


@pytest.mark.asyncio
async def test_stream_callback_failure_records_the_callback_error(sink: SnapshotSink) -> None:
    """The internal stream-callback carrier is unwrapped; the event names what the callback raised."""

    async def stream_receipt(_span: SpanContext) -> None:
        try:
            raise ConnectionResetError('printer offline')
        except ConnectionResetError as e:
            raise _StreamingCallbackError from e

    with pytest.raises(_StreamingCallbackError):
        await run_in_new_span('streamReceipt', stream_receipt)

    attrs = exception_event(sink.final('streamReceipt'))
    assert attrs is not None
    assert attrs['exception.type'] == 'ConnectionResetError'
    assert attrs['exception.message'] == 'printer offline'
    assert '_StreamingCallbackError' not in attrs['exception.stacktrace']


@pytest.mark.asyncio
async def test_finished_span_holds_strings_not_the_exception(sink: SnapshotSink) -> None:
    """The recorded exception is formatted text, so a finished span keeps no traceback frames alive."""
    held: list[ActiveSpan] = []

    class HoldingSink(SnapshotSink):
        def export_spans(self, spans: list[ActiveSpan]) -> None:
            held.extend(spans)

    reset_instrumentation()
    configure_instrumentation(DirectHttpInstrumentation(HoldingSink()))  # type: ignore[arg-type]

    async def charge_card(_span: SpanContext) -> None:
        raise RuntimeError('card declined')

    with pytest.raises(RuntimeError):
        await run_in_new_span('chargeCard', charge_card)

    exception = held[-1].exception
    assert exception is not None
    assert all(isinstance(v, str) for v in exception.values())


@pytest.mark.asyncio
async def test_interrupted_tool_renders_as_interrupted(sink: SnapshotSink) -> None:
    """An Interrupt records no exception event, so the Traces tab titles the tool span Interrupted."""
    ai = Genkit()
    scripted_model(ai, [calls_tool('approve_comp')])

    @ai.tool(name='approve_comp')
    async def approve_comp() -> str:
        raise Interrupt({'reason': 'manager approval'})

    response = await ai.generate(model='kitchen-model', prompt='comp the dessert', tools=['approve_comp'])
    assert response.finish_reason == FinishReason.INTERRUPTED

    tool, generate = sink.final('approve_comp'), sink.final('generate')
    assert tool['attributes']['genkit:state'] == 'error'
    assert tool['status'] == {'code': 2}
    assert tool['timeEvents'] == {'timeEvent': []}
    assert dev_ui_heading(tool) == 'Interrupted'
    assert generate['attributes']['genkit:state'] == 'success'
    assert generate['timeEvents'] == {'timeEvent': []}


@pytest.mark.asyncio
async def test_failure_returned_without_raising_titles_generate_with_the_message(sink: SnapshotSink) -> None:
    """generate boxes a tool failure into its response; its span still titles with the tool's message."""
    ai = Genkit()
    scripted_model(ai, [calls_tool('check_stock')])

    @ai.tool(name='check_stock')
    async def check_stock() -> str:
        raise RuntimeError('walk-in freezer offline')

    response = await ai.generate(model='kitchen-model', prompt='is the salmon in?', tools=['check_stock'])
    assert response.finish_reason == FinishReason.FAILED

    tool, generate = sink.final('check_stock'), sink.final('generate')
    assert dev_ui_heading(tool) == 'walk-in freezer offline'
    assert generate['attributes']['genkit:state'] == 'error'
    assert generate['status']['code'] == 2
    assert 'walk-in freezer offline' in generate['status']['message']
    assert dev_ui_heading(generate) == 'walk-in freezer offline'
    # Nothing raised in generate, so no stack here; the tool span has it.
    assert exception_event(generate) == {'exception.message': generate['status']['message']}
