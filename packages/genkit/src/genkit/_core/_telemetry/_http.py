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

"""Developer UI trace poster that POSTs TraceData to ``/api/traces``. No OpenTelemetry runtime.

Spans go to ``/api/traces`` rather than ``/api/otlp`` because the collector's
OTLP converter drops span events, and the Traces tab reads a failed span's
heading and stack from its ``exception`` event. Logs reach the Developer UI
through Genkit's own log exporter (``/api/otlp``), not here.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import math
import os
import secrets
import threading
import time
import urllib.request
from collections.abc import Mapping
from contextvars import ContextVar
from dataclasses import dataclass, field
from queue import Full, Queue
from typing import Literal, TypeVar
from urllib.parse import urljoin, urlparse

from .._environment import is_dev_environment
from .._error import GenkitError, Interrupt, _StreamingCallbackError, format_exception_traceback
from .._logger import get_logger
from .._reflection_config import reflection_enabled
from ._attrs import Attr, State, metadata_key
from ._instrumentation import (
    Instrumentation,
    SpanMetadata,
    SpanNext,
    configure_instrumentation,
    instrumentations,
    is_instrumented_by,
    parent_path_context,
    span_is_action,
    start_attributes,
    to_json_attr,
)
from ._log_exporter import QUEUE_SIZE, put_poison_pill
from ._path import build_path

logger = get_logger(__name__)

T = TypeVar('T')

TRACE_HEADERS = {'Content-Type': 'application/json', 'Accept': 'application/json'}
EXPORT_TIMEOUT_SECONDS = 300
# The Developer UI flushes right before it reads a trace. A hung collector
# should cost it a couple of seconds, not the full export timeout.
FLUSH_TIMEOUT_SECONDS = 2.0
# Cross-SDK value for each span's ``instrumentationLibrary`` field in the
# /api/traces body.
INSTRUMENTATION_LIBRARY = {'name': 'genkit-tracer', 'version': 'v1'}


@dataclass
class ActiveSpan:
    trace_id: str
    span_id: str
    parent_span_id: str | None
    name: str
    start_time_unix_nano: int
    end_time_unix_nano: int = 0
    attributes: dict[str, object] = field(default_factory=dict)
    status_code: int = 0
    status_message: str | None = None
    # exception.type / .message / .stacktrace, formatted when the span fails.
    # Plain strings, so a finished span holds no traceback frames.
    exception: dict[str, str] | None = field(default=None, repr=False, compare=False)
    parent: ActiveSpan | None = field(default=None, repr=False, compare=False)
    has_failed_child: bool = field(default=False, repr=False, compare=False)


_parent_span: ContextVar[ActiveSpan | None] = ContextVar('genkit_direct_http_parent', default=None)


def note_child_failed(span: ActiveSpan) -> None:
    curr = span.parent
    while curr is not None and not curr.has_failed_child:
        curr.has_failed_child = True
        curr = curr.parent


class DirectSpanContext:
    """Span handle that writes ``genkit:*`` attributes onto the live span."""

    def __init__(self, span: ActiveSpan) -> None:
        self._span = span

    @property
    def trace_id(self) -> str:
        return self._span.trace_id

    @property
    def span_id(self) -> str:
        return self._span.span_id

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        for key, value in metadata.items():
            try:
                encoded = value if isinstance(value, str) else to_json_attr(value)
            except Exception as e:
                encoded = f'Error encoding metadata: {e}'
            self._span.attributes[metadata_key(str(key))] = encoded

    def set_state(self, state: Literal['success', 'error'], *, message: str | None = None) -> None:
        self._span.attributes[Attr.STATE] = state
        if state == State.ERROR:
            self._span.status_code = 2
            if message:
                self._span.attributes[Attr.ERROR] = message
                self._span.status_message = message
            if not self._span.has_failed_child:
                self._span.attributes[Attr.IS_FAILURE_SOURCE] = True
            note_child_failed(self._span)


def now_unix_nano() -> int:
    return time.time_ns()


def new_trace_id() -> str:
    return secrets.token_hex(16)


def new_span_id() -> str:
    return secrets.token_hex(8)


def exception_attributes(exc: BaseException) -> dict[str, str]:
    """The attributes OpenTelemetry's ``Span.record_exception`` writes, minus ``exception.escaped``.

    An empty ``str(exc)`` leaves ``exception.message`` out, so the Traces tab
    falls back to its generic heading instead of a blank one.
    """
    cls = type(exc)
    module = cls.__module__
    attrs = {'exception.type': f'{module}.{cls.__qualname__}' if module and module != 'builtins' else cls.__qualname__}
    message = str(exc)
    if message:
        attrs['exception.message'] = message
    attrs['exception.stacktrace'] = format_exception_traceback(exc)
    return attrs


def nanos_to_millis(nanos: int) -> float:
    return nanos / 1_000_000


def encode_attribute_value(value: object) -> str | bool | int | float:
    # NaN and infinity aren't JSON; the collector would reject the whole trace.
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    if isinstance(value, (str, bool, int, float)):
        return value
    return str(value)


def encode_span(span: ActiveSpan) -> dict[str, object]:
    """One ``SpanData`` in the shape the collector's ``/api/traces`` schema takes.

    The start export has ``endTime`` 0, which the collector and the Traces tab
    read as still running. The end export replaces it in the collector's store.
    """
    end_time = nanos_to_millis(span.end_time_unix_nano) if span.end_time_unix_nano else 0
    status: dict[str, object] = {'code': span.status_code}
    if span.status_message:
        status['message'] = span.status_message
    time_events: list[dict[str, object]] = []
    if span.exception is not None and span.end_time_unix_nano:
        time_events.append({
            'time': end_time,
            'annotation': {'description': 'exception', 'attributes': dict(span.exception)},
        })
    body: dict[str, object] = {
        'spanId': span.span_id,
        'traceId': span.trace_id,
        'startTime': nanos_to_millis(span.start_time_unix_nano),
        'endTime': end_time,
        'attributes': {key: encode_attribute_value(value) for key, value in span.attributes.items()},
        'displayName': span.name,
        'links': [],
        'instrumentationLibrary': dict(INSTRUMENTATION_LIBRARY),
        'spanKind': 'INTERNAL',
        'sameProcessAsParentSpan': {'value': True},
        'status': status,
        'timeEvents': {'timeEvent': time_events},
    }
    if span.parent_span_id:
        body['parentSpanId'] = span.parent_span_id
    return body


def encode_trace(span: ActiveSpan) -> dict[str, object]:
    """One ``TraceData`` payload for a single span in the collector's ``/api/traces`` schema.

    Only the root span sets the trace's ``displayName`` and ``startTime``.
    ``endTime`` stays off until the root ends, so the Traces tab lists the
    trace as running until then.
    """
    encoded = encode_span(span)
    trace: dict[str, object] = {
        'traceId': span.trace_id,
        'spans': {span.span_id: encoded},
    }
    if span.parent_span_id is None:
        trace['displayName'] = span.name
        trace['startTime'] = encoded['startTime']
        if span.end_time_unix_nano:
            trace['endTime'] = encoded['endTime']
    return trace


def collector_traces_url(server: str) -> str:
    return urljoin(server.rstrip('/') + '/', 'api/traces')


def post_json(*, url: str, body: str) -> None:
    if urlparse(url).scheme not in ('http', 'https'):
        raise ValueError(f'invalid telemetry server URL {url!r}')
    request = urllib.request.Request(  # noqa: S310 — scheme checked above
        url,
        data=body.encode(),
        headers=TRACE_HEADERS,
        method='POST',
    )
    with urllib.request.urlopen(request, timeout=EXPORT_TIMEOUT_SECONDS) as response:  # noqa: S310
        response.read()


class CollectorHttpSink:
    """POSTs TraceData to the collector's ``/api/traces`` from one background worker."""

    def __init__(self, url: str) -> None:
        self.url = url
        self.closed = False
        self.queue: Queue[str | None] = Queue(maxsize=QUEUE_SIZE)
        self.worker = threading.Thread(target=self.run_worker, name='genkit-trace-export', daemon=True)
        self.worker.start()

    def export_spans(self, spans: list[ActiveSpan]) -> None:
        # Encoded on the caller's thread: the start export has to snapshot the
        # span before the body goes on to change it.
        if self.closed or not spans:
            return
        for span in spans:
            try:
                self.queue.put_nowait(json.dumps(encode_trace(span)))
            except Full:
                logger.debug('Developer UI trace export queue is full; dropping spans')

    def run_worker(self) -> None:
        while True:
            body = self.queue.get()
            try:
                if body is None:
                    return
                post_json(url=self.url, body=body)
            except Exception as e:
                logger.debug('Failed to export spans: %s', e)
            finally:
                self.queue.task_done()

    def flush(self) -> None:
        with self.queue.all_tasks_done:
            self.queue.all_tasks_done.wait_for(lambda: self.queue.unfinished_tasks == 0, timeout=FLUSH_TIMEOUT_SECONDS)

    def shutdown(self) -> None:
        self.closed = True
        self.flush()
        put_poison_pill(queue=self.queue)


class DirectHttpInstrumentation:
    """HTTP poster that mints Genkit spans and writes them to a sink.

    Testers pass a recording sink. The Developer UI Traces tab uses
    :class:`DevUIInstrumentation`, a distinct type, so a test sink
    does not count as the tab already being on.
    """

    def __init__(self, sink: CollectorHttpSink) -> None:
        self.sink = sink

    async def run_in_new_span(
        self,
        metadata: SpanMetadata,
        next: SpanNext[T],
    ) -> T:
        parent = _parent_span.get()
        is_action = span_is_action.get()
        qualified_path = build_qualified_path(metadata, is_action=is_action)
        span = ActiveSpan(
            trace_id=parent.trace_id if parent is not None else new_trace_id(),
            span_id=new_span_id(),
            parent_span_id=parent.span_id if parent is not None else None,
            name=metadata.name,
            start_time_unix_nano=now_unix_nano(),
            attributes=start_attributes(metadata, qualified_path=qualified_path, is_action=is_action),
            parent=parent,
        )
        if parent is None:
            span.attributes[Attr.IS_ROOT] = True
        self.sink.export_spans([span])
        path_token = parent_path_context.set(qualified_path)
        parent_token = _parent_span.set(span)
        ctx = DirectSpanContext(span)
        try:
            try:
                result = await next(ctx)
                if result is not None:
                    span.attributes[Attr.OUTPUT] = to_json_attr(result)
                if Attr.STATE not in span.attributes:
                    span.attributes[Attr.STATE] = State.SUCCESS
                    span.status_code = 1
                return result
            except Interrupt as e:
                # the Dev UI draws an interrupted tool from the error state plus
                # genkit:metadata:interrupt; generate above it still succeeds.
                # No exception event: the Traces tab titles the span
                # "Interrupted" only when there's no exception message.
                span.attributes[Attr.STATE] = State.ERROR
                span.status_code = 2
                mark_failure_source(span=span, exc=e)
                raise
            except (asyncio.CancelledError, KeyboardInterrupt):
                raise
            except (Exception, _StreamingCallbackError) as e:
                actual = e.__cause__ if isinstance(e, _StreamingCallbackError) and e.__cause__ is not None else e
                try:
                    span.attributes[Attr.STATE] = State.ERROR
                    err_text = actual.message if isinstance(actual, GenkitError) else str(actual)
                    span.attributes[Attr.ERROR] = err_text
                    span.status_code = 2
                    span.status_message = str(actual)
                    # The Traces tab heading reads exception.message and the stack
                    # box reads exception.stacktrace.
                    span.exception = exception_attributes(actual)
                    note_child_failed(span)
                    mark_failure_source(span=span, exc=actual)
                    raise
                finally:
                    actual = None
        finally:
            span.end_time_unix_nano = now_unix_nano()
            self.sink.export_spans([span])
            _parent_span.reset(parent_token)
            parent_path_context.reset(path_token)

    def dispose(self) -> None:
        self.sink.shutdown()

    def flush(self) -> None:
        self.sink.flush()


class DevUIInstrumentation(DirectHttpInstrumentation):
    """The poster that fills the Developer UI Traces tab.

    ``genkit start`` and handshake/notify install this. A recording sink
    used in tests is a plain DirectHttpInstrumentation and does not
    count as the Traces tab already being on.
    """


# Set on an exception once the span it was raised in claims it. Every span it
# then passes through still turns red, but only the first one says the failure
# started there, so the Traces tab can point at the culprit.
# Only explicit causes (`raise ... from e`, `GenkitError(cause=e)`) count as
# derived, same as JS `cause` and Go `%w`. Implicit `__context__` does not.
FAILURE_SOURCE_MARK = '_genkit_failure_source'


def mark_failure_source(*, span: ActiveSpan, exc: BaseException) -> None:
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        if getattr(current, FAILURE_SOURCE_MARK, False):
            return
        seen.add(id(current))
        current = current.__cause__
    span.attributes[Attr.IS_FAILURE_SOURCE] = True
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, FAILURE_SOURCE_MARK, True)


def mark_already_sourced(exc: BaseException) -> None:
    """Stamp an error rebuilt from a boxed response whose span already claimed the failure."""
    if not is_instrumented_by(DirectHttpInstrumentation):
        return
    with contextlib.suppress(AttributeError, TypeError):
        setattr(exc, FAILURE_SOURCE_MARK, True)


def reset_parent_span() -> None:
    _parent_span.set(None)


def build_qualified_path(metadata: SpanMetadata, *, is_action: bool = False) -> str:
    if is_action:
        return build_path(metadata.name, parent_path_context.get(), 'action', metadata.action_type)
    return build_path(metadata.name, parent_path_context.get(), metadata.action_type or '')


def telemetry_server_url() -> str | None:
    url = os.environ.get('GENKIT_TELEMETRY_SERVER')
    return url or None


def direct_http_for_collector(*, url: str) -> DevUIInstrumentation:
    return DevUIInstrumentation(CollectorHttpSink(collector_traces_url(url)))


def genkit_dev_instrumentation() -> Instrumentation | None:
    """Developer UI poster, or None when no collector URL is set.

    ``genkit start -- python app.py`` sets ``GENKIT_TELEMETRY_SERVER``
    before spawn; ``Genkit()`` calls this. Does not boot OpenTelemetry.
    """
    url = telemetry_server_url()
    if url is None:
        return None
    return direct_http_for_collector(url=url)


def connect_developer_ui_collector(*, url: str) -> None:
    """Turn on the Developer UI poster from a handshake / notify URL.

    No-op when the URL is empty or the poster is already registered, so
    ``Genkit()`` under ``genkit start`` and a later notify cannot double-post.
    A ``GENKIT_TELEMETRY_SERVER`` already in the production shell (no poster
    yet) does not block this — today's handshake URL still fills the Traces tab.
    """
    if not url:
        return
    if is_instrumented_by(DevUIInstrumentation):
        return
    configure_instrumentation(direct_http_for_collector(url=url))


def flush_developer_ui_collector() -> None:
    """Drain the local HTTP trace poster before a run response returns its trace id.

    Flushes ``DirectHttpInstrumentation`` (``DevUIInstrumentation``) only; remote
    APM backends (``GenAiInstrumentation``, Cloud Trace, custom providers) stay
    isolated from interactive Dev UI clicks.
    """
    for inst in list(instrumentations):
        if isinstance(inst, DirectHttpInstrumentation):
            try:
                inst.flush()
            except Exception as e:
                logger.debug('Failed to flush Developer UI trace collector: %s', e)


def maybe_inject_dev_instrumentation() -> None:
    """``Genkit()`` installs the poster once when a collector URL is set.

    Keyed on reflection being on rather than GENKIT_ENV alone, so a non-dev
    runtime with reflection enabled still posts traces to the server it was
    given.
    """
    if not is_dev_environment() and not reflection_enabled():
        return
    if is_instrumented_by(DevUIInstrumentation):
        return
    inst = genkit_dev_instrumentation()
    if inst is not None:
        configure_instrumentation(inst)
