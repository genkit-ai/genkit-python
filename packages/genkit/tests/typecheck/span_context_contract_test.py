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

# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Typed contract of ``SpanContext.set_state``.

pyright, pyrefly and ty must accept ``span.set_state('error')`` and
``span.set_state('error', message='...')``, reject passing ``message``
positionally, and reject a custom ``SpanContext`` whose ``set_state`` omits
the keyword-only ``message`` parameter.
"""

from collections.abc import Mapping

import pytest

from genkit.telemetry import SpanContext


class _RecordingSpan:
    def __init__(self) -> None:
        self.trace_id = 'a' * 32
        self.span_id = 'b' * 16
        self.calls: list[tuple[str, str | None]] = []

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        _ = metadata

    def set_state(self, state: str, *, message: str | None = None) -> None:
        self.calls.append((state, message))


class _OneArgSpan:
    @property
    def trace_id(self) -> str:
        return ''

    @property
    def span_id(self) -> str:
        return ''

    def set_metadata(self, metadata: Mapping[str, object]) -> None:
        _ = metadata

    def set_state(self, state: str) -> None:
        _ = state


def test_span_context_set_state_accepts_optional_keyword_message() -> None:
    """Both ``set_state('error')`` and ``set_state('error', message=...)`` type-check."""
    raw = _RecordingSpan()
    span: SpanContext = raw

    span.set_state('success')
    span.set_state('error')
    span.set_state('error', message='allergen service offline')

    assert raw.calls == [
        ('success', None),
        ('error', None),
        ('error', 'allergen service offline'),
    ]


def test_span_context_set_state_rejects_positional_message_and_one_arg_impl() -> None:
    """``message`` is keyword-only and ``SpanContext`` requires it on providers."""
    span: SpanContext = _RecordingSpan()

    with pytest.raises(TypeError, match='positional argument'):
        span.set_state('error', 'allergen service offline')  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[bad-argument-count]  # ty: ignore[too-many-positional-arguments]

    bad: SpanContext = _OneArgSpan()  # pyright: ignore[reportAssignmentType]  # pyrefly: ignore[bad-assignment]  # ty: ignore[invalid-assignment]
    assert bad.trace_id == ''
