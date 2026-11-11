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

"""Tests for the plugin's own Interactions wire types (parsing and open unions)."""

from __future__ import annotations

import pytest
from genkit_google_genai._interactions._types import (
    CONTENT_ADAPTER,
    STEP_ADAPTER,
    UnknownContent,
    UnknownSseEvent,
    UnknownStep,
    UnknownStepDelta,
    UnknownTextAnnotation,
    parse_interaction,
    parse_sse_event,
)
from pydantic import ValidationError


class TestParseInteraction:
    def test_keeps_exactly_what_the_server_sent(self) -> None:
        body = {
            'id': 'v1_abc123',
            'status': 'completed',
            'steps': [
                {'type': 'url_context_call', 'id': 'u1'},
                {'type': 'mcp_server_tool_result', 'call_id': 'm1'},
                {'type': 'model_output', 'content': [{'type': 'text', 'text': 'Hi'}]},
            ],
            'usage': {'total_tokens': 5},
        }
        # Nothing is filled in (the SDK added `arguments: {}`, `result: {}`).
        assert parse_interaction(body) == body

    def test_keeps_fields_the_types_do_not_list(self) -> None:
        body = {
            'status': 'completed',
            'new_top_level_field': {'a': 1},
            'steps': [{'type': 'google_search_call', 'id': 'g1', 'new_step_field': True}],
        }
        assert parse_interaction(body) == body

    def test_keeps_an_unknown_step_type_as_sent(self) -> None:
        future = {'type': 'future_step', 'payload': {'x': [1, 2]}}
        parsed = parse_interaction({'status': 'completed', 'steps': [future]})
        steps = parsed.get('steps') or []
        assert steps == [future]
        assert isinstance(steps[0], UnknownStep)

    def test_keeps_a_known_step_with_mismatched_fields_as_unknown(self) -> None:
        odd = {'type': 'url_context_call', 'id': 'u1', 'arguments': 'not an object'}
        parsed = parse_interaction({'status': 'completed', 'steps': [odd]})
        steps = parsed.get('steps') or []
        assert steps == [odd]
        assert isinstance(steps[0], UnknownStep)

    def test_known_step_is_not_unknown(self) -> None:
        step = STEP_ADAPTER.validate_python({'type': 'processing_call', 'id': 'p1', 'signature': 'c2ln'})
        assert not isinstance(step, UnknownStep)
        assert step['type'] == 'processing_call'

    def test_keeps_unknown_content_inside_a_known_step(self) -> None:
        step = STEP_ADAPTER.validate_python({
            'type': 'model_output',
            'content': [{'type': 'hologram', 'data': 'x'}, {'type': 'text', 'text': 'ok'}],
        })
        assert not isinstance(step, UnknownStep)
        assert step['type'] == 'model_output'
        content = step.get('content') or []
        assert isinstance(content[0], UnknownContent)
        assert not isinstance(content[1], UnknownContent)

    def test_keeps_unknown_text_annotations(self) -> None:
        content = CONTENT_ADAPTER.validate_python({
            'type': 'text',
            'text': 'hi',
            'annotations': [{'type': 'future_citation', 'x': 1}, {'type': 'url_citation', 'url': 'https://a.b'}],
        })
        assert not isinstance(content, UnknownContent)
        assert content['type'] == 'text'
        annotations = content.get('annotations') or []
        assert isinstance(annotations[0], UnknownTextAnnotation)
        assert not isinstance(annotations[1], UnknownTextAnnotation)
        assert content == {
            'type': 'text',
            'text': 'hi',
            'annotations': [{'type': 'future_citation', 'x': 1}, {'type': 'url_citation', 'url': 'https://a.b'}],
        }

    def test_raises_for_a_body_that_is_not_an_interaction(self) -> None:
        with pytest.raises(ValidationError):
            parse_interaction(['not', 'an', 'object'])
        with pytest.raises(ValidationError):
            parse_interaction({'steps': 'not a list'})

    def test_raises_for_a_step_without_a_type(self) -> None:
        with pytest.raises(ValidationError):
            parse_interaction({'steps': [{'id': 'no-type'}]})


class TestParseSseEvent:
    def test_parses_a_step_delta_event(self) -> None:
        event = parse_sse_event({'event_type': 'step.delta', 'index': 0, 'delta': {'type': 'text', 'text': 'Hi'}})
        assert not isinstance(event, UnknownSseEvent)
        assert event['event_type'] == 'step.delta'
        delta = event.get('delta')
        assert delta is not None and not isinstance(delta, UnknownStepDelta)
        assert delta == {'type': 'text', 'text': 'Hi'}

    def test_keeps_unknown_events_and_deltas(self) -> None:
        event = parse_sse_event({'event_type': 'future.event', 'x': 1})
        assert isinstance(event, UnknownSseEvent)
        assert event == {'event_type': 'future.event', 'x': 1}
        delta_event = parse_sse_event({'event_type': 'step.delta', 'index': 1, 'delta': {'type': 'future_delta'}})
        assert not isinstance(delta_event, UnknownSseEvent)
        assert delta_event['event_type'] == 'step.delta'
        delta = delta_event.get('delta')
        assert isinstance(delta, UnknownStepDelta)
