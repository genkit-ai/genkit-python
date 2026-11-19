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

"""The A2UI part's wire shape: written as ``messages``, read as either key."""

from __future__ import annotations

from genkit_a2ui import A2UI_MIME_TYPE, a2ui_part, envelopes_from_parts, is_a2ui_part

from genkit import Part

CREATE = {'version': 'v0.9', 'createSurface': {'surfaceId': 's', 'catalogId': 'c'}}
DELETE = {'version': 'v0.9', 'deleteSurface': {'surfaceId': 's'}}


def _part(data: dict[str, object]) -> Part:
    return Part.from_data(data, metadata={'mimeType': A2UI_MIME_TYPE})


def test_a2ui_part_writes_the_spec_messages_key() -> None:
    assert a2ui_part([CREATE]).data == {'messages': [CREATE]}


def test_reads_the_legacy_envelopes_key() -> None:
    legacy = _part({'envelopes': [CREATE]})
    assert is_a2ui_part(legacy)
    assert envelopes_from_parts([legacy]) == [CREATE]


def test_messages_wins_when_both_keys_are_present() -> None:
    assert envelopes_from_parts([_part({'messages': [CREATE], 'envelopes': [DELETE]})]) == [CREATE]


def test_empty_messages_list_wins_over_legacy_envelopes() -> None:
    empty = _part({'messages': [], 'envelopes': [DELETE]})
    assert is_a2ui_part(empty)
    assert envelopes_from_parts([empty]) == []


def test_falls_back_to_envelopes_when_messages_is_none() -> None:
    part = _part({'messages': None, 'envelopes': [CREATE]})
    assert is_a2ui_part(part)
    assert envelopes_from_parts([part]) == [CREATE]


def test_rejects_non_list_message_lists() -> None:
    for bad in [
        _part({'messages': 'oops'}),
        _part({'messages': 'oops', 'envelopes': [CREATE]}),
        _part({'envelopes': 'oops'}),
    ]:
        assert not is_a2ui_part(bad)
        assert envelopes_from_parts([bad]) == []


def test_rejects_non_a2ui_parts() -> None:
    wrong_mime = Part.from_data({'messages': [CREATE]}, metadata={'mimeType': 'application/json'})
    missing_mime = Part.from_data({'messages': [CREATE]})
    plain_text = Part.from_text('hello')
    for part in [wrong_mime, missing_mime, plain_text]:
        assert not is_a2ui_part(part)
        assert envelopes_from_parts([part]) == []
