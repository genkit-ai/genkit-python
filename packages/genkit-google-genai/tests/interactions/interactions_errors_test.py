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

"""Tests for the Interactions error codes (ported from the JS utils tests)."""

from __future__ import annotations

import pytest
from genkit_google_genai._interactions._errors import (
    interaction_error_status,
    is_interaction_content_block_code,
    unwrap_error_body,
)

CONTENT_BLOCK_CODES = [
    'safety',
    'recitation',
    'language',
    'prohibited_content',
    'spii',
    'blocklist',
    'image_safety',
    'image_prohibited_content',
    'image_recitation',
    'image_other',
    'content_blocked',
    'jailbreak',
    'model_armor',
]


class TestInteractionErrorStatus:
    def test_maps_every_known_interactions_error_code(self) -> None:
        expected = {
            'service_unavailable': 'UNAVAILABLE',
            'rate_limit_exceeded': 'RESOURCE_EXHAUSTED',
            'quota_exceeded': 'RESOURCE_EXHAUSTED',
            'invalid_request': 'INVALID_ARGUMENT',
            'parameter_unknown': 'INVALID_ARGUMENT',
            'failed_precondition': 'FAILED_PRECONDITION',
            'out_of_range': 'OUT_OF_RANGE',
            'agent_max_token_limit': 'INVALID_ARGUMENT',
            'model_not_found': 'NOT_FOUND',
            'not_found': 'NOT_FOUND',
            'authentication': 'UNAUTHENTICATED',
            'permission_denied': 'PERMISSION_DENIED',
            'already_exists': 'ALREADY_EXISTS',
            'aborted': 'ABORTED',
            'cancelled': 'CANCELLED',
            'deadline_exceeded': 'DEADLINE_EXCEEDED',
            'api_error': 'INTERNAL',
            'unimplemented': 'UNIMPLEMENTED',
            'malformed_function_call': 'ABORTED',
            'unexpected_tool_call': 'ABORTED',
            'no_image': 'ABORTED',
            **dict.fromkeys(CONTENT_BLOCK_CODES, 'FAILED_PRECONDITION'),
        }
        for code, status in expected.items():
            assert interaction_error_status(code) == status, code

    def test_matches_codes_case_insensitively_and_ignores_surrounding_whitespace(self) -> None:
        assert interaction_error_status('SERVICE_UNAVAILABLE') == 'UNAVAILABLE'
        assert interaction_error_status(' rate_limit_exceeded ') == 'RESOURCE_EXHAUSTED'

    @pytest.mark.parametrize('code', ['weird', '', None, 503, {}])
    def test_returns_none_for_unmappable_codes(self, code: object) -> None:
        assert interaction_error_status(code) is None


class TestIsInteractionContentBlockCode:
    def test_recognizes_content_policy_codes_case_insensitively(self) -> None:
        assert is_interaction_content_block_code('safety')
        assert is_interaction_content_block_code('MODEL_ARMOR')
        assert is_interaction_content_block_code('image_other')

    @pytest.mark.parametrize('code', ['malformed_function_call', 'service_unavailable', None, 400])
    def test_rejects_non_content_policy_codes_and_non_strings(self, code: object) -> None:
        assert not is_interaction_content_block_code(code)


class TestUnwrapErrorBody:
    def test_unwraps_a_one_element_list(self) -> None:
        body = {'error': {'code': 400, 'message': 'API key not valid. Please pass a valid API key.'}}
        assert unwrap_error_body([body]) == body

    @pytest.mark.parametrize('body', [{'error': {'message': 'x'}}, [], [{'a': 1}, {'b': 2}], 'text', None])
    def test_returns_anything_else_as_is(self, body: object) -> None:
        assert unwrap_error_body(body) == body
