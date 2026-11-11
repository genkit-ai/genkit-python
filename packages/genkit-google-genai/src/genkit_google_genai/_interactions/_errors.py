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

"""Interactions API error codes and error bodies (shared by the client and the converters)."""

from __future__ import annotations

from genkit_google_genai._interactions._types import API_ERROR_BODY_ADAPTER
from pydantic import ValidationError

from genkit.plugin_api import StatusName

# Content-policy blocks (safety, recitation, ...). A failed interaction with
# one of these codes is returned as a response with finish_reason 'blocked'
# (like a blocked generateContent response) rather than raised. When raised
# (e.g. in an error event), they map to FAILED_PRECONDITION, which isn't
# retried: the same input would be blocked again.
INTERACTION_CONTENT_BLOCK_CODES: frozenset[str] = frozenset({
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
})

# Interactions API error codes and the Genkit status for each. The API
# reports errors as `{code: string, message: string}`; the spec doesn't list
# the codes, so this is the set the API returns (with the HTTP status the
# API pairs each one with).
INTERACTION_ERROR_CODE_TO_STATUS: dict[str, StatusName] = {
    # Generation failures reported with `status: 'failed'` + `errors[]` after
    # the server's own retries ran out. The server message says to retry, so
    # these are ABORTED ("retry at a higher level"), which Retry retries.
    'malformed_function_call': 'ABORTED',
    'unexpected_tool_call': 'ABORTED',
    'no_image': 'ABORTED',
    **dict.fromkeys(INTERACTION_CONTENT_BLOCK_CODES, 'FAILED_PRECONDITION'),
    'service_unavailable': 'UNAVAILABLE',  # 503: server or model overloaded
    'rate_limit_exceeded': 'RESOURCE_EXHAUSTED',  # 429: RPM/TPM limits
    'quota_exceeded': 'RESOURCE_EXHAUSTED',  # 429: RPD/daily limits
    'invalid_request': 'INVALID_ARGUMENT',  # 400: general bad request
    'parameter_unknown': 'INVALID_ARGUMENT',  # 400: unrecognized field
    'failed_precondition': 'FAILED_PRECONDITION',  # 400
    'out_of_range': 'OUT_OF_RANGE',  # 400
    # 400: over the agent's max token limit. The same request fails the same
    # way, so deliberately not RESOURCE_EXHAUSTED (which is retried).
    'agent_max_token_limit': 'INVALID_ARGUMENT',
    'model_not_found': 'NOT_FOUND',  # 404
    'not_found': 'NOT_FOUND',  # 404
    'authentication': 'UNAUTHENTICATED',  # 401
    'permission_denied': 'PERMISSION_DENIED',  # 403
    'already_exists': 'ALREADY_EXISTS',  # 409
    'aborted': 'ABORTED',  # 409
    'cancelled': 'CANCELLED',  # 499
    'deadline_exceeded': 'DEADLINE_EXCEEDED',  # 504
    'api_error': 'INTERNAL',  # 500
    'unimplemented': 'UNIMPLEMENTED',  # 501
}


def interaction_error_status(code: object) -> StatusName | None:
    """The Genkit status for an Interactions error code (case-insensitive), or None.

    Only the API's string codes: numeric codes (HTTP or gRPC) are mapped by
    the caller, which knows which kind it has.
    """
    if not isinstance(code, str):
        return None
    return INTERACTION_ERROR_CODE_TO_STATUS.get(code.strip().lower())


def is_interaction_content_block_code(code: object) -> bool:
    """True for a content-policy error code (safety, recitation, blocklist, ...)."""
    return isinstance(code, str) and code.strip().lower() in INTERACTION_CONTENT_BLOCK_CODES


def api_error_fields(body: object) -> tuple[str | None, str | int | None]:
    """The message and code of an API error body, or ``(None, None)`` when it isn't one."""
    try:
        parsed = API_ERROR_BODY_ADAPTER.validate_python(body)
    except ValidationError:
        return None, None
    error = parsed.get('error')
    if error is None:
        return None, None
    return error.get('message'), error.get('code')


def unwrap_error_body(body: object) -> object:
    """Unwrap an error body sent as a one-element JSON array.

    The Interactions API sometimes wraps its error that way: a bad API key
    gives ``[{"error": {"code": 400, "message": "API key not valid. ..."}}]``
    (seen with the Dart plugin, Nov 2026). Anything else is returned as is.
    """
    if isinstance(body, list) and len(body) == 1:
        return body[0]
    return body
