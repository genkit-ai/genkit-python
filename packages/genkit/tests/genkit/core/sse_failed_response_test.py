#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""BoxedFailure fidelity for Retry-After hints and ErrorReason."""

from genkit._core._error import BoxedFailure, ErrorReason, GenkitError, PublicError


def test_boxed_failure_keeps_retry_after_and_reason() -> None:
    """to_exception() re-raises the Retry-After hint and reason Retry middleware reads."""
    original = GenkitError(
        status='RESOURCE_EXHAUSTED',
        message='quota',
        retry_after_ms=1500.0,
        reason=ErrorReason.TOOL_FAILED,
    )

    again = BoxedFailure.from_exc(original).to_exception()

    assert again.status == 'RESOURCE_EXHAUSTED'
    assert again.retry_after_ms == 1500.0
    assert again.reason is ErrorReason.TOOL_FAILED


def test_boxed_public_failure_keeps_retry_after() -> None:
    original = PublicError('RESOURCE_EXHAUSTED', 'slow down')
    original.retry_after_ms = 2000.0

    again = BoxedFailure.from_exc(original).to_exception()

    assert isinstance(again, PublicError)
    assert again.retry_after_ms == 2000.0
