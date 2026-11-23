#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""A callable instance whose __call__ is async is a handler everywhere, not only in Tool."""

import pytest

from genkit._core._action import Action
from genkit._core._flow import Flow
from genkit._core._telemetry._instrumentation import run_in_new_span
from genkit.plugin_api import ActionKind


class Echo:
    """Async callable instance; deliberately has no __name__."""

    async def __call__(self, value: str) -> str:
        """Return the argument unchanged."""
        return value


class SyncEcho:
    """Sync callable instance; still not a valid handler."""

    def __call__(self, value: str) -> str:
        """Return the argument unchanged."""
        return value


@pytest.mark.asyncio
async def test_action_accepts_async_callable_instance() -> None:
    """Action builds and runs from an instance, not just a bare coroutine function."""
    action = Action(ActionKind.CUSTOM, 'echo', Echo())
    assert (await action.run('hi')).response == 'hi'


@pytest.mark.asyncio
async def test_flow_accepts_async_callable_instance() -> None:
    """Flow builds, names the instance after its class, and runs."""
    flow = Flow(Echo())
    assert flow.name == 'Echo'
    assert (await flow.run('yo')).response == 'yo'


@pytest.mark.asyncio
async def test_run_in_new_span_accepts_async_callable_instance() -> None:
    """The span helper accepts the same objects Action and Flow accept."""
    span = await run_in_new_span('probe', Echo())
    assert span is not None


def test_sync_callable_instance_is_rejected() -> None:
    """A sync __call__ keeps the existing TypeError and message."""
    with pytest.raises(TypeError, match='Action handlers must be async functions'):
        Action(ActionKind.CUSTOM, 'sync-echo', SyncEcho())
    with pytest.raises(TypeError, match='Flow must be async'):
        Flow(SyncEcho())


@pytest.mark.asyncio
async def test_run_in_new_span_rejects_sync_instance() -> None:
    """The span helper still refuses a sync callable instance."""
    with pytest.raises(TypeError, match='expected an async callback'):
        await run_in_new_span('probe', SyncEcho())
