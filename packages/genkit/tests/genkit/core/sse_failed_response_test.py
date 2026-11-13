#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""to_sse frames: generate streams unbox failures; flows return data as-is."""

import json
import math
from collections.abc import AsyncIterator

import pytest

from genkit import Genkit, ModelResponse
from genkit._core._typing import FinishReason
from genkit.testing import define_scripted_model
from genkit.web import to_sse


async def _frames(handle: object) -> list[dict[str, object]]:
    return [json.loads(line.removeprefix('data: ')) async for line in to_sse(handle)]  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_generate_stream_failure_sends_an_error_frame() -> None:
    """generate_stream boxes the failure on .response; to_sse unboxes it into an error frame."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = []  # The scripted model raises when it runs out of responses.

    frames = await _frames(ai.generate_stream(prompt='tartine'))

    assert 'error' in frames[-1]
    assert 'result' not in frames[-1]


@pytest.mark.asyncio
async def test_flow_returning_a_failed_response_sends_it_as_the_result() -> None:
    """A flow's return value is data: a FAILED ModelResponse goes out as result, same as JSON."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = []

    @ai.flow()
    async def suggest_dish(order: str) -> ModelResponse:
        return await ai.generate(prompt=order)

    frames = await _frames(suggest_dish.stream('tartine'))

    assert len(frames) == 1
    assert frames[0]['result']['finishReason'] == 'failed'  # type: ignore[index]


class _Handle:
    def __init__(self, result: object) -> None:
        self._result = result

    @property
    def stream(self) -> AsyncIterator[object]:
        async def chunks() -> AsyncIterator[object]:
            yield 'chunk'

        return chunks()

    @property
    def response(self) -> object:
        async def result() -> object:
            return self._result

        return result()


@pytest.mark.asyncio
async def test_non_generate_handle_with_a_failed_response_sends_the_result() -> None:
    failed = ModelResponse(finish_reason=FinishReason.FAILED, finish_message='quota exceeded')

    frames = await _frames(_Handle(failed))

    assert frames[0] == {'message': 'chunk'}
    assert frames[-1]['result']['finishReason'] == 'failed'  # type: ignore[index]


@pytest.mark.asyncio
async def test_successful_response_still_sends_the_result() -> None:
    ok = ModelResponse(finish_reason=FinishReason.STOP)

    frames = await _frames(_Handle(ok))

    assert frames[-1]['result']['finishReason'] == 'stop'  # type: ignore[index]


@pytest.mark.asyncio
async def test_non_finite_floats_go_out_as_null() -> None:
    """A flow returning NaN/Infinity streams null, like tool outputs, not a 500."""
    ai = Genkit()

    @ai.flow()
    async def score(dish: str) -> dict[str, float]:
        return {'z_score': math.nan, 'upper': math.inf}

    frames = await _frames(score.stream('tartine'))

    assert frames == [{'result': {'z_score': None, 'upper': None}}]
