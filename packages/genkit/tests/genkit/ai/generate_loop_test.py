#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Turn-loop regressions: usage totals, reasoning accessors, tool-response parts, and post-middleware request."""

from collections.abc import Awaitable, Callable

import pytest

from genkit import (
    Document,
    Genkit,
    Message,
    ModelResponse,
    ModelResponseChunk,
    MultipartToolResponse,
    Part,
)
from genkit._core._typing import FinishReason, GenerationUsage, Role, ToolRequest
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.testing import define_scripted_model


def _calls_tool(*, ref: str, usage: GenerationUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part(tool_request=ToolRequest(name='lookup', input={}, ref=ref))]),
        usage=usage,
    )


def _answers(text: str, usage: GenerationUsage | None = None) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
        usage=usage,
    )


@pytest.mark.asyncio
async def test_total_usage_sums_every_turn_of_a_tool_loop() -> None:
    """usage is the last call; total_usage is what the whole loop billed."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return 'in stock'

    pm.responses = [
        _calls_tool(
            ref='1',
            usage=GenerationUsage(input_tokens=10, output_tokens=5, thoughts_tokens=3, custom={'calls': 1, 'id': 'a'}),
        ),
        _calls_tool(ref='2', usage=GenerationUsage(input_tokens=20, output_tokens=6, custom={'calls': 1, 'id': 'b'})),
        _answers('done', usage=GenerationUsage(input_tokens=30, output_tokens=7)),
    ]

    response = await ai.generate(prompt='check stock', tools=['lookup'])

    assert response.text == 'done'
    assert response.usage is not None
    assert response.usage.input_tokens == 30
    assert response.total_usage == GenerationUsage(
        input_tokens=60,
        output_tokens=18,
        thoughts_tokens=3,
        custom={'calls': 2, 'id': 'b'},
    )


@pytest.mark.asyncio
async def test_total_usage_matches_usage_on_a_single_turn() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('hi', usage=GenerationUsage(input_tokens=4, output_tokens=2))]

    response = await ai.generate(prompt='hello')

    assert response.total_usage == response.usage


@pytest.mark.asyncio
async def test_total_usage_survives_a_max_turns_abort() -> None:
    """The rounds billed before the cap still show up on the aborted response."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='lookup')
    async def lookup() -> str:
        return 'again'

    pm.responses = [
        _calls_tool(ref='1', usage=GenerationUsage(input_tokens=10)),
        _calls_tool(ref='2', usage=GenerationUsage(input_tokens=11)),
    ]

    response = await ai.generate(prompt='loop', tools=['lookup'], max_turns=1)

    assert response.finish_reason == FinishReason.ABORTED
    assert response.total_usage is not None
    assert response.total_usage.input_tokens == 21


@pytest.mark.asyncio
async def test_total_usage_stays_off_when_no_model_reported_usage() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('hi')]

    response = await ai.generate(prompt='hello')

    assert response.total_usage is None


def test_reasoning_accessors_stay_out_of_text() -> None:
    message = Message(
        role=Role.MODEL,
        content=[
            Part.from_reasoning('Check allergens first. '),
            Part.from_reasoning('Nuts: none.'),
            Part.from_text('Safe.'),
        ],
    )
    response = ModelResponse(message=message)
    chunk = ModelResponseChunk(role=Role.MODEL, content=[Part.from_reasoning('Thinking')])

    assert message.reasoning == 'Check allergens first. Nuts: none.'
    assert response.reasoning == message.reasoning
    assert response.text == 'Safe.'
    assert chunk.reasoning == 'Thinking'
    assert ModelResponse().reasoning == ''


def test_from_tool_response_carries_media_content() -> None:
    chart = Part.from_media('data:image/png;base64,iVBOR', content_type='image/png')

    part = Part.from_tool_response(name='sales_chart', ref='c1', output={'total': 42}, content=[chart])

    assert part.tool_response is not None
    assert part.tool_response.output == {'total': 42}
    assert part.tool_response.content == [{'media': {'url': 'data:image/png;base64,iVBOR', 'contentType': 'image/png'}}]


def test_from_tool_response_unwraps_a_multipart_output() -> None:
    chart = Part.from_media('data:image/png;base64,iVBOR', content_type='image/png')
    multipart = MultipartToolResponse(output={'total': 42}, content=[chart], metadata={'cached': True})

    part = Part.from_tool_response(name='sales_chart', ref='c1', output=multipart)

    assert part.tool_response is not None
    assert part.tool_response.output == {'total': 42}
    assert part.tool_response.content is not None
    assert len(part.tool_response.content) == 1
    assert part.metadata == {'cached': True}


@pytest.mark.asyncio
async def test_response_request_is_what_the_model_received() -> None:
    """A wrap_model that strips sensitive docs must not see them echoed on response.request.

    Messages on response.request stay the resendable history by design; scrub
    those in wrap_generate.
    """
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok')]

    class DropPatientDocs(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            return await next_fn(ModelHookParams(request=params.request.model_copy(update={'docs': None})), ctx)

    response = await ai.generate(
        prompt='summarize',
        docs=[Document.from_text('patient SSN 123-45-6789')],
        use=[DropPatientDocs()],
    )

    assert pm.last_request is not None
    assert pm.last_request.docs is None
    assert response.request is not None
    assert response.request.docs is None
    assert [m.text for m in response.request.messages] == ['summarize']
