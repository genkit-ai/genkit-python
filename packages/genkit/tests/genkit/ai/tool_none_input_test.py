# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tool input: an explicit None from Python is a value; null from JSON means omitted."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from genkit import Genkit, GenkitError, Message, ModelResponse, Part
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._typing import FinishReason, Role, ToolRequest
from genkit.testing import define_scripted_model


class Allergens(BaseModel):
    nuts: bool = False


@pytest.mark.asyncio
async def test_tool_with_explicit_none_on_non_optional_default_raises_invalid_argument() -> None:
    """`await todays_soup(None)` on `day: str = 'monday'` validates None and fails."""
    ai = Genkit()

    @ai.tool
    async def todays_soup(day: str = 'monday') -> str:
        return f'{day}: bouillabaisse'

    with pytest.raises(GenkitError) as exc:
        await todays_soup(None)

    assert exc.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_tool_with_no_input_uses_python_default() -> None:
    """`await todays_soup()` runs with the function's default."""
    ai = Genkit()

    @ai.tool
    async def todays_soup(day: str = 'monday') -> str:
        return f'{day}: bouillabaisse'

    assert await todays_soup() == 'monday: bouillabaisse'


@pytest.mark.asyncio
async def test_tool_with_optional_input_receives_explicit_none() -> None:
    """`x: Allergens | None = Allergens()` gets None when the caller passes None, the default when omitted."""
    ai = Genkit()
    seen: list[Allergens | None] = []

    @ai.tool
    async def check_order(allergens: Allergens | None = Allergens(nuts=True)) -> str:  # noqa: B008
        seen.append(allergens)
        return 'checked'

    await check_order(None)
    await check_order()

    assert seen == [None, Allergens(nuts=True)]


@pytest.mark.asyncio
async def test_model_tool_request_with_null_input_uses_python_default() -> None:
    """A model's tool request with `input: null` runs the tool with its default."""
    ai = Genkit()
    pm, _ = define_scripted_model(ai)

    @ai.tool
    async def todays_soup(day: str = 'monday') -> str:
        return f'{day}: bouillabaisse'

    # 1. The model calls the tool with no input, then answers
    pm.responses.append(
        ModelResponse(
            message=Message(
                role=Role.MODEL,
                content=[Part(tool_request=ToolRequest(name='todays_soup', input=None, ref='r1'))],
            )
        )
    )
    pm.responses.append(
        ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('done')]),
        )
    )

    # 2. The tool response carries the default-day soup
    response = await ai.generate(model='scriptedModel', prompt="What's the soup?", tools=['todays_soup'])
    part = response.messages[2].content[0]
    assert part.tool_response is not None
    assert part.tool_response.output == 'monday: bouillabaisse'


@asynccontextmanager
async def _dev_ui_client(ai: Genkit) -> AsyncIterator[AsyncClient]:
    client = AsyncClient(transport=ASGITransport(app=create_reflection_asgi_app(ai._registry)), base_url='http://test')
    try:
        yield client
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_dev_ui_run_tool_with_null_input_uses_python_default() -> None:
    """A Dev UI runAction on a tool with `"input": null` behaves like no input."""
    ai = Genkit()

    @ai.tool
    async def todays_soup(day: str = 'monday') -> str:
        return f'{day}: bouillabaisse'

    async with _dev_ui_client(ai) as client:
        response = await client.post('/api/runAction', json={'key': '/tool/todays_soup', 'input': None})

    assert response.status_code == 200
    assert response.json()['result'] == 'monday: bouillabaisse'
