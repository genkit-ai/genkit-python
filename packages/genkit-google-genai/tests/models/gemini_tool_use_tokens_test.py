# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tokens Gemini bills for built-in tool results (``tool_use_prompt_token_count``) reach ``usage.custom``."""

import pytest
from genkit_google_genai._models._gemini import GeminiModel
from google.genai import types as genai_types
from pytest_mock import MockerFixture

from genkit import Message, Part, Role
from genkit.model import ModelContext, ModelRequest


def _reply(**usage: int) -> genai_types.GenerateContentResponse:
    part = genai_types.Part(text='ok')
    return genai_types.GenerateContentResponse(
        candidates=[genai_types.Candidate(content=genai_types.Content(role='model', parts=[part]))],
        usage_metadata=genai_types.GenerateContentResponseUsageMetadata(**usage),
    )


async def _usage(mocker: MockerFixture, reply: genai_types.GenerateContentResponse):  # noqa: ANN202
    client = mocker.AsyncMock()
    client.aio.models.generate_content.return_value = reply
    request = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])])
    response = await GeminiModel('gemini-2.5-flash', client).generate(request, ModelContext())
    return response.usage


@pytest.mark.asyncio
async def test_tool_use_prompt_tokens_reported_in_custom(mocker: MockerFixture) -> None:
    """Tool-use prompt tokens are reported as ``custom.toolUseTokens`` (as in the JS plugin), not added to input."""
    usage = await _usage(
        mocker,
        _reply(prompt_token_count=10, candidates_token_count=5, total_token_count=22, tool_use_prompt_token_count=7),
    )

    assert usage.custom == {'toolUseTokens': 7}
    assert usage.input_tokens == 10


@pytest.mark.asyncio
async def test_no_custom_without_tool_use_tokens(mocker: MockerFixture) -> None:
    """Without tool-use tokens, ``usage.custom`` stays unset."""
    usage = await _usage(mocker, _reply(prompt_token_count=10, candidates_token_count=5, total_token_count=15))

    assert usage.custom is None
