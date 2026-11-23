# Copyright 2025 Google LLC
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

"""Integration tests for Ollama plugin with Genkit."""

from typing import Any
from unittest.mock import Mock

import ollama as ollama_api
import pytest
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, Part, Role


@pytest.mark.asyncio
async def test_adding_ollama_chat_model_to_genkit_veneer(
    ollama_model: str,
    genkit_veneer_chat_model: Genkit,
) -> None:
    """Test adding ollama chat model to genkit veneer."""
    action = await genkit_veneer_chat_model.lookup_model(ollama_model)
    assert action is not None


@pytest.mark.asyncio
async def test_async_get_chat_model_response_from_llama_api_flow(
    mock_ollama_api_async_client: Mock,
    genkit_veneer_chat_model: Genkit,
) -> None:
    """Test async get chat model response from llama api flow."""
    mock_response_message = 'Mocked response message'

    async def fake_chat_response(*args: object, **kwargs: object) -> ollama_api.ChatResponse:
        return ollama_api.ChatResponse(
            message=ollama_api.Message(
                content=mock_response_message,
                role=Role.USER,
            )
        )

    mock_ollama_api_async_client.return_value.chat.side_effect = fake_chat_response

    async def _test_fun() -> ModelResponse:
        return await genkit_veneer_chat_model.generate(
            messages=[
                Message(
                    role=Role.USER,
                    content=[
                        Part.from_text('Test message'),
                    ],
                )
            ]
        )

    response = await genkit_veneer_chat_model.flow()(_test_fun)()

    assert isinstance(response, ModelResponse)
    assert response.message is not None
    assert response.message.content[0].text == mock_response_message


class _NoInput(BaseModel):
    pass


@pytest.mark.asyncio
async def test_tool_turn_is_sent_back_with_its_call(
    mock_ollama_api_async_client: Mock,
    genkit_veneer_chat_model: Genkit,
) -> None:
    """The follow-up request carries the model's tool call and the named tool result."""
    ai = genkit_veneer_chat_model
    sent: list[list[ollama_api.Message]] = []
    replies = [
        ollama_api.Message(
            role='assistant',
            content='',
            tool_calls=[
                ollama_api.Message.ToolCall(
                    function=ollama_api.Message.ToolCall.Function(name='getCurrentTime', arguments={})
                )
            ],
        ),
        ollama_api.Message(role='assistant', content='It is 18:00.'),
    ]

    async def fake_chat(*args: object, **kwargs: Any) -> ollama_api.ChatResponse:
        sent.append(list(kwargs['messages']))
        return ollama_api.ChatResponse(message=replies[len(sent) - 1])

    mock_ollama_api_async_client.return_value.chat.side_effect = fake_chat

    @ai.tool(name='getCurrentTime')
    async def get_current_time(_: _NoInput) -> str:
        """Returns the current time."""
        return '18:00'

    response = await ai.generate(prompt='What time is it?', tools=['getCurrentTime'], max_turns=2)

    assert response.text == 'It is 18:00.'
    assert len(sent) == 2
    assistant, tool = sent[1][-2:]
    assert [call.function.name for call in assistant.tool_calls or []] == ['getCurrentTime']
    assert (tool.role, tool.tool_name, tool.content) == ('tool', 'getCurrentTime', '18:00')


# Integration tests are covered by the above test cases
