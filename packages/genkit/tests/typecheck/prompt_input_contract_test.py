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

"""Type-checker contract for Prompt.__call__ / Prompt.stream input.

A prompt's input stays optional, typed or not: the template can fill in
defaults, so ``await order()`` runs. Typed prompts also take a dict of
template variables. pyright, pyrefly, and ty check this directory.
"""

from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Genkit, ModelResponse, ModelStreamResponse
from genkit.testing import define_echo_model


class Order(BaseModel):
    dish: str = 'pad thai'


@pytest.mark.asyncio
async def test_prompt_input_is_optional_and_accepts_model_or_dict() -> None:
    """Untyped and typed prompts accept no input, the model, or a dict, with every keyword."""
    # 1. One prompt without input_schema, one with
    ai = Genkit()
    define_echo_model(ai)
    greeting = ai.define_prompt(name='greeting', model='echoModel', prompt='Welcome to the bistro.')
    order = ai.define_prompt(name='order', model='echoModel', prompt='Order {{dish}}.', input_schema=Order)

    # 2. No input: the model's defaults fill the template
    reply = await order()
    assert_type(reply, ModelResponse[Any])
    assert 'pad thai' in reply.text

    # 3. Model or dict input, with the full keyword set
    reply = await greeting(max_turns=1, context={}, resume_metadata=None, on_chunk=None)
    assert_type(reply, ModelResponse[Any])
    reply = await order(Order(dish='curry'), max_turns=1, resume_metadata=None)
    assert_type(reply, ModelResponse[Any])
    reply = await order({'dish': 'ramen'})
    assert 'ramen' in reply.text

    # 4. stream takes the same input shapes
    stream = greeting.stream(max_turns=1, context={}, resume_metadata=None)
    assert_type(stream, ModelStreamResponse[Any])
    stream = order.stream(resume_metadata=None)
    assert_type(stream, ModelStreamResponse[Any])
    stream = order.stream({'dish': 'tartine'})
    assert 'tartine' in (await stream.response).text
