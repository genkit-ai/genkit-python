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

"""Type-checker contract for ``.output``.

With ``output_schema=T``, ``response.output`` types as ``T | None``:
``generate`` boxes failures, so a reply
that doesn't match the schema is ``None`` at runtime. JS types it ``O | null``.
Stream chunks are ``T | None`` too. With no schema or a JSON-schema dict,
``.output`` types as ``Any | None``.

pyright, pyrefly, and ty check this file (see ``[tool.pyright]``); the tests
also run so each static type matches the runtime value.
"""

from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit.testing import define_scripted_model, define_static_response_model


class Order(BaseModel):
    dish: str
    quantity: int


def _scripted(ai: Genkit, text: str, chunks: list[str] | None = None) -> None:
    define_scripted_model(
        ai,
        name='kitchen',
        responses=[ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))],
        chunks=[[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(c)]) for c in chunks]] if chunks else None,
    )


@pytest.mark.asyncio
async def test_generate_output_is_schema_type() -> None:
    """``output_schema=Order`` gives ``Order | None``; a guard narrows it."""
    ai = Genkit()
    _scripted(ai, '{"dish": "tartine", "quantity": 2}')

    res = await ai.generate(model='kitchen', prompt='order', output_schema=Order)

    assert_type(res, ModelResponse[Order])
    assert_type(res.output, Order | None)
    assert res.output is not None
    total: int = res.output.quantity * 3
    assert total == 6


@pytest.mark.asyncio
async def test_generate_output_with_config_dict_is_schema_type() -> None:
    """The ``config={...}`` overload narrows the same way."""
    ai = Genkit()
    _scripted(ai, '{"dish": "soup", "quantity": 1}')

    res = await ai.generate(model='kitchen', prompt='order', config={'temperature': 0.1}, output_schema=Order)

    assert_type(res.output, Order | None)
    assert res.output is not None
    assert res.output.dish == 'soup'


@pytest.mark.asyncio
async def test_untyped_output_is_any() -> None:
    """No schema or a dict schema types as ``Any | None``; a guard makes it indexable."""
    ai = Genkit()
    _, _ = define_static_response_model(ai, '{"dish": "tartine"}', name='menu')

    res = await ai.generate(model='menu', prompt='x')
    assert_type(res.output, Any | None)

    res_dict = await ai.generate(model='menu', prompt='x', output_schema={'type': 'object'})
    assert_type(res_dict.output, Any | None)
    assert res_dict.output is not None
    assert res_dict.output['dish'] == 'tartine'


@pytest.mark.asyncio
async def test_stream_chunk_and_final_output_are_optional() -> None:
    """Chunks and the awaited response are both ``Order | None``."""
    ai = Genkit()
    _scripted(ai, '{"dish": "tartine", "quantity": 2}', chunks=['{"dish": "tar', 'tine", "quantity": 2}'])

    stream = ai.generate_stream(model='kitchen', prompt='order', output_schema=Order)
    seen: list[Order | None] = []
    async for chunk in stream.stream:
        assert_type(chunk, ModelResponseChunk[Order])
        assert_type(chunk.output, Order | None)
        seen.append(chunk.output)
    final = await stream.response

    assert_type(final.output, Order | None)
    assert final.output is not None
    assert final.output.quantity == 2
    last = seen[-1]
    assert last is not None and last.dish == 'tartine'


@pytest.mark.asyncio
async def test_prompt_output_is_optional() -> None:
    """``define_prompt`` / ``ai.prompt`` with ``output_schema`` type ``.output`` as ``Order | None``."""
    ai = Genkit()
    _, _ = define_static_response_model(ai, '{"dish": "tartine", "quantity": 2}', name='kitchen')
    order_prompt = ai.define_prompt(name='order', model='kitchen', prompt='order', output_schema=Order)

    res = await order_prompt()
    assert_type(res.output, Order | None)
    assert res.output is not None
    assert res.output.dish == 'tartine'

    looked_up = ai.prompt('order', output_schema=Order)
    looked_up_res = await looked_up()
    assert_type(looked_up_res.output, Order | None)
    assert looked_up_res.output is not None
    assert looked_up_res.output.quantity == 2

    stream = order_prompt.stream()
    async for chunk in stream.stream:
        assert_type(chunk.output, Order | None)
    final = await stream.response
    assert_type(final.output, Order | None)
    assert final.output is not None
    assert final.output.dish == 'tartine'


def test_bare_model_response_output_is_object() -> None:
    """Unparameterized ``ModelResponse`` defaults ``OutputT`` to ``object``, so ``.output`` is ``object | None``."""

    def read(res: ModelResponse) -> object:
        assert_type(res.output, object | None)
        return res.output

    assert read(ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('hi')]))) is None
