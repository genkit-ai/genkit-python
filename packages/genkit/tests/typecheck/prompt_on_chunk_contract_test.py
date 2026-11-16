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

# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Type-checker contract for ``Prompt.__call__(on_chunk=...)``.

The callback receives ``ModelResponseChunk[OutputT]``: ``.text`` reads with no
cast, a prompt with a typed ``output_schema`` types the chunk's output, a
callback that takes ``object`` still fits, and one that takes an unrelated
type is rejected.
"""

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit.model import ModelContext, ModelRequest


class Dish(BaseModel):
    name: str


def _app() -> Genkit:
    ai = Genkit()

    async def line_cook(req: ModelRequest, ctx: ModelContext) -> ModelResponse:
        del req
        ctx.send_chunk(ModelResponseChunk(content=[Part.from_text('{"name": ')]))
        ctx.send_chunk(ModelResponseChunk(content=[Part.from_text('"tartine"}')]))
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"name": "tartine"}')]))

    ai.define_model(name='bistro/line-cook', fn=line_cook)
    return ai


@pytest.mark.asyncio
async def test_on_chunk_reads_text_without_a_cast() -> None:
    """``on_chunk=lambda c: ...`` sees a ModelResponseChunk, so ``.text`` type-checks."""
    ai = _app()
    special = ai.define_prompt(name='special', model='bistro/line-cook', prompt="Today's special?")
    seen: list[str] = []

    await special(on_chunk=lambda c: seen.append(c.text))

    assert seen == ['{"name": ', '"tartine"}']


@pytest.mark.asyncio
async def test_on_chunk_is_typed_by_the_output_schema() -> None:
    """A prompt with ``output_schema=Dish`` hands the callback ``ModelResponseChunk[Dish]``."""
    ai = _app()
    plate = ai.define_prompt(name='plate', model='bistro/line-cook', prompt='Plate it.', output_schema=Dish)
    chunks: list[ModelResponseChunk[Dish]] = []

    def on_dish(chunk: ModelResponseChunk[Dish]) -> None:
        assert_type(chunk.output, Dish | None)
        chunks.append(chunk)

    await plate(on_chunk=on_dish)

    assert len(chunks) == 2
    assert all(isinstance(c, ModelResponseChunk) for c in chunks)


@pytest.mark.asyncio
async def test_object_callbacks_still_fit_and_unrelated_ones_do_not() -> None:
    """A callback taking ``object`` still fits; one taking an unrelated type doesn't."""
    ai = _app()
    special = ai.define_prompt(name='special', model='bistro/line-cook', prompt="Today's special?")
    received: list[object] = []

    # 1. Callable params are contravariant: object accepts every chunk
    def log_anything(chunk: object) -> None:
        received.append(chunk)

    await special(on_chunk=log_anything)
    assert len(received) == 2
    assert all(isinstance(c, ModelResponseChunk) for c in received)

    # 2. A callback returning a value fits too
    await special(on_chunk=received.append)

    # 3. A callback for some other type doesn't
    def count_covers(covers: int) -> None:
        del covers

    await special(on_chunk=count_covers)  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
