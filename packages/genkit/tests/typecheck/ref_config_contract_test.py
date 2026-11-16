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

"""Type-checker contract for a ref's config class at every entry point.

Each entry point that takes a ``ModelRef`` / ``EmbedderRef`` has two arms.
With no ``config``, any ref fits, including one picked between providers
(``ModelRef[BistroConfig] | ModelRef[CafeConfig]``). With a ``config``, the
config is pinned to the ref's class: the right class or a dict passes, another
family's class fails, and so does a typed config on a ref picked between
providers. A typed ``output_schema`` types ``response.output`` in both arms.
"""

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Embedding, Genkit, GenkitError, Message, ModelResponse, Operation, Part, Role
from genkit.embedder import EmbedderRef, EmbedRequest, EmbedResponse
from genkit.exp import Genkit as ExpGenkit
from genkit.model import ModelConfig, ModelContext, ModelRef, ModelRequest


class BistroConfig(ModelConfig):
    spice_level: int = 1


class CafeConfig(ModelConfig):
    roast: str = 'medium'


class BistroEmbedConfig(BaseModel):
    dimensions: int = 4


class CafeEmbedConfig(BaseModel):
    task_type: str = 'menu'


class Dish(BaseModel):
    name: str


BISTRO = ModelRef(name='bistro/chef', config_schema=BistroConfig)
CAFE = ModelRef(name='cafe/barista', config_schema=CafeConfig)
BISTRO_REEL = ModelRef(name='bistro/reel', config_schema=BistroConfig)
CAFE_REEL = ModelRef(name='cafe/reel', config_schema=CafeConfig)
BISTRO_EMBED = EmbedderRef(name='bistro/embed', config_schema=BistroEmbedConfig)
CAFE_EMBED = EmbedderRef(name='cafe/embed', config_schema=CafeEmbedConfig)


def pick_model(use_bistro: bool) -> ModelRef[BistroConfig] | ModelRef[CafeConfig]:
    return BISTRO if use_bistro else CAFE


def pick_reel(use_bistro: bool) -> ModelRef[BistroConfig] | ModelRef[CafeConfig]:
    return BISTRO_REEL if use_bistro else CAFE_REEL


def pick_embedder(use_bistro: bool) -> EmbedderRef[BistroEmbedConfig] | EmbedderRef[CafeEmbedConfig]:
    return BISTRO_EMBED if use_bistro else CAFE_EMBED


def _app() -> Genkit:
    ai = Genkit()

    async def chef(req: ModelRequest[BistroConfig], ctx: ModelContext) -> ModelResponse:
        del req, ctx
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"name": "tartine"}')]))

    async def barista(req: ModelRequest[CafeConfig], ctx: ModelContext) -> ModelResponse:
        del req, ctx
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"name": "cortado"}')]))

    async def start_reel(req: ModelRequest, ctx: ModelContext) -> Operation:
        del req, ctx
        return Operation(id='reel-1')

    async def check_reel(op: Operation[ModelResponse], ctx: ModelContext) -> Operation:
        del ctx
        return op

    async def embed(req: EmbedRequest) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[0.25]) for _ in req.input])

    ai.define_model(name='bistro/chef', fn=chef, config_schema=BistroConfig)
    ai.define_model(name='cafe/barista', fn=barista, config_schema=CafeConfig)
    ai.define_background_model(name='bistro/reel', start=start_reel, check=check_reel, config_schema=BistroConfig)
    ai.define_background_model(name='cafe/reel', start=start_reel, check=check_reel, config_schema=CafeConfig)
    ai.define_embedder(name='bistro/embed', fn=embed, config_schema=BistroEmbedConfig)
    ai.define_embedder(name='cafe/embed', fn=embed, config_schema=CafeEmbedConfig)
    return ai


@pytest.mark.asyncio
async def test_generate_ref_config() -> None:
    """``generate``: any ref without config; a config pinned to the ref's class."""
    ai = _app()
    either = pick_model(use_bistro=False)

    # 1. A ref picked between providers, no config, untyped and typed output
    assert (await ai.generate(model=either, prompt='Suggest a drink.')).text == '{"name": "cortado"}'
    picked = await ai.generate(model=either, prompt='Suggest a drink.', output_schema=Dish)
    assert_type(picked.output, Dish | None)
    assert picked.output is not None
    assert picked.output.name == 'cortado'

    # 2. The ref's own config class or a dict, untyped and typed output
    await ai.generate(model=BISTRO, prompt='Suggest a dish.', config=BistroConfig(spice_level=3))
    await ai.generate(model=BISTRO, prompt='Suggest a dish.', config={'spice_level': 3})
    pinned = await ai.generate(model=BISTRO, prompt='Suggest a dish.', config=BistroConfig(), output_schema=Dish)
    assert_type(pinned.output, Dish | None)
    assert pinned.output is not None
    assert pinned.output.name == 'tartine'

    # 3. Another family's config fails, with a single ref or one picked between providers
    with pytest.raises(GenkitError):
        await ai.generate(model=BISTRO, prompt='Suggest a dish.', config=CafeConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        await ai.generate(model=either, prompt='Suggest a drink.', config=BistroConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


@pytest.mark.asyncio
async def test_generate_stream_ref_config() -> None:
    """``generate_stream``: same two arms as ``generate``."""
    ai = _app()
    either = pick_model(use_bistro=False)

    # 1. No config
    assert (await ai.generate_stream(model=either, prompt='Suggest a drink.').response).text == '{"name": "cortado"}'
    picked = await ai.generate_stream(model=either, prompt='Suggest a drink.', output_schema=Dish).response
    assert_type(picked.output, Dish | None)

    # 2. Pinned config or a dict
    await ai.generate_stream(model=BISTRO, prompt='Suggest a dish.', config=BistroConfig()).response
    await ai.generate_stream(model=BISTRO, prompt='Suggest a dish.', config={'spice_level': 2}).response
    pinned = await ai.generate_stream(model=BISTRO, prompt='x', config=BistroConfig(), output_schema=Dish).response
    assert_type(pinned.output, Dish | None)

    # 3. Wrong family
    with pytest.raises(GenkitError):
        await ai.generate_stream(model=BISTRO, prompt='Suggest a dish.', config=CafeConfig()).response  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        await ai.generate_stream(model=either, prompt='Suggest a drink.', config=BistroConfig()).response  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


@pytest.mark.asyncio
async def test_generate_operation_ref_config() -> None:
    """``generate_operation``: same two arms for background models."""
    ai = _app()
    either = pick_reel(use_bistro=False)

    assert (await ai.generate_operation(model=either, prompt='A pour-over.')).id == 'reel-1'
    await ai.generate_operation(model=BISTRO_REEL, prompt='A risotto.', config=BistroConfig())
    await ai.generate_operation(model=BISTRO_REEL, prompt='A risotto.', config={'spice_level': 2})

    with pytest.raises(GenkitError):
        await ai.generate_operation(model=BISTRO_REEL, prompt='A risotto.', config=CafeConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        await ai.generate_operation(model=either, prompt='A pour-over.', config=BistroConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


@pytest.mark.asyncio
async def test_embed_ref_config() -> None:
    """``embed`` and ``embed_many``: any ref without config; a config pinned to the ref's class."""
    ai = _app()
    either = pick_embedder(use_bistro=False)

    # 1. No config
    assert (await ai.embed(embedder=either, content='cortado'))[0].embedding == [0.25]
    assert len(await ai.embed_many(embedder=either, content=['cortado', 'flat white'])) == 2

    # 2. Pinned config or a dict
    await ai.embed(embedder=BISTRO_EMBED, content='tartine', config=BistroEmbedConfig(dimensions=8))
    await ai.embed(embedder=BISTRO_EMBED, content='tartine', config={'dimensions': 8})
    await ai.embed_many(embedder=BISTRO_EMBED, content=['tartine'], config=BistroEmbedConfig())
    await ai.embed_many(embedder=BISTRO_EMBED, content=['tartine'], config={'dimensions': 8})

    # 3. Wrong family, with a single ref or one picked between providers
    with pytest.raises(GenkitError):
        await ai.embed(embedder=BISTRO_EMBED, content='tartine', config=CafeEmbedConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        await ai.embed(embedder=either, content='cortado', config=BistroEmbedConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        await ai.embed_many(embedder=BISTRO_EMBED, content=['tartine'], config=CafeEmbedConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        await ai.embed_many(embedder=either, content=['cortado'], config=BistroEmbedConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


@pytest.mark.asyncio
async def test_define_prompt_ref_config() -> None:
    """``define_prompt``: any ref without config; a config pinned to the ref's class; typed output either way."""
    ai = _app()
    either = pick_model(use_bistro=False)

    # 1. No config, untyped and typed output
    drink = ai.define_prompt(name='drink', model=either, prompt='Suggest a drink.')
    typed_drink = ai.define_prompt(name='typed_drink', model=either, prompt='Suggest a drink.', output_schema=Dish)
    assert (await drink()).text == '{"name": "cortado"}'
    assert_type((await typed_drink()).output, Dish | None)

    # 2. Pinned config or a dict, typed output
    ai.define_prompt(name='dish', model=BISTRO, prompt='Suggest a dish.', config=BistroConfig())
    ai.define_prompt(name='dish_dict', model=BISTRO, prompt='Suggest a dish.', config={'spice_level': 2})
    typed_dish = ai.define_prompt(
        name='typed_dish', model=BISTRO, prompt='Suggest a dish.', config=BistroConfig(), output_schema=Dish
    )
    # ty < 0.0.33 bug: the overload result depends on keyword-argument order, so prompt=
    # before config= types this Unknown. Fixed upstream (astral-sh/ty#3139). Remove when
    # ty is bumped.
    assert_type((await typed_dish()).output, Dish | None)  # ty: ignore[type-assertion-failure]

    # 3. Wrong family
    with pytest.raises(GenkitError):
        ai.define_prompt(name='bad', model=BISTRO, prompt='x', config=CafeConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        ai.define_prompt(name='bad_either', model=either, prompt='x', config=BistroConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


def test_define_agent_ref_config() -> None:
    """``genkit.exp`` ``define_agent``: same two arms."""
    ai = ExpGenkit()
    either = pick_model(use_bistro=False)

    ai.define_agent('host', model=either, system='Seat guests.')
    ai.define_agent('sommelier', model=BISTRO, config=BistroConfig())
    ai.define_agent('sommelier_dict', model=BISTRO, config={'spice_level': 2})
    with pytest.raises(GenkitError):
        ai.define_agent('bad', model=BISTRO, config=CafeConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(GenkitError):
        ai.define_agent('bad_either', model=either, config=BistroConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
