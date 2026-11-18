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

"""Type-checker contract for model, embedder and background-model handler shapes.

A handler takes its request plus an optional context, in either order, the
same rule as flows and tools. Every supported shape type-checks at
``ai.define_*``, the plugin-level constructors (``Model``, ``Embedder``,
``BackgroundModel``) and the registry-level ``define_*``, and runs. A lambda
infers its parameter as the request. A ``ModelRequest[Cfg]`` request
keeps its config class with the context in either position. Two inputs or a
keyword-only parameter fail every checker and raise ``TypeError`` at
definition.
"""

import pytest
from pydantic import BaseModel

from genkit import Embedding, Genkit, Message, ModelResponse, ModelResponseChunk, Operation, Part, Role
from genkit._ai._embedding import define_embedder
from genkit._ai._model import define_model
from genkit._core._registry import Registry
from genkit.embedder import EmbedContext, Embedder, EmbedRequest, EmbedResponse
from genkit.model import BackgroundModel, Model, ModelConfig, ModelContext, ModelRequest


class SpiceConfig(ModelConfig):
    heat: int = 1


class MenuEmbedConfig(BaseModel):
    dimensions: int = 4


def _reply(text: str) -> ModelResponse:
    return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))


def _config_name(req: ModelRequest[SpiceConfig] | ModelRequest) -> str:
    return type(req.config).__name__


# Model handlers, one per shape.


async def chef(req: ModelRequest) -> ModelResponse:
    return _reply(f'chef: {len(req.messages)} messages')


async def chef_streaming(req: ModelRequest, ctx: ModelContext) -> ModelResponse:
    ctx.send_chunk(ModelResponseChunk(content=[Part.from_text('plating')]))
    return _reply('chef_streaming')


async def chef_ctx_first(ctx: ModelContext, req: ModelRequest) -> ModelResponse:
    del ctx
    return _reply(f'chef_ctx_first: {len(req.messages)} messages')


async def spicy_chef(req: ModelRequest[SpiceConfig]) -> ModelResponse:
    return _reply(_config_name(req))


async def spicy_chef_ctx_last(req: ModelRequest[SpiceConfig], ctx: ModelContext) -> ModelResponse:
    del ctx
    return _reply(_config_name(req))


async def spicy_chef_ctx_first(ctx: ModelContext, req: ModelRequest[SpiceConfig]) -> ModelResponse:
    del ctx
    return _reply(_config_name(req))


# Unsupported model shapes.


async def chef_two_inputs(req: ModelRequest, table: str) -> ModelResponse:
    return _reply(table)


async def chef_kwonly(req: ModelRequest, *, ctx: ModelContext) -> ModelResponse:
    del ctx
    return _reply('kwonly')


# Embedder handlers.


def _embed(req: EmbedRequest) -> EmbedResponse:
    return EmbedResponse(embeddings=[Embedding(embedding=[0.25]) for _ in req.input])


async def embed_menu(req: EmbedRequest) -> EmbedResponse:
    return _embed(req)


async def embed_menu_ctx_last(req: EmbedRequest, ctx: EmbedContext) -> EmbedResponse:
    del ctx
    return _embed(req)


async def embed_menu_ctx_first(ctx: EmbedContext, req: EmbedRequest) -> EmbedResponse:
    del ctx
    return _embed(req)


async def embed_typed_ctx_first(ctx: EmbedContext, req: EmbedRequest[MenuEmbedConfig]) -> EmbedResponse:
    del ctx
    assert isinstance(req.options, MenuEmbedConfig)
    return _embed(req)


async def embed_two_inputs(req: EmbedRequest, locale: str) -> EmbedResponse:
    del locale
    return _embed(req)


async def embed_kwonly(req: EmbedRequest, *, ctx: EmbedContext) -> EmbedResponse:
    del ctx
    return _embed(req)


# Background model handlers.


def _finished(op: Operation) -> Operation:
    return Operation(id=op.id, done=True, output={'message': {'role': 'model', 'content': [{'text': 'video ready'}]}})


async def start_render(req: ModelRequest[SpiceConfig]) -> Operation:
    return Operation(id=f'render-{_config_name(req)}')


async def start_render_ctx_last(req: ModelRequest[SpiceConfig], ctx: ModelContext) -> Operation:
    del ctx
    return Operation(id=f'render-{_config_name(req)}')


async def start_render_ctx_first(ctx: ModelContext, req: ModelRequest[SpiceConfig]) -> Operation:
    del ctx
    return Operation(id=f'render-{_config_name(req)}')


async def check_render(op: Operation[ModelResponse]) -> Operation:
    return _finished(op)


async def check_render_ctx_last(op: Operation[ModelResponse], ctx: ModelContext) -> Operation:
    del ctx
    return _finished(op)


async def check_render_ctx_first(ctx: ModelContext, op: Operation[ModelResponse]) -> Operation:
    del ctx
    return _finished(op)


MODEL_SHAPES = (chef, chef_streaming, chef_ctx_first, spicy_chef, spicy_chef_ctx_last, spicy_chef_ctx_first)
EMBEDDER_SHAPES = (embed_menu, embed_menu_ctx_last, embed_menu_ctx_first, embed_typed_ctx_first)


def test_model_shapes_type_check_at_every_entry_point() -> None:
    """``(req)``, ``(req, ctx)`` and ``(ctx, req)`` type-check at ai.define_model, Model and define_model."""
    ai = Genkit()
    registry = Registry()

    # 1. ai.define_model
    ai.define_model(name='chef', fn=chef)
    ai.define_model(name='chef_streaming', fn=chef_streaming)
    ai.define_model(name='chef_ctx_first', fn=chef_ctx_first)
    ai.define_model(name='spicy_chef', fn=spicy_chef)
    ai.define_model(name='spicy_chef_ctx_last', fn=spicy_chef_ctx_last)
    ai.define_model(name='spicy_chef_ctx_first', fn=spicy_chef_ctx_first)

    # 2. Model(...), what a plugin's init / resolve returns
    Model('chef', chef)
    Model('chef_streaming', chef_streaming)
    Model('chef_ctx_first', chef_ctx_first)
    Model('spicy_chef_ctx_first', spicy_chef_ctx_first)

    # 3. Registry-level define_model
    define_model(registry, 'chef', chef)
    define_model(registry, 'chef_streaming', chef_streaming)
    define_model(registry, 'chef_ctx_first', chef_ctx_first)
    define_model(registry, 'spicy_chef_ctx_first', spicy_chef_ctx_first)


def test_embedder_shapes_type_check_at_every_entry_point() -> None:
    """``(req)``, ``(req, ctx)`` and ``(ctx, req)`` type-check at ai.define_embedder, Embedder and define_embedder."""
    ai = Genkit()
    registry = Registry()

    ai.define_embedder(name='embed_menu', fn=embed_menu)
    ai.define_embedder(name='embed_menu_ctx_last', fn=embed_menu_ctx_last)
    ai.define_embedder(name='embed_menu_ctx_first', fn=embed_menu_ctx_first)
    ai.define_embedder(name='embed_typed_ctx_first', fn=embed_typed_ctx_first)

    Embedder('embed_menu', embed_menu)
    Embedder('embed_menu_ctx_last', embed_menu_ctx_last)
    Embedder('embed_menu_ctx_first', embed_menu_ctx_first)

    define_embedder(registry, 'embed_menu', embed_menu)
    define_embedder(registry, 'embed_menu_ctx_last', embed_menu_ctx_last)
    define_embedder(registry, 'embed_menu_ctx_first', embed_menu_ctx_first)


def test_background_model_shapes_type_check() -> None:
    """Start, check and cancel take their input plus an optional context, in either order, mixed freely."""
    ai = Genkit()

    ai.define_background_model(name='render', start=start_render, check=check_render, cancel=check_render)
    ai.define_background_model(
        name='render_ctx_last', start=start_render_ctx_last, check=check_render_ctx_last, cancel=check_render_ctx_last
    )
    ai.define_background_model(
        name='render_ctx_first',
        start=start_render_ctx_first,
        check=check_render_ctx_first,
        cancel=check_render_ctx_first,
    )
    ai.define_background_model(name='render_mixed', start=start_render_ctx_first, check=check_render)
    BackgroundModel('render_plugin', start=start_render_ctx_first, check=check_render_ctx_first)


def test_lambdas_infer_the_request_at_every_entry_point() -> None:
    """``lambda req: ...`` infers ``req`` as the request, not the context.

    A two-argument lambda can't run: its ``ctx`` has no ``ModelContext``
    annotation, so Genkit reads it as a second input. Background-model
    handlers must be ``async def``, so they take no lambdas at all.
    """
    ai = Genkit()
    registry = Registry()

    # 1. Models
    ai.define_model(name='m1', fn=lambda req: chef(req))
    Model('m2', lambda req: chef(req))
    define_model(registry, 'm3', lambda req: chef(req))

    # 2. Embedders
    ai.define_embedder(name='e1', fn=lambda req: embed_menu(req))
    Embedder('e2', lambda req: embed_menu(req))
    define_embedder(registry, 'e3', lambda req: embed_menu(req))


@pytest.mark.asyncio
async def test_lambda_handlers_run() -> None:
    """Lambda handlers answer generate and embed."""
    ai = Genkit()
    ai.define_model(name='chef_lambda', fn=lambda req: chef(req))
    ai.define_embedder(name='embed_lambda', fn=lambda req: embed_menu(req))

    assert (await ai.generate(model='chef_lambda', prompt='hi')).text == 'chef: 1 messages'
    assert (await ai.embed(embedder='embed_lambda', content='tartine'))[0].embedding == [0.25]


def test_unsupported_shapes_fail_at_definition() -> None:
    """Two inputs or a keyword-only parameter fail every checker and raise TypeError at definition."""
    ai = Genkit()

    # 1. Models
    with pytest.raises(TypeError, match="model 'chef_two_inputs' takes one input, but 'table' is a second parameter"):
        ai.define_model(name='chef_two_inputs', fn=chef_two_inputs)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match="model 'chef_kwonly' parameter 'ctx' is keyword-only"):
        ai.define_model(name='chef_kwonly', fn=chef_kwonly)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]

    # 2. Embedders
    with pytest.raises(TypeError, match="embedder 'embed_two_inputs' takes one input, but 'locale' is a second"):
        ai.define_embedder(name='embed_two_inputs', fn=embed_two_inputs)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match="embedder 'embed_kwonly' parameter 'ctx' is keyword-only"):
        ai.define_embedder(name='embed_kwonly', fn=embed_kwonly)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


@pytest.mark.asyncio
async def test_every_model_shape_runs_and_keeps_its_config_class() -> None:
    """Each model shape answers generate, and ``ModelRequest[SpiceConfig]`` gets a SpiceConfig either way round."""
    ai = Genkit()
    for fn in MODEL_SHAPES:
        ai.define_model(name=fn.__name__, fn=fn)

    # 1. Untyped shapes run with the context in either position
    assert (await ai.generate(model='chef', prompt='hi')).text == 'chef: 1 messages'
    assert (await ai.generate(model='chef_streaming', prompt='hi')).text == 'chef_streaming'
    assert (await ai.generate(model='chef_ctx_first', prompt='hi')).text == 'chef_ctx_first: 1 messages'

    # 2. A dict config validates into the annotated class in every order
    for name in ('spicy_chef', 'spicy_chef_ctx_last', 'spicy_chef_ctx_first'):
        res = await ai.generate(model=name, prompt='hi', config={'heat': 3})
        assert res.text == 'SpiceConfig', name
        model = await ai.lookup_model(name)
        assert model is not None
        assert model.config_schema is SpiceConfig, name


@pytest.mark.asyncio
async def test_every_embedder_shape_runs() -> None:
    """Each embedder shape answers embed; ``EmbedRequest[Cfg]`` keeps its options class with ctx first."""
    ai = Genkit()
    for fn in EMBEDDER_SHAPES:
        ai.define_embedder(name=fn.__name__, fn=fn)

    for fn in EMBEDDER_SHAPES:
        vecs = await ai.embed(embedder=fn.__name__, content='bouillabaisse')
        assert vecs[0].embedding == [0.25], fn.__name__

    vecs = await ai.embed(embedder='embed_typed_ctx_first', content='tartine', config={'dimensions': 8})
    assert vecs[0].embedding == [0.25]


@pytest.mark.asyncio
async def test_every_background_model_shape_runs() -> None:
    """Start, check and cancel run in every shape, and start keeps its config class."""
    ai = Genkit()
    pairs = (
        ('render', start_render, check_render),
        ('render_ctx_last', start_render_ctx_last, check_render_ctx_last),
        ('render_ctx_first', start_render_ctx_first, check_render_ctx_first),
    )
    for name, start, check in pairs:
        bm = ai.define_background_model(name=name, start=start, check=check, cancel=check)
        req = ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('a lobster reel')])])
        op = await bm.start(ModelRequest(messages=req.messages, config={'heat': 2}))
        assert op.id == 'render-SpiceConfig', name
        assert (await bm.check(op)).done, name
        assert (await bm.cancel(op)).done, name
