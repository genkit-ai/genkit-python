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

"""Type-checker contract for Action input overloads, Ref invariance, and public exports.

Checked by pyright, pyrefly, and ty with unused-ignore errors enabled so every
negative assertion below must fail all three type checkers.
"""

from typing import Any

import pytest
from genkit_mcp import McpServerEntry
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import (
    Document,
    Embedding,
    FlowContext,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    Part,
    Role,
    StreamResponse,
    ToolContext,
)
from genkit._core._action import ActionResponse
from genkit.embedder import EmbedderRef, EmbedRequest, EmbedResponse
from genkit.model import ModelConfig, ModelContext, ModelFn, ModelRef, ModelRequest
from genkit.plugin_api import Action, ActionKind


class OrderReq(BaseModel):
    dish: str


class Invoice(BaseModel):
    total_cents: int


class BistroConfig(ModelConfig):
    spice_level: int = 1


class OtherModelConfig(ModelConfig):
    frames: int = 24


class MenuEmbedConfig(BaseModel):
    dimensions: int = 256


class OtherEmbedConfig(BaseModel):
    task_type: str = 'search'


@pytest.mark.asyncio
async def test_public_signature_exports_and_action_response() -> None:
    """``ModelFn`` and ``McpServerEntry`` import cleanly and ``action.run`` preserves ``ActionResponse[T]``."""
    ai = Genkit()

    async def bistro_model(req: ModelRequest, ctx: ModelContext) -> ModelResponse[Any]:
        del req, ctx
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('Smoked Salmon Tartine')]))

    fn: ModelFn = bistro_model
    ai.define_model(name='bistro/chef', fn=fn)

    entry: McpServerEntry = {'command': 'npx'}
    assert entry['command'] == 'npx'

    async def bill_order(order: OrderReq) -> Invoice:
        return Invoice(total_cents=1800 if order.dish == 'tartine' else 1200)

    action: Action[OrderReq, Invoice] = Action(ActionKind.CUSTOM, 'bill_order', bill_order)

    async def run_traced(act: Action[OrderReq, Invoice], req: OrderReq) -> ActionResponse[Invoice]:
        return await act.run(req)

    traced = await run_traced(action, OrderReq(dish='tartine'))
    assert_type(traced, ActionResponse[Invoice])
    assert_type(traced.response, Invoice)
    assert_type(traced.trace_id, str)
    assert traced.response.total_cents == 1800
    assert isinstance(traced.trace_id, str)


@pytest.mark.asyncio
async def test_typed_actions_require_input_while_unit_actions_omit_it() -> None:
    """Typed flows, tools, and actions reject missing ``input``; ``None``-input ones allow ``()``."""
    ai = Genkit()

    @ai.flow
    async def create_invoice(order: OrderReq, ctx: FlowContext[str]) -> Invoice:
        ctx.send_chunk(f'billing {order.dish}')
        return Invoice(total_cents=1800)

    @ai.flow
    async def kitchen_status(ctx: FlowContext[str]) -> str:
        ctx.send_chunk('checking')
        return 'open'

    @ai.tool
    async def lookup_allergens(order: OrderReq) -> str:
        return f'{order.dish}: gluten'

    @ai.tool
    async def list_specials(ctx: ToolContext) -> list[str]:
        del ctx
        return ['tartine']

    invoice = await create_invoice(OrderReq(dish='tartine'))
    assert_type(invoice, Invoice)
    assert invoice.total_cents == 1800

    run_res = await create_invoice.run(OrderReq(dish='tartine'))
    assert_type(run_res, ActionResponse[Invoice])

    stream_res = create_invoice.stream(OrderReq(dish='tartine'))
    assert_type(stream_res, StreamResponse[str, Invoice])
    assert (await stream_res.response).total_cents == 1800

    assert_type(await kitchen_status(), str)
    assert_type(await list_specials(), list[str])

    with pytest.raises(GenkitError):
        await create_invoice()  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]

    with pytest.raises(GenkitError):
        await create_invoice.run()  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]

    with pytest.raises(GenkitError):
        await lookup_allergens()  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]


@pytest.mark.asyncio
async def test_model_and_embedder_refs_are_invariant_and_default_to_any() -> None:
    """Typed refs reject mismatched configs while bare ``ModelRef`` / ``EmbedderRef`` accept any ref."""
    ai = Genkit()

    async def chef(req: ModelRequest[BistroConfig], ctx: ModelContext) -> ModelResponse[Any]:
        del ctx
        level = req.config.spice_level if isinstance(req.config, BistroConfig) else 1
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(f'spice={level}')]))

    async def embed_menu(req: EmbedRequest[MenuEmbedConfig]) -> EmbedResponse:
        return EmbedResponse(embeddings=[Embedding(embedding=[0.25]) for _ in req.input])

    ai.define_model(name='bistro/chef', fn=chef, config_schema=BistroConfig)
    ai.define_embedder(name='bistro/embed', fn=embed_menu, config_schema=MenuEmbedConfig)

    bistro_ref = ModelRef(name='bistro/chef', config_schema=BistroConfig)
    embed_ref = EmbedderRef(name='bistro/embed', config_schema=MenuEmbedConfig)

    # Bare ModelRef and EmbedderRef default their config parameter to Any.
    any_model_ref: ModelRef = bistro_ref
    any_embed_ref: EmbedderRef = embed_ref
    assert any_model_ref.name == 'bistro/chef'
    assert any_embed_ref.name == 'bistro/embed'

    reply = await ai.generate(model=bistro_ref, prompt='dish', config=BistroConfig(spice_level=3))
    assert reply.text == 'spice=3'

    vecs = await ai.embed_many(
        embedder=embed_ref,
        content=('tartine', Document.from_text('bouillabaisse')),
        config=MenuEmbedConfig(dimensions=128),
    )
    assert len(vecs) == 2

    with pytest.raises(GenkitError):
        await ai.generate(model=bistro_ref, prompt='dish', config=OtherModelConfig())  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]

    with pytest.raises(GenkitError):
        await ai.embed(embedder=embed_ref, content='tartine', config=OtherEmbedConfig())  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[bad-argument-type]  # ty: ignore[invalid-argument-type]
