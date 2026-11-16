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

"""``use=[instance]`` keeps the caller's instance; 0.12 hook names fail loudly."""

import re
from collections.abc import Awaitable, Callable
from typing import Any

import pytest
from pydantic import BaseModel, ConfigDict

from genkit import Genkit, Message, ModelResponse, Part, Role
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.model import ModelContext, ModelRequest

Next = Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]]


def _echo_app() -> Genkit:
    ai = Genkit()

    async def echo(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(request.messages[-1].text)]))

    ai.define_model(name='echo', fn=echo)
    return ai


class SecretGuard(Middleware):
    """Blocks prompts matching a pattern; holds a client set up in ``__init__``."""

    name = 'secret_guard'

    def __init__(self, pattern: re.Pattern[str], audit: list[str]) -> None:
        super().__init__()
        self.pattern = pattern
        self.audit = audit

    async def wrap_model(self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: Next) -> ModelResponse:
        text = params.request.messages[-1].text
        if self.pattern.search(text):
            self.audit.append(text)
            return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('[blocked]')]))
        return await next_fn(params, ctx)


@pytest.mark.asyncio
async def test_custom_init_and_regex_flags_survive_use() -> None:
    """The instance runs as built: no ``cls(config=...)`` crash, IGNORECASE kept."""
    ai = _echo_app()
    audit: list[str] = []
    guard = SecretGuard(re.compile('secret', re.IGNORECASE), audit)

    response = await ai.generate(model='echo', prompt='the SECRET menu', use=[guard])

    assert response.text == '[blocked]'
    assert audit == ['the SECRET menu']


class _CallableConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    redact: Callable[[str], str]


class Redact(Middleware[_CallableConfig]):
    name = 'redact'

    async def wrap_model(self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: Next) -> ModelResponse:
        response = await next_fn(params, ctx)
        return ModelResponse(
            message=Message(role=Role.MODEL, content=[Part.from_text(self.config.redact(response.text))])
        )


@pytest.mark.asyncio
async def test_callable_config_field_does_not_crash() -> None:
    ai = _echo_app()

    response = await ai.generate(
        model='echo', prompt='card 4111', use=[Redact(redact=lambda s: s.replace('4111', '****'))]
    )

    assert response.text == 'card ****'


@pytest.mark.asyncio
async def test_registered_middleware_instance_keeps_state() -> None:
    """A registered class passed as an instance still runs that instance."""
    ai = _echo_app()
    ai.define_middleware(SecretGuard)
    audit: list[str] = []

    await ai.generate(model='echo', prompt='secret sauce', use=[SecretGuard(re.compile('SECRET', re.I), audit)])

    assert audit == ['secret sauce']


def test_legacy_model_hook_raises_at_class_definition() -> None:
    with pytest.raises(TypeError, match=r'`model\(\)`.*wrap_model'):

        class PiiRedact(Middleware):  # pyright: ignore[reportUnusedClass]
            async def model(self, req: Any, ctx: Any, next_fn: Any) -> Any:  # noqa: ANN401
                return await next_fn(req, ctx)


def test_legacy_name_alongside_wrap_hook_is_allowed() -> None:
    class Both(Middleware):  # pyright: ignore[reportUnusedClass]
        async def tool(self, *args: Any) -> None:  # noqa: ANN401
            return None

        async def wrap_tool(self, params: Any, ctx: Any, next_fn: Any) -> Any:  # noqa: ANN401
            return await next_fn(params, ctx)

    class WithAttr(Middleware):  # pyright: ignore[reportUnusedClass]
        model = 'googleai/gemini-flash-lite-latest'


def test_helper_method_named_like_a_legacy_hook_is_allowed() -> None:
    """Only the 0.12 hook shape is rejected; a sync helper or other arity passes."""

    class AuditLog(Middleware):  # pyright: ignore[reportUnusedClass]
        def tool(self, name: str) -> str:
            return f'tool:{name}'

        async def model(self, order_id: str) -> str:
            return order_id

        def generate(self, req: Any, ctx: Any, next_fn: Any) -> None:  # noqa: ANN401
            return None


def test_legacy_tool_and_generate_hooks_raise() -> None:
    with pytest.raises(TypeError, match=r'`tool\(\)`.*wrap_tool'):

        class AllergyGate(Middleware):  # pyright: ignore[reportUnusedClass]
            async def tool(self, req: Any, ctx: Any, next_fn: Any) -> Any:  # noqa: ANN401
                return await next_fn(req, ctx)

    with pytest.raises(TypeError, match=r'`generate\(\)`.*wrap_generate'):

        class TurnLimit(Middleware):  # pyright: ignore[reportUnusedClass]
            async def generate(self, req: Any, ctx: Any, next_fn: Any) -> Any:  # noqa: ANN401
                return await next_fn(req, ctx)


@pytest.mark.asyncio
async def test_define_prompt_with_registered_callable_config_middleware() -> None:
    """ai.define_prompt(use=[Redact(...)]) does not crash on non-JSON config fields."""
    ai = _echo_app()
    ai.define_middleware(Redact)

    prompt = ai.define_prompt(
        name='p',
        model='echo',
        prompt='card 4111',
        use=[Redact(redact=lambda s: s.replace('4111', '****'))],
    )
    response = await prompt()

    assert response.text == 'card ****'


def test_resolve_middleware_from_use_mismatch_raises_genkit_error() -> None:
    """A mismatch between registered inline instances and use= fails with a clear GenkitError."""
    from genkit import GenkitError
    from genkit._ai._generate import register_middleware, resolve_middleware_from_use
    from genkit._core._typing import MiddlewareRef

    ai = _echo_app()
    registry: Any = ai._registry
    register_middleware(registry, [SecretGuard(re.compile('secret'), [])])

    with pytest.raises(GenkitError, match='Inline middleware instances on the per-call registry do not match'):
        resolve_middleware_from_use(registry, [MiddlewareRef(name='other')])
