#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What app code and middleware get back from lookups, listings, lookup_value, and define_value."""

import re
from collections.abc import Awaitable, Callable
from pathlib import Path

import pytest
from pydantic import BaseModel

from genkit import (
    FinishReason,
    Genkit,
    GenkitError,
    Message,
    ModelResponse,
    Operation,
    Part,
    Prompt,
    PromptDescription,
    Role,
    Tool,
)
from genkit._core._typing import MiddlewareRef
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams
from genkit.model import GenerateActionOptions, Model, ModelContext, ModelRequest
from genkit.plugin_api import Action, ActionKind, Plugin
from genkit.testing import define_echo_model

NextModel = Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]]


def _answer(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


def _define_answering_model(ai: Genkit, name: str, text: str) -> None:
    async def fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return _answer(text)

    ai.define_model(name=name, fn=fn)


class LazyPlugin(Plugin):
    """Serves `lazy/fast` only through resolve, never from init."""

    name = 'lazy'

    async def init(self) -> list[Action]:
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        if action_type != ActionKind.MODEL or name != 'fast':
            return None

        async def fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
            return _answer('from lazy fast')

        return Model(name, fn)

    async def list_actions(self) -> list[Action]:
        return []


@pytest.mark.asyncio
async def test_lookup_model_returns_registered_model_action() -> None:
    """`await ai.lookup_model('test/echo')` returns the model action, and running it gets that model's answer."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'echoed')

    action = await ai.lookup_model('test/echo')

    assert isinstance(action, Model)
    assert action.name == 'test/echo'
    result = await action.run(ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('hi')])]))
    assert isinstance(result.response, ModelResponse)
    assert result.response.text == 'echoed'


@pytest.mark.asyncio
async def test_lookup_model_unknown_name_returns_none() -> None:
    """`await ai.lookup_model('nope')` returns None instead of raising."""
    ai = Genkit()

    assert await ai.lookup_model('nope') is None


@pytest.mark.asyncio
async def test_lookup_model_resolves_plugin_model_not_yet_loaded() -> None:
    """A plugin model only reachable through the plugin's resolve comes back from `lookup_model` before any generate."""
    ai = Genkit(plugins=[LazyPlugin()])

    action = await ai.lookup_model('lazy/fast')

    assert action is not None
    assert action.name == 'lazy/fast'
    result = await action.run(ModelRequest(messages=[]))
    assert result.response.text == 'from lazy fast'


@pytest.mark.asyncio
async def test_lookup_model_does_not_return_background_models() -> None:
    """A background model runs through `generate_operation`, so `lookup_model` returns None for it."""
    ai = Genkit()

    async def start(request: ModelRequest, ctx: ModelContext) -> Operation:
        return Operation(id='op-1')

    async def check(operation: Operation, ctx: ModelContext) -> Operation:
        return operation

    ai.define_background_model('veo', start=start, check=check)

    assert await ai.lookup_model('veo') is None


@pytest.mark.asyncio
async def test_lookup_tool_returns_a_callable_tool_with_the_same_definition() -> None:
    """`ai.lookup_tool('menu_price')` gives back a Tool that runs and describes itself like the decorated one."""
    ai = Genkit()

    @ai.tool()
    async def menu_price(dish: str) -> float:
        """Price of a dish in dollars."""
        return 14.5

    found = await ai.lookup_tool('menu_price')

    assert isinstance(found, Tool)
    assert found.definition() == menu_price.definition()
    assert await found('ramen') == 14.5


@pytest.mark.asyncio
async def test_lookup_tool_unknown_name_returns_none() -> None:
    """A name nothing answers to returns None."""
    ai = Genkit()

    assert await ai.lookup_tool('missing') is None


@pytest.mark.asyncio
async def test_lookup_tool_finds_interrupts() -> None:
    """An interrupt is a tool, so `lookup_tool` finds it."""
    ai = Genkit()
    ai.define_interrupt('confirm_order', description='Ask the diner to confirm.')

    found = await ai.lookup_tool('confirm_order')

    assert found is not None
    assert found.name == 'confirm_order'


@pytest.mark.asyncio
async def test_lookup_tool_wraps_plain_tool_action_via_from_action() -> None:
    """A hand-registered ``Action(kind=ActionKind.TOOL, ...)`` comes back from ``lookup_tool`` as a ``Tool``."""
    ai = Genkit()

    async def table_status(table_id: str) -> str:
        return f'table {table_id}: seated'

    raw_action = Action(kind=ActionKind.TOOL, name='table_status', fn=table_status, description='Check table status.')
    ai._registry.register_action_from_instance(raw_action)

    found = await ai.lookup_tool('table_status')

    assert isinstance(found, Tool)
    assert found.wire_action is None
    assert found.metadata is raw_action.metadata
    assert found.definition().output_schema == {'type': 'string'}
    assert await found('T4') == 'table T4: seated'
    assert Tool.from_action(found) is found


class Guest(BaseModel):
    name: str


def _prompt_names(prompts: list[Prompt]) -> list[str | None]:
    return [prompt.name for prompt in prompts]


@pytest.mark.asyncio
async def test_list_prompts_returns_prompt_handles_sorted_by_name() -> None:
    """`await ai.list_prompts()` gives one Prompt handle per registered prompt, sorted by registered name."""
    ai = Genkit()
    ai.define_prompt(name='welcome', prompt='Welcome {{name}}')
    ai.define_prompt(name='greet', prompt='Hello {{name}}')

    prompts = await ai.list_prompts()

    assert _prompt_names(prompts) == ['greet', 'welcome']
    assert all(isinstance(prompt, Prompt) for prompt in prompts)


@pytest.mark.asyncio
async def test_list_prompts_on_an_app_without_prompts_is_empty() -> None:
    """An app that defines no prompts lists none."""
    assert await Genkit().list_prompts() == []


@pytest.mark.asyncio
async def test_list_prompts_lists_each_variant_as_its_own_prompt() -> None:
    """A variant is listed as its own entry named `name.variant`, next to the base prompt."""
    ai = Genkit()
    ai.define_prompt(name='greet', variant='formal', prompt='Good day {{name}}')
    ai.define_prompt(name='greet', prompt='Hello {{name}}')

    prompts = await ai.list_prompts()

    assert _prompt_names(prompts) == ['greet', 'greet.formal']


@pytest.mark.asyncio
async def test_list_prompts_lists_prompt_files_without_partials(tmp_path: Path) -> None:
    """`.prompt` files are listed by file name, a subfolder file keeps its folder, and partials are not listed."""
    (tmp_path / 'greet.prompt').write_text('{{> header}}Hello {{name}}\n')
    (tmp_path / 'greet.formal.prompt').write_text('Good day {{name}}\n')
    (tmp_path / '_header.prompt').write_text('Menu: ')
    (tmp_path / 'myns').mkdir()
    (tmp_path / 'myns' / 'a.prompt').write_text('From a folder\n')
    ai = Genkit(prompt_dir=tmp_path)

    prompts = await ai.list_prompts()

    assert _prompt_names(prompts) == ['greet', 'greet.formal', 'myns/a']


@pytest.mark.asyncio
async def test_listed_prompt_generates_like_ai_prompt() -> None:
    """Calling a listed prompt sends the same request and gets the same answer as `ai.prompt(name)`."""
    ai = Genkit(model='echoModel')
    define_echo_model(ai)
    ai.define_prompt(name='greet', variant='formal', prompt='Good day {{name}}')

    [listed] = await ai.list_prompts()
    listed_response = await listed({'name': 'Ada'})
    looked_up_response = await ai.prompt('greet', variant='formal')({'name': 'Ada'})

    assert listed_response.text == '[ECHO] user: "Good day Ada"'
    assert listed_response.text == looked_up_response.text


@pytest.mark.asyncio
async def test_listed_prompt_renders_without_calling_the_model() -> None:
    """`await listed.render(input)` returns the messages the model would get, and the model never runs."""
    ai = Genkit(model='echoModel')
    echo, _ = define_echo_model(ai)
    ai.define_prompt(name='greet', system='Be brief.', prompt='Hello {{name}}')

    [listed] = await ai.list_prompts()
    rendered = await listed.render({'name': 'Ada'})

    assert [(message.role, message.text) for message in rendered.messages] == [
        (Role.SYSTEM, 'Be brief.'),
        (Role.USER, 'Hello Ada'),
    ]
    assert echo.last_request is None


@pytest.mark.asyncio
async def test_prompt_name_includes_variant_and_namespace(tmp_path: Path) -> None:
    """`prompt.name` is `greet.formal` for a variant, `myns/a` for a subfolder file, and None for an unnamed prompt."""
    (tmp_path / 'myns').mkdir()
    (tmp_path / 'myns' / 'a.prompt').write_text('From a folder\n')
    ai = Genkit(prompt_dir=tmp_path)

    defined = ai.define_prompt(name='greet', variant='formal', prompt='Good day {{name}}')
    unnamed = ai.define_prompt(prompt='Hi')

    assert defined.name == 'greet.formal'
    assert ai.prompt('greet', variant='formal').name == 'greet.formal'
    assert ai.prompt('myns/a').name == 'myns/a'
    assert unnamed.name is None


@pytest.mark.asyncio
async def test_prompt_describe_gives_description_and_input_schema() -> None:
    """`await prompt.describe()` gives the name, description, and input JSON schema, or None for what's not set."""
    ai = Genkit()
    greet = ai.define_prompt(name='greet', description='Say hello.', input_schema=Guest, prompt='Hello {{name}}')
    ai.define_prompt(name='plain', prompt='Hi')

    assert await greet.describe() == PromptDescription(
        name='greet', description='Say hello.', input_schema=Guest.model_json_schema()
    )
    assert [await prompt.describe() for prompt in await ai.list_prompts()] == [
        PromptDescription(name='greet', description='Say hello.', input_schema=Guest.model_json_schema()),
        PromptDescription(name='plain', description=None, input_schema=None),
    ]


@pytest.mark.asyncio
async def test_prompt_describe_loads_a_prompt_file_never_run_before(tmp_path: Path) -> None:
    """`describe()` on a `.prompt` file nothing has run gives its frontmatter description and input schema."""
    (tmp_path / 'greet.prompt').write_text(
        '---\ndescription: Greets a guest.\ninput:\n  schema:\n    name: string\n---\nHello {{name}}\n'
    )
    ai = Genkit(prompt_dir=tmp_path)

    description = await ai.prompt('greet').describe()

    assert description.name == 'greet'
    assert description.description == 'Greets a guest.'
    assert description.input_schema is not None
    assert description.input_schema['properties'] == {'name': {'type': 'string'}}


@pytest.mark.asyncio
async def test_prompt_describe_works_on_an_ai_prompt_handle_never_run() -> None:
    """`ai.prompt('greet').describe()` on a fresh handle gives what the prompt was defined with."""
    ai = Genkit()
    ai.define_prompt(name='greet', description='Say hello.', input_schema=Guest, prompt='Hello {{name}}')

    assert await ai.prompt('greet').describe() == PromptDescription(
        name='greet', description='Say hello.', input_schema=Guest.model_json_schema()
    )


@pytest.mark.asyncio
async def test_prompt_describe_raises_not_found_for_an_unknown_prompt() -> None:
    """`ai.prompt('nope').describe()` raises the same NOT_FOUND GenkitError that calling the prompt raises."""
    ai = Genkit(model='echoModel')
    define_echo_model(ai)

    with pytest.raises(GenkitError) as described:
        await ai.prompt('nope').describe()
    with pytest.raises(GenkitError) as called:
        await ai.prompt('nope')()

    assert described.value.status == 'NOT_FOUND'
    assert str(described.value) == str(called.value)


def test_define_value_then_lookup_value_returns_it() -> None:
    """A value stored with `ai.define_value(...)` comes back from `ai.lookup_value(...)` for the same kind and name."""
    ai = Genkit()
    catalog = {'id': 'banner', 'components': []}

    ai.define_value(kind='catalog', name='banner', value=catalog)

    assert ai.lookup_value(kind='catalog', name='banner') is catalog


def test_lookup_value_unknown_returns_none() -> None:
    """`ai.lookup_value(kind=..., name=...)` for nothing defined returns None."""
    ai = Genkit()
    ai.define_value(kind='catalog', name='banner', value={'id': 'banner'})

    assert ai.lookup_value(kind='catalog', name='other') is None
    assert ai.lookup_value(kind='other-kind', name='banner') is None


def test_define_value_same_name_twice_overwrites() -> None:
    """Re-running a notebook cell redefines the value instead of raising."""
    ai = Genkit()
    ai.define_value(kind='catalog', name='banner', value='first')
    ai.define_value(kind='catalog', name='banner', value='second')

    assert ai.lookup_value(kind='catalog', name='banner') == 'second'


def test_define_schema_same_name_twice_overwrites() -> None:
    from pydantic import BaseModel

    class Dish(BaseModel):
        name: str

    class DishV2(BaseModel):
        name: str
        allergens: list[str]

    ai = Genkit()
    ai.define_schema('Dish', Dish)
    ai.define_schema('Dish', DishV2)

    assert ai._registry.lookup_schema_type('Dish') is DishV2


@pytest.mark.asyncio
async def test_duplicate_middleware_registration_raises() -> None:
    """Middleware collisions raise instead of letting the last registration silently win."""

    class PluginRetry(Middleware):
        name = 'retry'
        description = 'Plugin retry middleware.'

        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: NextModel,
        ) -> ModelResponse:
            res = await next_fn(params, ctx)
            return _answer(f'plugin:{res.text}')

    class AppRetry(Middleware):
        name = 'retry'
        description = 'Conflicting app retry middleware.'

    class MiddlewarePlugin(Plugin):
        def __init__(self, plugin_name: str, mw_cls: type[Middleware]) -> None:
            self.name = plugin_name
            self._mw_cls = mw_cls

        async def init(self) -> list[Action]:
            return []

        async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
            return None

        async def list_actions(self) -> list[Action]:
            return []

        def list_middleware(self) -> list[type[Middleware]]:
            return [self._mw_cls]

    with pytest.raises(ValueError, match='value for kind "middleware" and name "retry" is already registered'):
        Genkit(plugins=[MiddlewarePlugin('first-mw', PluginRetry), MiddlewarePlugin('second-mw', AppRetry)])

    ai = Genkit(plugins=[MiddlewarePlugin('first-mw', PluginRetry)])
    _define_answering_model(ai, 'kitchen', 'salmon')

    with pytest.raises(ValueError, match='value for kind "middleware" and name "retry" is already registered'):
        ai.define_middleware(AppRetry)

    util = await ai._registry.resolve_action(ActionKind.UTIL, 'generate')  # pyright: ignore[reportPrivateUsage]
    assert util is not None
    out = await util.run(
        GenerateActionOptions(
            model='kitchen',
            messages=[Message(role=Role.USER, content=[Part.from_text('dish?')])],
            use=[MiddlewareRef(name='retry')],
        )
    )
    assert out.response.text == 'plugin:salmon'


def test_duplicate_format_registration_raises() -> None:
    """Registering a format under an existing name raises instead of overwriting."""
    from genkit.formats import FormatHandler, Formatter, FormatterConfig

    class DuplicateJsonFormat(Formatter):
        def __init__(self) -> None:
            super().__init__('json', FormatterConfig(content_type='application/json'))

        def handle(self, schema: dict[str, object] | None) -> FormatHandler[str, str]:
            return FormatHandler(
                message_parser=lambda msg: msg.text,
                chunk_parser=lambda chunk: chunk.text,
            )

    ai = Genkit()
    with pytest.raises(ValueError, match='value for kind "format" and name "json" is already registered'):
        ai.define_format(DuplicateJsonFormat())


def test_define_value_requires_keywords() -> None:
    """`ai.define_value('k', 'n', v)` raises TypeError, so swapped kind and name can't store under the wrong key."""
    ai = Genkit()

    with pytest.raises(TypeError):
        ai.define_value('catalog', 'banner', 'value')  # type: ignore[misc]  # pyright: ignore[reportCallIssue]

    assert ai.lookup_value(kind='catalog', name='banner') is None
    assert ai.lookup_value(kind='banner', name='catalog') is None


@pytest.mark.parametrize(
    ('kind', 'api'),
    [('middleware', 'define_middleware'), ('format', 'define_format'), ('defaultModel', 'Genkit(model=...)')],
)
def test_define_value_rejects_kinds_genkit_owns(kind: str, api: str) -> None:
    """`define_value` on a kind core reads with a fixed type raises and names the API to use."""
    ai = Genkit()

    with pytest.raises(ValueError, match=f'reserved; use {re.escape(api)}'):
        ai.define_value(kind=kind, name='audit', value={'not': 'the right type'})

    assert ai.lookup_value(kind=kind, name='audit') is None


def test_define_value_rejects_none() -> None:
    """`define_value(value=None)` raises, so None from `lookup_value` always means "not defined"."""
    ai = Genkit()

    with pytest.raises(ValueError, match='must not be None'):
        ai.define_value(kind='catalog', name='banner', value=None)

    ai.define_value(kind='catalog', name='banner', value='later')
    assert ai.lookup_value(kind='catalog', name='banner') == 'later'


def test_genkit_has_no_registry_attribute() -> None:
    """`ai.registry` raises AttributeError; app code uses the lookup and define methods."""
    ai = Genkit()

    with pytest.raises(AttributeError):
        _ = ai.registry  # type: ignore[attr-defined]  # pyright: ignore[reportAttributeAccessIssue]


@pytest.mark.asyncio
async def test_middleware_ctx_ai_is_the_app_on_every_generate_path() -> None:
    """`ctx.ai` is the app's Genkit for `ai.generate`, a prompt call, and Dev UI's `/util/generate`."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    seen: list[object] = []

    class PeekAi(Middleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            seen.append(ctx.ai)
            return await next_fn(params, ctx)

    ai.define_middleware(PeekAi, name='peek_ai')

    # 1. Veneer
    await ai.generate(model='test/echo', prompt='hi', use=[PeekAi()])

    # 2. Prompt
    prompt = ai.define_prompt(model='test/echo', prompt='hi', use=[PeekAi()])
    await prompt()

    # 3. Dev UI / reflection
    util = await ai._registry.resolve_action(ActionKind.UTIL, 'generate')  # pyright: ignore[reportPrivateUsage]
    assert util is not None
    await util.run(
        GenerateActionOptions(
            model='test/echo',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            use=[MiddlewareRef(name='peek_ai')],
        )
    )

    assert len(seen) == 3
    assert all(s is ai for s in seen)


@pytest.mark.asyncio
async def test_middleware_ctx_ai_lookup_model_finds_app_model() -> None:
    """Middleware calling `await ctx.ai.lookup_model('test/other')` gets the app's model and can answer with it."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    _define_answering_model(ai, 'test/other', 'from other')

    class SwapModel(Middleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            other = await ctx.ai.lookup_model('test/other')
            assert other is not None
            result = await other.run(params.request)
            return result.response

    response = await ai.generate(model='test/echo', prompt='hi', use=[SwapModel()])

    assert response.text == 'from other'


@pytest.mark.asyncio
async def test_middleware_ctx_ai_lookup_value_sees_app_value() -> None:
    """A value from `ai.define_value` is what middleware gets from `ctx.ai.lookup_value`."""
    ai = Genkit()
    _define_answering_model(ai, 'test/echo', 'from echo')
    ai.define_value(kind='greeting', name='default', value='hello')
    seen: list[object | None] = []

    class ReadValue(Middleware):
        async def wrap_model(
            self, params: ModelHookParams, ctx: MiddlewareContext, next_fn: NextModel
        ) -> ModelResponse:
            seen.append(ctx.ai.lookup_value(kind='greeting', name='default'))
            seen.append(ctx.ai.lookup_value(kind='greeting', name='missing'))
            return await next_fn(params, ctx)

    await ai.generate(model='test/echo', prompt='hi', use=[ReadValue()])

    assert seen == ['hello', None]
