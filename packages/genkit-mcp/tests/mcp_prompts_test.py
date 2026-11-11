# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests serving Genkit prompts to an in-memory MCP client."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path
from typing import Any

import pytest
from genkit_mcp._prompts import PromptArgumentError, prompt_input, request_to_messages
from genkit_mcp._server import McpServer, create_mcp_server
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import INTERNAL_ERROR, INVALID_PARAMS, ImageContent, PromptMessage, ResourceLink, TextContent
from pydantic import BaseModel, Field

from genkit import (
    Document,
    FinishReason,
    Genkit,
    GenkitError,
    Media,
    Message,
    ModelResponse,
    Part,
    PublicError,
    Role,
)
from genkit.model import ModelContext, ModelRequest

PIXEL = 'iVBORw0KGgo='


class Person(BaseModel):
    """Input of a prompt whose fields are all text."""

    name: str


class Order(BaseModel):
    """Input of a prompt whose fields MCP sends as strings but are not text."""

    count: int
    vip: bool
    items: list[str]


@asynccontextmanager
async def connected(server: McpServer) -> AsyncIterator[ClientSession]:
    """Drive a server over the SDK's in-memory transport."""
    async with create_connected_server_and_client_session(server._server, raise_exceptions=True) as session:
        yield session


def app(prompt_dir: Path | None = None) -> tuple[Genkit, list[ModelRequest]]:
    """An app with a default model that records every call it gets."""
    ai = Genkit(model='test/recorder', prompt_dir=prompt_dir)
    calls: list[ModelRequest] = []

    async def recorder(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        calls.append(request)
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=Message(role=Role.MODEL, content=[Part.from_text('unused')]),
        )

    ai.define_model(name='test/recorder', fn=recorder)
    return ai, calls


def texts(messages: list[PromptMessage]) -> list[tuple[str, str]]:
    """Read each message as its role and text, failing on any other content."""
    out: list[tuple[str, str]] = []
    for message in messages:
        assert isinstance(message.content, TextContent)
        out.append((message.role, message.content.text))
    return out


async def get_error(session: ClientSession, name: str, arguments: dict[str, str] | None = None) -> McpError:
    """Ask for a prompt that must fail, and return the error the client got."""
    with pytest.raises(McpError) as caught:
        await session.get_prompt(name, arguments)
    return caught.value


@pytest.mark.asyncio
async def test_a_prompt_defined_after_the_server_is_listed_with_its_arguments() -> None:
    """Listing reads the app per request, and each input field becomes one argument."""
    ai, _ = app()
    server = create_mcp_server(ai, 'bookshop')
    ai.define_prompt(name='order', description='Place an order.', input_schema=Order, prompt='{{count}}')

    async with connected(server) as session:
        [prompt] = (await session.list_prompts()).prompts

    assert prompt.name == 'order'
    assert prompt.description == 'Place an order.'
    assert [(argument.name, argument.required) for argument in prompt.arguments or []] == [
        ('count', True),
        ('items', True),
        ('vip', True),
    ]


@pytest.mark.asyncio
async def test_getting_a_prompt_renders_it_without_calling_the_model() -> None:
    """A system turn goes out as a marked user message, and the model never runs."""
    ai, calls = app()
    ai.define_prompt(
        name='greet', description='Say hello.', input_schema=Person, system='Be brief.', prompt='Hello {{name}}'
    )

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.get_prompt('greet', {'name': 'Ada'})

    assert result.description == 'Say hello.'
    assert texts(result.messages) == [('user', 'System instructions:\nBe brief.'), ('user', 'Hello Ada')]
    assert calls == []


@pytest.mark.asyncio
async def test_a_variant_is_its_own_prompt() -> None:
    ai, _ = app()
    ai.define_prompt(name='greet', input_schema=Person, prompt='Hello {{name}}')
    ai.define_prompt(name='greet', variant='formal', input_schema=Person, prompt='Good day, {{name}}')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        listed = sorted(prompt.name for prompt in (await session.list_prompts()).prompts)
        result = await session.get_prompt('greet.formal', {'name': 'Ada'})

    assert listed == ['greet', 'greet.formal']
    assert texts(result.messages) == [('user', 'Good day, Ada')]


@pytest.mark.asyncio
async def test_string_arguments_are_read_as_the_types_the_schema_names() -> None:
    """A `false` that stayed a string would be truthy and print VIP."""
    ai, _ = app()
    ai.define_prompt(
        name='order',
        input_schema=Order,
        prompt='{{#if vip}}VIP {{/if}}{{count}} of {{#each items}}{{this}};{{/each}}',
    )

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.get_prompt('order', {'count': '3', 'vip': 'false', 'items': '["tea", "cake"]'})

    assert texts(result.messages) == [('user', '3 of tea;cake;')]


@pytest.mark.asyncio
async def test_an_argument_that_does_not_parse_is_invalid_params() -> None:
    ai, _ = app()
    ai.define_prompt(name='order', input_schema=Order, prompt='{{count}}')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'order', {'count': 'many', 'vip': 'true', 'items': '[]'})

    assert error.error.code == INVALID_PARAMS
    assert "'count'" in error.error.message


@pytest.mark.asyncio
async def test_a_missing_required_argument_is_invalid_params() -> None:
    """The prompt's own schema check catches it, after its defaults are filled in, and names the field."""
    ai, _ = app()
    ai.define_prompt(name='greet', input_schema=Person, prompt='Hello {{name}}')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'greet', {})

    assert error.error.code == INVALID_PARAMS
    assert error.error.message == "Invalid input for prompt 'greet': name: Field required"


@pytest.mark.asyncio
async def test_a_default_fills_an_argument_the_client_leaves_out(tmp_path: Path) -> None:
    (tmp_path / 'greet.prompt').write_text(
        '---\ninput:\n  schema:\n    name: string\n  default:\n    name: friend\n---\nHello {{name}}\n'
    )
    ai, _ = app(prompt_dir=tmp_path)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.get_prompt('greet')

    assert texts(result.messages) == [('user', 'Hello friend')]


@pytest.mark.asyncio
async def test_an_unknown_prompt_is_invalid_params() -> None:
    ai, _ = app()

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'missing')

    assert error.error.code == INVALID_PARAMS
    assert error.error.message == "Prompt 'missing' is not registered"


@pytest.mark.asyncio
async def test_a_prompt_without_an_input_schema_takes_its_fields_as_json() -> None:
    ai, _ = app()
    ai.define_prompt(name='greet', prompt='Hello {{name}}')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        [prompt] = (await session.list_prompts()).prompts
        result = await session.get_prompt('greet', {'inputJson': '{"name": "Ada"}'})

    assert [(argument.name, argument.required) for argument in prompt.arguments or []] == [('inputJson', False)]
    assert texts(result.messages) == [('user', 'Hello Ada')]


@pytest.mark.asyncio
async def test_an_app_without_a_model_cannot_render() -> None:
    """The cause is in the server's log; the client gets only the status."""
    ai = Genkit()
    ai.define_prompt(name='greet', input_schema=Person, prompt='Hello {{name}}')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'greet', {'name': 'Ada'})

    assert error.error.code == INTERNAL_ERROR
    assert error.error.message == "Prompt 'greet' failed to render: INVALID_ARGUMENT"


@pytest.mark.asyncio
async def test_a_render_that_raises_keeps_its_text_off_the_wire(monkeypatch: pytest.MonkeyPatch) -> None:
    """Any error that is not a GenkitError can carry a URL or a credential."""
    ai, _ = app()
    ai.define_prompt(name='greet', input_schema=Person, prompt='Hello {{name}}')
    [action] = await ai.list_prompts()

    async def explode(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise RuntimeError('https://user:secret@example.com')

    monkeypatch.setattr(action, 'run', explode)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'greet', {'name': 'Ada'})

    assert error.error.code == INTERNAL_ERROR
    assert error.error.message == "Prompt 'greet' failed to render"


@pytest.mark.asyncio
async def test_documents_come_first_and_model_turns_are_assistant_messages() -> None:
    ai, _ = app()
    ai.define_prompt(
        name='recall',
        docs=[Document.from_text('Ada likes tea.', metadata={'source': 'notes'})],
        messages=[Message(role=Role.MODEL, content=[Part.from_text('What would you like?')])],
        prompt='Something to drink.',
    )

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.get_prompt('recall')

    assert texts(result.messages) == [
        ('user', 'Context document 1:\nMetadata: {"source": "notes"}'),
        ('user', 'Ada likes tea.'),
        ('assistant', 'What would you like?'),
        ('user', 'Something to drink.'),
    ]


@pytest.mark.asyncio
async def test_inline_media_is_embedded_and_remote_media_is_linked() -> None:
    ai, _ = app()
    ai.define_prompt(
        name='look',
        prompt=[
            Part(media=Media(url=f'data:image/png;base64,{PIXEL}', content_type='image/png')),
            Part(media=Media(url='https://example.com/cat.png', content_type='image/png')),
        ],
    )

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.get_prompt('look')

    inline, remote = (message.content for message in result.messages)
    assert isinstance(inline, ImageContent)
    assert (inline.mimeType, inline.data) == ('image/png', PIXEL)
    assert isinstance(remote, ResourceLink)
    assert str(remote.uri) == 'https://example.com/cat.png'


class Address(BaseModel):
    city: str


class Delivery(BaseModel):
    """Input whose fields pydantic describes with a union and a reference."""

    count: int | None = None
    address: Address


@pytest.mark.asyncio
async def test_optional_and_nested_fields_are_read_as_their_types() -> None:
    """Pydantic writes `int | None` as anyOf and a nested model as a `$ref`."""
    ai, _ = app()
    ai.define_prompt(name='deliver', input_schema=Delivery, prompt='{{count}} to {{address.city}}')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.get_prompt('deliver', {'count': '3', 'address': '{"city": "Paris"}'})

    assert texts(result.messages) == [('user', '3 to Paris')]


@pytest.mark.asyncio
async def test_an_optional_field_of_a_file_prompt_is_read_as_its_type(tmp_path: Path) -> None:
    """Picoschema writes an optional field as a type list that includes null."""
    (tmp_path / 'order.prompt').write_text(
        '---\ninput:\n  schema:\n    count?: integer\n    vip?: boolean\n---\n{{#if vip}}VIP {{/if}}{{count}}\n'
    )
    ai, _ = app(prompt_dir=tmp_path)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.get_prompt('order', {'count': '3', 'vip': 'false'})

    assert texts(result.messages) == [('user', '3')]


@pytest.mark.asyncio
async def test_a_failure_after_rendering_keeps_its_text_off_the_wire() -> None:
    """Mapping runs after the prompt renders, and its errors are internal ones too."""
    ai, _ = app()
    ai.define_prompt(
        name='recall',
        docs=[Document.from_text('Ada likes tea.', metadata={'when': date(2020, 1, 1)})],
        prompt='Something to drink.',
    )

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'recall')

    assert error.error.code == INTERNAL_ERROR
    assert error.error.message == "Prompt 'recall' failed to render"


@pytest.mark.asyncio
async def test_a_genkit_error_while_rendering_sends_only_its_status(monkeypatch: pytest.MonkeyPatch) -> None:
    """A provider's error text can name a project or a URL."""
    ai, _ = app()
    ai.define_prompt(name='greet', input_schema=Person, prompt='Hello {{name}}')
    [action] = await ai.list_prompts()

    async def unavailable(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise GenkitError(status='UNAVAILABLE', message='try later')

    monkeypatch.setattr(action, 'run', unavailable)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'greet', {'name': 'Ada'})

    assert error.error.code == INTERNAL_ERROR
    assert error.error.message == "Prompt 'greet' failed to render: UNAVAILABLE"


@pytest.mark.asyncio
async def test_a_public_error_while_rendering_sends_its_status_and_message(monkeypatch: pytest.MonkeyPatch) -> None:
    ai, _ = app()
    ai.define_prompt(name='greet', input_schema=Person, prompt='Hello {{name}}')
    [action] = await ai.list_prompts()

    async def busy(*args: Any, **kwargs: Any) -> Any:  # noqa: ANN401
        raise PublicError(status='UNAVAILABLE', message='try later')

    monkeypatch.setattr(action, 'run', busy)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'greet', {'name': 'Ada'})

    assert error.error.code == INTERNAL_ERROR
    assert error.error.message == 'UNAVAILABLE: try later'


@pytest.mark.asyncio
async def test_input_a_file_prompt_rejects_names_only_the_prompt(tmp_path: Path) -> None:
    """A JSON schema's check quotes the value it rejects, so none of its text is sent."""
    (tmp_path / 'greet.prompt').write_text('---\ninput:\n  schema:\n    name: string\n---\nHello {{name}}\n')
    ai, _ = app(prompt_dir=tmp_path)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'greet', {'inputJson': '{}'})

    assert error.error.code == INVALID_PARAMS
    assert error.error.message == "Invalid input for prompt 'greet'"


class Stock(BaseModel):
    """Input with a constraint Pydantic checks after the argument is read."""

    count: int = Field(gt=0)


@pytest.mark.asyncio
async def test_rejected_input_names_the_field_without_the_value_sent() -> None:
    ai, _ = app()
    ai.define_prompt(name='restock', input_schema=Stock, prompt='{{count}}')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        error = await get_error(session, 'restock', {'count': '-41'})

    assert error.error.code == INVALID_PARAMS
    assert error.error.message == "Invalid input for prompt 'restock': count: Input should be greater than 0"


def test_a_field_that_may_be_several_types_stays_text() -> None:
    schema = {'type': 'object', 'properties': {'n': {'anyOf': [{'type': 'string'}, {'type': 'integer'}]}}}
    assert prompt_input(schema, {'n': '3'}) == {'n': '3'}


def test_a_scalar_union_takes_the_input_argument() -> None:
    assert prompt_input({'anyOf': [{'type': 'integer'}, {'type': 'null'}]}, {'input': '7'}) == 7


def test_inline_media_with_parameters_is_embedded() -> None:
    request = ModelRequest(
        messages=[
            Message(role=Role.USER, content=[Part(media=Media(url=f'data:image/png;name=dot.png;base64,{PIXEL}'))])
        ]
    )
    [message] = request_to_messages(request, 'look')
    assert isinstance(message.content, ImageContent)
    assert message.content.data == PIXEL


def test_a_whole_number_may_be_spelled_as_json_writes_it() -> None:
    assert prompt_input({'properties': {'n': {'type': 'integer'}}}, {'n': '3.0'}) == {'n': 3}


@pytest.mark.parametrize(
    ('field', 'text'),
    [
        ({'type': 'integer'}, '3.5'),
        ({'type': 'number'}, 'inf'),
        ({'type': 'boolean'}, 'yes'),
        ({'type': 'object'}, '[1]'),
        ({'type': 'array'}, '{"a": 1}'),
        ({'type': 'array'}, '[1'),
    ],
)
def test_an_argument_that_is_not_its_fields_type_is_refused(field: dict[str, Any], text: str) -> None:
    with pytest.raises(PromptArgumentError, match="'n'"):
        prompt_input({'properties': {'n': field}}, {'n': text})


def test_a_scalar_prompt_takes_only_the_input_argument() -> None:
    assert prompt_input({'type': 'integer'}, {'input': '7'}) == 7
    with pytest.raises(PromptArgumentError):
        prompt_input({'type': 'integer'}, {'n': '7'})


def test_a_field_the_schema_does_not_name_is_read_by_what_unnamed_fields_must_be() -> None:
    schema = {'type': 'object', 'properties': {}, 'additionalProperties': {'type': 'integer'}}
    assert prompt_input(schema, {'inputJson': '{"a": "text"}', 'b': '2'}) == {'a': 'text', 'b': 2}
