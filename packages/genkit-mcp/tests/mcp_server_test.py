# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Tests driving a Genkit-backed MCP server through an in-memory client."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import pytest
from genkit_mcp._server import McpServer, create_mcp_server
from mcp import ClientSession
from mcp.shared.exceptions import McpError
from mcp.shared.memory import create_connected_server_and_client_session
from mcp.types import CallToolResult, ImageContent, TextContent
from pydantic import BaseModel, Field, field_validator
from structlog.testing import capture_logs

from genkit import Genkit, MultipartToolResponse, Part, PublicError, Tool, ToolContext
from genkit.plugin_api import Action, ActionKind, Plugin, provider_error


class Shelf(BaseModel):
    """Output of a tool whose schema MCP can advertise."""

    title: str
    count: int


class Greeting(BaseModel):
    """Input of a tool whose schema MCP can carry."""

    name: str


class Label(BaseModel):
    """Output whose field travels under an alias."""

    shelf_title: str = Field(alias='shelfTitle')


class Order(BaseModel):
    """Input whose validator rejects values its JSON schema allows."""

    qty: int

    @field_validator('qty')
    @classmethod
    def positive(cls, value: int) -> int:
        if value <= 0:
            raise ValueError(f'{value} is not a positive quantity')
        return value


@asynccontextmanager
async def connected(server: McpServer) -> AsyncIterator[ClientSession]:
    """Drive a server over the SDK's in-memory transport."""
    async with create_connected_server_and_client_session(server._server, raise_exceptions=True) as session:
        yield session


def app() -> Genkit:
    return Genkit()


def text_of(result: CallToolResult) -> str:
    """Join the text blocks of a result, ignoring any other block kind."""
    return ''.join(block.text for block in result.content if isinstance(block, TextContent))


@pytest.mark.asyncio
async def test_a_tool_defined_after_a_listing_is_in_the_next_one_and_callable() -> None:
    """Listing reads the app per request, so late definitions are not missed."""
    ai = app()
    server = create_mcp_server(ai, 'bookshop')

    async with connected(server) as session:
        before = await session.list_tools()

        @ai.tool()
        async def late() -> str:
            """Defined after the first listing."""
            return 'here'

        after = await session.list_tools()
        result = await session.call_tool('late', {})

    assert before.tools == []
    assert [tool.name for tool in after.tools] == ['late']
    assert after.tools[0].description == 'Defined after the first listing.'
    assert text_of(result) == 'here'


@pytest.mark.asyncio
async def test_the_server_advertises_its_name_and_version() -> None:
    ai = app()
    server = create_mcp_server(ai, 'bookshop', version='2.1.0')

    async with connected(server) as session:
        result = await session.initialize()

    assert result.serverInfo.name == 'bookshop'
    assert result.serverInfo.version == '2.1.0'


@pytest.mark.asyncio
async def test_capabilities_offer_tools_and_nothing_else() -> None:
    """Prompts and resources are not served, so they are not advertised."""
    ai = app()
    server = create_mcp_server(ai, 'bookshop')

    async with connected(server) as session:
        result = await session.initialize()

    assert result.capabilities.tools is not None
    assert result.capabilities.prompts is None
    assert result.capabilities.resources is None


@pytest.mark.asyncio
async def test_a_string_tool_returns_text() -> None:
    ai = app()

    @ai.tool()
    async def greet(greeting: Greeting) -> str:
        """Greet."""
        return f'hello {greeting.name}'

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.call_tool('greet', {'name': 'ada'})

    assert result.content == [TextContent(type='text', text='hello ada')]
    assert result.isError is False


@pytest.mark.asyncio
async def test_a_tool_with_a_scalar_argument_is_neither_listed_nor_callable() -> None:
    """MCP arguments are an object, so such a tool cannot be described or invoked."""
    ai = app()

    @ai.tool()
    async def shout(word: str) -> str:
        """Shout."""
        return word.upper()

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        listed = await session.list_tools()
        result = await session.call_tool('shout', {'word': 'hi'})

    assert listed.tools == []
    assert result.isError is True
    assert 'is not registered' in text_of(result)


@pytest.mark.asyncio
async def test_an_object_tool_returns_structured_content_and_advertises_its_schema() -> None:
    ai = app()

    @ai.tool()
    async def shelf() -> Shelf:
        """Describe a shelf."""
        return Shelf(title='Fiction', count=2)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        listed = await session.list_tools()
        result = await session.call_tool('shelf', {})

    assert listed.tools[0].outputSchema == Shelf.model_json_schema()
    assert result.structuredContent == {'title': 'Fiction', 'count': 2}


@pytest.mark.asyncio
async def test_a_tool_returning_a_tool_response_advertises_no_output_schema() -> None:
    """Its output can be left out, and a client rejects a result missing what the schema promised."""
    ai = app()

    @ai.tool()
    async def shelf_photo() -> MultipartToolResponse[Shelf]:
        """Photograph a shelf."""
        return MultipartToolResponse(content=[Part.from_media(url='data:image/png;base64,aW1n')])

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        listed = await session.list_tools()
        result = await session.call_tool('shelf_photo', {})

    assert listed.tools[0].outputSchema is None
    assert result.isError is False
    assert result.content == [ImageContent(type='image', mimeType='image/png', data='aW1n')]


@pytest.mark.asyncio
async def test_a_tool_registered_as_a_plain_action_is_served_by_alias_without_an_output_schema() -> None:
    """A tool action with no ``tool.v2`` companion returns its value, which becomes the output."""
    ai = app()

    async def label(_: dict[str, object]) -> Label:
        return Label(shelfTitle='Poetry')

    async def photo(_: dict[str, object]) -> MultipartToolResponse[Label]:
        return MultipartToolResponse(
            output=Label(shelfTitle='Poetry'),
            content=[Part.from_media(url='data:image/png;base64,aW1n')],
        )

    ai._registry.register_action_from_instance(Action(kind=ActionKind.TOOL, name='label', fn=label))
    ai._registry.register_action_from_instance(Action(kind=ActionKind.TOOL, name='photo', fn=photo))

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        listed = await session.list_tools()
        labelled = await session.call_tool('label', {})
        photographed = await session.call_tool('photo', {})

    assert [tool.outputSchema for tool in listed.tools] == [None, None]
    assert labelled.structuredContent == {'shelfTitle': 'Poetry'}
    assert photographed.isError is False
    assert photographed.structuredContent == {'shelfTitle': 'Poetry'}
    assert photographed.content[1:] == [ImageContent(type='image', mimeType='image/png', data='aW1n')]


@pytest.mark.asyncio
async def test_arguments_the_tool_rejects_name_the_field_but_not_the_validator_text() -> None:
    """The MCP SDK checks the JSON schema only, so the model learns which argument to fix from Genkit."""
    ai = app()

    @ai.tool()
    async def order(order: Order) -> str:
        """Order copies."""
        return f'{order.qty} ordered'

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.call_tool('order', {'qty': -3})

    assert result.isError is True
    assert text_of(result).startswith("INVALID_ARGUMENT: Invalid input for tool 'order': qty:")
    assert 'positive quantity' not in text_of(result)


@pytest.mark.asyncio
async def test_a_result_that_cannot_be_encoded_is_an_error_result() -> None:
    """Text with a lone surrogate, as a non-UTF-8 file name decodes to, would otherwise end the session."""
    ai = app()

    @ai.tool()
    async def filename() -> str:
        """Name a file."""
        return 'notes-\udcff.txt'

    @ai.tool()
    async def missing() -> str:
        """Report a file that is not there."""
        raise PublicError(status='NOT_FOUND', message='no file notes-\udcff.txt')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        returned = await session.call_tool('filename', {})
        raised = await session.call_tool('missing', {})

    assert (returned.isError, text_of(returned)) == (True, "Tool 'filename' failed")
    assert (raised.isError, text_of(raised)) == (True, "Tool 'missing' failed")


@pytest.mark.asyncio
async def test_a_tool_registered_again_under_its_name_is_described_afresh() -> None:
    """A plugin may hand the same metadata to a replacement, so the description follows what it is built from."""
    ai = app()
    metadata: dict[str, object] = {}

    async def lookup(_: dict[str, object]) -> str:
        return 'found'

    def register(input_schema: dict[str, object]) -> None:
        action = Action(kind=ActionKind.TOOL, name='lookup', fn=lookup, metadata=metadata)
        action.input_schema = input_schema
        ai._registry.register_action_from_instance(action)

    register({'type': 'object', 'properties': {'isbn': {'type': 'string'}}})
    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        before = await session.list_tools()
        register({'type': 'object', 'properties': {'title': {'type': 'string'}}})
        after = await session.list_tools()

    assert list(before.tools[0].inputSchema['properties']) == ['isbn']
    assert list(after.tools[0].inputSchema['properties']) == ['title']


@pytest.mark.asyncio
async def test_a_media_tool_returns_image_content() -> None:
    ai = app()

    @ai.tool()
    async def cover() -> MultipartToolResponse:
        """Return a cover image."""
        return MultipartToolResponse(
            output=None,
            content=[Part.from_media(url='data:image/png;base64,aW1n', content_type='image/png')],
        )

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.call_tool('cover', {})

    assert result.content == [ImageContent(type='image', mimeType='image/png', data='aW1n')]


@pytest.mark.asyncio
async def test_a_raising_tool_becomes_an_error_result_without_its_message() -> None:
    ai = app()

    @ai.tool()
    async def explode() -> str:
        """Fail."""
        raise ValueError('shelf collapsed')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.call_tool('explode', {})

    assert result.isError is True
    assert result.content == [TextContent(type='text', text="Tool 'explode' failed")]


@pytest.mark.asyncio
async def test_a_public_error_sends_its_status_and_message() -> None:
    """A PublicError is the app's message for the caller."""
    ai = app()

    @ai.tool()
    async def book() -> str:
        """Fail on purpose."""
        raise PublicError(status='INVALID_ARGUMENT', message='date must be in the future')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.call_tool('book', {})

    assert result.isError is True
    assert result.content == [TextContent(type='text', text='INVALID_ARGUMENT: date must be in the future')]


@pytest.mark.asyncio
async def test_an_error_for_the_caller_logs_one_line_and_a_crash_logs_its_traceback() -> None:
    """A deliberate error is the tool answering, so only a real failure keeps its traceback."""
    ai = app()

    @ai.tool()
    async def find_book() -> str:
        """Find a book."""
        raise PublicError(status='NOT_FOUND', message='no such book')

    @ai.tool()
    async def explode() -> str:
        """Fail."""
        raise ValueError('shelf collapsed')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        with capture_logs() as logs:
            await session.call_tool('find_book', {})
            await session.call_tool('explode', {})

    failures = [
        (entry['tool'], entry['log_level'], entry.get('exc_info', False))
        for entry in logs
        if entry['event'] == 'an MCP tool call failed'
    ]
    assert failures == [('find_book', 'warning', False), ('explode', 'error', True)]


@pytest.mark.asyncio
async def test_a_provider_error_sends_only_its_status() -> None:
    """A provider's error text can name a project or a resource path, so it stays on the server."""
    ai = app()

    @ai.tool()
    async def lookup() -> str:
        """Call a model that is not reachable."""
        raise provider_error(RuntimeError('Permission denied on resource project acme-prod-1234'), http_status=403)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.call_tool('lookup', {})

    assert result.isError is True
    assert result.content == [TextContent(type='text', text="Tool 'lookup' failed: PERMISSION_DENIED")]


@pytest.mark.asyncio
async def test_an_interrupt_is_an_error_result_without_a_traceback() -> None:
    """MCP has no way to pause a call for the user, so an interrupt says so instead of looking like a crash."""
    ai = app()
    ai.define_interrupt('confirm_order', description='Ask the diner to confirm.')

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        with capture_logs() as logs:
            result = await session.call_tool('confirm_order', {})

    assert result.isError is True
    assert text_of(result) == "Tool 'confirm_order' paused for input from the user, which this server cannot ask for"
    assert [log['log_level'] for log in logs if log.get('tool') == 'confirm_order'] == ['warning']


class _FailingPlugin(Plugin):
    """Fails to start with text the client must not see."""

    name = 'failing'

    async def init(self) -> list[Action]:
        raise RuntimeError('GET https://api.example.com/v1?key=SECRET failed')

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        return None

    async def list_actions(self) -> list[Action]:
        return []


@pytest.mark.asyncio
async def test_a_plugin_that_fails_to_start_keeps_its_error_text_on_the_server() -> None:
    """Listing starts every plugin, and a failure there reaches the client only as a fixed sentence."""
    ai = Genkit(plugins=[_FailingPlugin()])

    @ai.tool()
    async def known() -> str:
        """Exist."""
        return 'here'

    async with create_connected_server_and_client_session(create_mcp_server(ai, 'bookshop')._server) as session:
        with pytest.raises(McpError) as listing:
            await session.list_tools()
        result = await session.call_tool('known', {})

    assert listing.value.error.message == 'The server could not list its tools'
    assert result.isError is True
    assert text_of(result) == 'The server could not list its tools'


@pytest.mark.asyncio
async def test_the_request_meta_reaches_the_tool_context() -> None:
    ai = app()
    seen: list[object] = []

    @ai.tool()
    async def trace(ctx: ToolContext) -> str:
        """Record the context."""
        seen.append(ctx.context)
        return 'ok'

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        await session.call_tool('trace', {}, meta={'requestId': 'r-1'})

    assert seen == [{'mcp': {'_meta': {'requestId': 'r-1'}}}]


@pytest.mark.asyncio
async def test_a_tool_mcp_cannot_describe_is_warned_about_once() -> None:
    """Listing and calling reuse each description, so the warning does not repeat per request."""
    ai = app()

    @ai.tool()
    async def shout(word: str) -> str:
        """Shout."""
        return word.upper()

    @ai.tool()
    async def ping() -> str:
        """Answer."""
        return 'pong'

    async def whisper(word: str) -> str:
        return word.lower()

    ai._registry.register_action_from_instance(Action(kind=ActionKind.TOOL, name='whisper', fn=whisper))

    with capture_logs() as logs:
        async with connected(create_mcp_server(ai, 'bookshop')) as session:
            await session.list_tools()
            await session.call_tool('ping', {})
            await session.list_tools()

    skipped = [entry['tool'] for entry in logs if entry['log_level'] == 'warning']
    assert sorted(skipped) == ['shout', 'whisper']


@pytest.mark.asyncio
async def test_an_unknown_tool_is_an_error_result() -> None:
    ai = app()

    @ai.tool()
    async def known() -> str:
        """Exist."""
        return 'here'

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        result = await session.call_tool('missing', {})

    assert result.isError is True
    assert 'is not registered' in text_of(result)


@pytest.mark.asyncio
async def test_tools_a_dynamic_action_provider_serves_are_not_served() -> None:
    """Only tools defined on the app are served, as in JS and Go."""
    ai = app()

    @ai.tool()
    async def known() -> str:
        """Exist."""
        return 'here'

    async def remote() -> str:
        """Served by a provider."""
        return 'there'

    async def provide() -> dict[str, list[Action]]:
        return {'tool': [Tool(remote, name='remote')]}

    ai.define_dynamic_action_provider('elsewhere', provide)

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        listed = await session.list_tools()
        result = await session.call_tool('remote', {})

    assert [tool.name for tool in listed.tools] == ['known']
    assert result.isError is True


@pytest.mark.asyncio
async def test_a_no_arg_tool_advertises_an_object_input_schema() -> None:
    ai = app()

    @ai.tool()
    async def ping() -> str:
        """Answer."""
        return 'pong'

    async with connected(create_mcp_server(ai, 'bookshop')) as session:
        listed = await session.list_tools()

    assert listed.tools[0].inputSchema == {'type': 'object', 'properties': {}}
