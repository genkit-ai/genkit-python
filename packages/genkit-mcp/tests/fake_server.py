# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A small subprocess MCP server used by connection lifecycle tests."""

# ruff: noqa: E402

import asyncio
import json
import os
import sys
import threading
from typing import Any, cast

# mcp.server imports jsonschema, which loads rfc3987_syntax when it is installed
# (dotpromptz pulls it in). That module compiles a Lark grammar at import, which
# is most of this server's startup and stretches every handshake on a busy
# machine. None in sys.modules makes the import fail, and jsonschema skips the
# IRI format checks this server never runs.
sys.modules['rfc3987_syntax'] = None  # type: ignore[assignment]

import anyio
from genkit_mcp._compat import McpError, field
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    CONNECTION_CLOSED,
    CallToolRequest,
    CallToolResult,
    ErrorData,
    ListResourcesResult,
    ListToolsRequest,
    ListToolsResult,
    ServerResult,
    TextContent,
    Tool,
)

MCP_V2 = not hasattr(Server, 'list_tools')
"""mcp 2 replaced the lowlevel server's handler decorators with ``on_*`` constructor arguments."""


def mcp_error(code: int, message: str) -> Exception:
    """Build the SDK's JSON-RPC error. mcp 2 takes the code and message, mcp 1.x an ``ErrorData``."""
    data = ErrorData(code=code, message=message)
    from_error_data = getattr(McpError, 'from_error_data', None)
    return from_error_data(data) if from_error_data is not None else McpError(data)


def tool(name: str, description: str, **fields: object) -> Tool:
    """Build a tool from wire-shaped fields, which mcp 1.x and 2.x both parse."""
    return Tool.model_validate({'name': name, 'description': description, 'inputSchema': {'type': 'object'}, **fields})


def text_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type='text', text=text)])


TOOLS = [
    tool('echo', 'Echo input.'),
    tool('second', 'Another tool.'),
    tool('wait', 'Wait before replying.'),
    tool('list_requests', 'Count tools/list requests.'),
    tool('who_called', 'Report the client that connected.'),
    tool('stall', 'Never reply.'),
    tool('rendezvous', 'Reply once two calls are in flight together.'),
    tool('await_stall', 'Reply once a stall call is in flight.'),
    tool(
        'bad_output',
        'Violate its own output schema.',
        outputSchema={'type': 'object', 'properties': {'value': {'type': 'integer'}}, 'required': ['value']},
    ),
]

COLLIDING_TOOLS = [
    tool('look.up', 'Collides once rewritten.'),
    tool('look/up', 'Collides once rewritten.'),
]

PAGE_SIZE = int(os.environ.get('MCP_FAKE_PAGE_SIZE', '2'))

TOOL_NAME_PREFIX = os.environ.get('MCP_FAKE_TOOL_NAME_PREFIX', '')
if TOOL_NAME_PREFIX:
    TOOLS = [tool.model_copy(update={'name': TOOL_NAME_PREFIX + tool.name}) for tool in TOOLS]

list_requests = 0

rendezvous_arrivals = 0
rendezvous_met = asyncio.Event()
stall_started = asyncio.Event()


def write_marker(path: str, text: str) -> None:
    """Write a file the tests poll for in one step, so a reader never finds it created but still empty."""
    staging = f'{path}.partial'
    with open(staging, 'w', encoding='utf-8') as marker:
        marker.write(text)
    os.replace(staging, path)


def page(tools: list[Tool], next_cursor: str | None) -> ListToolsResult:
    return ListToolsResult.model_validate({'tools': tools, 'nextCursor': next_cursor})


async def list_tools(cursor: str | None, *, cache_refresh: bool = False) -> ListToolsResult:
    """Answer one tools/list page.

    Args:
        cursor: The cursor the client sent, if any.
        cache_refresh: Whether mcp 1.x's server is refreshing its own tool cache,
            which must see every tool or call_tool skips input validation.
    """
    if os.environ.get('MCP_FAKE_COLLIDING_TOOLS'):
        return ListToolsResult(tools=COLLIDING_TOOLS)

    if cache_refresh:
        return ListToolsResult(tools=TOOLS)

    global list_requests
    list_requests += 1

    if os.environ.get('MCP_FAKE_APP_ERROR') and list_requests == 1:
        # -32000 is the JSON-RPC implementation-defined server error range, which
        # a healthy server may use and the SDK also synthesises on transport death.
        code = int(os.environ.get('MCP_FAKE_APP_ERROR_CODE', CONNECTION_CLOSED))
        raise mcp_error(code, 'application-level failure')

    stall_list_marker = os.environ.get('MCP_FAKE_STALL_LIST')
    if stall_list_marker:
        write_marker(stall_list_marker, 'stalled')
        await asyncio.Event().wait()

    if os.environ.get('MCP_FAKE_STUCK_CURSOR'):
        return page(TOOLS[:1], 'stuck')

    if os.environ.get('MCP_FAKE_EMPTY_CURSOR'):
        return page(TOOLS, '')

    if os.environ.get('MCP_FAKE_ENDLESS_CURSOR'):
        return page(TOOLS[:1], str(list_requests))

    start = int(cursor) if cursor is not None else 0
    end = start + PAGE_SIZE
    return page(TOOLS[start:end], str(end) if end < len(TOOLS) else None)


async def call_tool(name: str, arguments: dict[str, object], client: object) -> CallToolResult:
    """Answer one tools/call.

    Args:
        name: The tool the client called.
        arguments: The tool input.
        client: The ``Implementation`` the client named at initialize.
    """
    call_marker = os.environ.get('MCP_FAKE_CALL_MARKER')
    if call_marker:
        write_marker(call_marker, name)
    if os.environ.get('MCP_FAKE_APP_ERROR_CALL'):
        # A JSON-RPC error on the wire, not an isError result.
        raise mcp_error(CONNECTION_CLOSED, 'application-level failure')
    if name == 'wait':
        # Must stay under the SDK's PROCESS_TERMINATION_TIMEOUT of 2.0 seconds.
        await asyncio.sleep(0.25)
    if name == 'stall':
        stall_started.set()
        await asyncio.Event().wait()
    if name == 'await_stall':
        await stall_started.wait()
        return text_result('stalled')
    if name == 'rendezvous':
        global rendezvous_arrivals
        rendezvous_arrivals += 1
        if rendezvous_arrivals == 2:
            rendezvous_met.set()
        await rendezvous_met.wait()
        return text_result(str(rendezvous_arrivals))
    if name == 'bad_output':
        # Neither SDK's server validates a ready-made CallToolResult against the output schema.
        return CallToolResult.model_validate({
            'content': [TextContent(type='text', text='not an integer')],
            'structuredContent': {'value': 'not an integer'},
        })
    if name == 'list_requests':
        return text_result(str(list_requests))
    if name == 'who_called':
        assert client is not None
        return text_result(f'{getattr(client, "name")} {getattr(client, "version")}')  # noqa: B009
    return text_result(str(arguments.get('message', name)))


def build_v2_server() -> Server[Any]:
    """Register the handlers through mcp 2's ``on_*`` constructor arguments."""
    if os.environ.get('MCP_FAKE_NO_TOOLS'):
        # Without tool handlers the server neither advertises tools nor answers tools/list.
        async def list_resources(ctx: Any, params: Any) -> ListResourcesResult:  # noqa: ANN401
            return ListResourcesResult(resources=[])

        return Server('fake-server', on_list_resources=list_resources)

    async def on_list_tools(ctx: Any, params: Any) -> ListToolsResult:  # noqa: ANN401
        return await list_tools(params.cursor if params is not None else None)

    async def on_call_tool(ctx: Any, params: Any) -> CallToolResult:  # noqa: ANN401
        client_params = ctx.session.client_params
        client = field(client_params, 'client_info') if client_params is not None else None
        return await call_tool(params.name, params.arguments or {}, client)

    return Server('fake-server', on_list_tools=on_list_tools, on_call_tool=on_call_tool)


def build_v1_server() -> Any:  # noqa: ANN401 - mcp 1.x's Server, whose decorators mcp 2 removed.
    """Register the handlers through mcp 1.x's decorators and request handler table."""
    app = cast(Any, Server('fake-server'))

    @app.list_tools()
    # The annotation must be exactly ListToolsRequest: mcp 1.x reads it to decide whether to pass the request.
    async def on_list_tools(req: ListToolsRequest) -> ListToolsResult:
        # mcp 1.x passes None when its server refreshes its own tool cache.
        if req is None:  # pyright: ignore[reportUnnecessaryComparison]
            return await list_tools(None, cache_refresh=True)
        return await list_tools(req.params.cursor if req.params is not None else None)

    @app.call_tool()
    async def on_call_tool(name: str, arguments: dict[str, object]) -> CallToolResult:
        client_params = app.request_context.session.client_params
        return await call_tool(name, arguments, client_params.clientInfo if client_params is not None else None)

    if os.environ.get('MCP_FAKE_APP_ERROR_CALL'):
        # The call_tool decorator turns handler exceptions into isError results, so
        # putting a JSON-RPC error code on the wire needs the raw handler.
        async def failing_call(req: CallToolRequest) -> ServerResult:
            raise mcp_error(CONNECTION_CLOSED, 'application-level failure')

        app.request_handlers[CallToolRequest] = failing_call

    if os.environ.get('MCP_FAKE_NO_TOOLS'):
        # Without these handlers the server neither advertises tools nor answers tools/list.
        del app.request_handlers[ListToolsRequest]
        del app.request_handlers[CallToolRequest]

        @app.list_resources()
        async def list_resources() -> list[object]:
            return []

    return app


app: Any = build_v2_server() if MCP_V2 else build_v1_server()


def initialize_reply(message: dict[str, Any]) -> dict[str, Any]:
    return {
        'jsonrpc': '2.0',
        'id': message['id'],
        'result': {
            'protocolVersion': message['params']['protocolVersion'],
            'capabilities': {'tools': {}},
            'serverInfo': {'name': 'fake-server', 'version': '0'},
        },
    }


def reply_with_empty_results() -> None:
    """Answer every request after initialize with an empty result.

    An empty result is valid JSON-RPC but lacks the fields tools/list and
    tools/call require. Written by hand because mcp 2's server validates what a
    handler returns before it reaches the wire.
    """
    for line in sys.stdin.buffer:
        message = json.loads(line)
        if 'id' not in message or 'method' not in message:
            continue
        reply = (
            initialize_reply(message)
            if message['method'] == 'initialize'
            else {
                'jsonrpc': '2.0',
                'id': message['id'],
                'result': {},
            }
        )
        os.write(1, (json.dumps(reply) + '\n').encode())


def close_stdin_after_handshake(marker: str) -> None:
    """Answer initialize, then close stdin while stdout stays open, so the client's next write breaks."""
    while True:
        message = json.loads(sys.stdin.buffer.readline())
        if message.get('method') == 'initialize':
            os.write(1, (json.dumps(initialize_reply(message)) + '\n').encode())
        elif message.get('method') == 'notifications/initialized':
            break
    os.close(0)
    write_marker(marker, 'closed')
    threading.Event().wait()


async def main() -> None:
    if os.environ.get('MCP_FAKE_BAD_UTF8'):
        # Bypasses the SDK's text stream deliberately: this is a byte a strict
        # decoder cannot read, not a message.
        os.write(1, b'\xff\xfe not utf-8\n')
    pid_file = os.environ.get('MCP_FAKE_PID_FILE')
    if pid_file:
        write_marker(pid_file, str(os.getpid()))
    if os.environ.get('MCP_FAKE_MALFORMED_REPLY'):
        reply_with_empty_results()
        return
    close_stdin_marker = os.environ.get('MCP_FAKE_CLOSE_STDIN')
    if close_stdin_marker:
        close_stdin_after_handshake(close_stdin_marker)
    if os.environ.get('MCP_FAKE_STALL'):
        # Never answers initialize, so the client stays in its connecting state.
        await asyncio.Event().wait()
    async with stdio_server() as (read_stream, write_stream):
        await app.run(read_stream, write_stream, app.create_initialization_options())
    eof_marker = os.environ.get('MCP_FAKE_EOF_MARKER')
    if eof_marker:
        # Stands in for state a server flushes once stdin closes.
        await anyio.sleep(0.3)
        write_marker(eof_marker, 'flushed')


if __name__ == '__main__':
    asyncio.run(main())
