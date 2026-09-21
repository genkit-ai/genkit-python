# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""A Genkit app served to MCP clients as an MCP server."""

import sys
from io import TextIOWrapper
from typing import Any
from weakref import WeakKeyDictionary

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, Tool as McpTool
from pydantic import TypeAdapter

from genkit import Genkit, GenkitError, MultipartToolResponse, Tool, get_logger
from genkit_mcp._export import error_result, multipart_to_call_result, tool_to_mcp

logger = get_logger(__name__)

_JSON = TypeAdapter(object)

DEFAULT_VERSION = '1.0.0'
"""What JS and Go both advertise when the app names no version."""


class McpServer:
    """Serves a Genkit app's tools to MCP clients.

    Tools are read from the app on every ``tools/list``, so a tool defined
    after the server was built is still served. Tools a dynamic action
    provider serves are not.

    Attributes:
        name: Name the server advertises to clients.
        version: Version the server advertises to clients.
    """

    def __init__(
        self,
        ai: Genkit,
        name: str,
        *,
        version: str = DEFAULT_VERSION,
        instructions: str | None = None,
        redirect_stdout: bool = True,
    ) -> None:
        """Build a server over a Genkit app.

        Args:
            ai: The app whose tools are served.
            name: Name the server advertises.
            version: Version the server advertises.
            instructions: How a client should use this server.
            redirect_stdout: Whether ``start`` moves this process's stdout to
                stderr. Leave it on unless the app already keeps stdout clear.
        """
        self._ai = ai
        self._name = name
        self._version = version
        self._redirect_stdout = redirect_stdout
        self._described: WeakKeyDictionary[Tool, McpTool | None] = WeakKeyDictionary()
        self._server: Server[Any, Any] = Server(name, version=version, instructions=instructions)
        self._server.list_tools()(self._list_tools)
        self._server.call_tool()(self._call_tool)

    @property
    def name(self) -> str:
        """Name the server advertises to clients."""
        return self._name

    @property
    def version(self) -> str:
        """Version the server advertises to clients."""
        return self._version

    async def serve(self, read_stream: Any, write_stream: Any, *, raise_exceptions: bool = False) -> None:  # noqa: ANN401
        """Serve on a stream pair until the client disconnects.

        Args:
            read_stream: Where requests arrive, as the MCP SDK shapes it.
            write_stream: Where responses go, as the MCP SDK shapes it.
            raise_exceptions: Whether a handler error propagates instead of
                becoming an error response. For tests.
        """
        await self._server.run(
            read_stream,
            write_stream,
            self._server.create_initialization_options(),
            raise_exceptions=raise_exceptions,
        )

    async def start(self) -> None:
        """Serve on this process's stdin and stdout until stdin closes.

        stdout is the wire, so it is captured before anything can write to it
        and the process's own stdout is then pointed at stderr for the rest of
        its life. A logger built before this call keeps the descriptor it
        captured.
        """
        streams = stdio_server(
            stdin=anyio.wrap_file(TextIOWrapper(sys.stdin.buffer, encoding='utf-8')),
            stdout=anyio.wrap_file(TextIOWrapper(sys.stdout.buffer, encoding='utf-8')),
        )
        if self._redirect_stdout:
            _redirect_stdout_to_stderr()
        async with streams as (read_stream, write_stream):
            await self.serve(read_stream, write_stream)

    async def _list_tools(self) -> list[McpTool]:
        """Describe every tool this server offers, read fresh from the app."""
        return [described for _, described in await self._served_tools()]

    async def _served_tools(self) -> list[tuple[Tool, McpTool]]:
        """Pair each tool this server serves with how it is described.

        Listing and calling read the same pairs, so a tool MCP cannot describe
        is not reachable by name either.
        """
        served: list[tuple[Tool, McpTool]] = []
        for tool in await self._ai.list_tools():
            if tool not in self._described:
                self._described[tool] = tool_to_mcp(tool)
            described = self._described[tool]
            if described is not None:
                served.append((tool, described))
        return served

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Run one tool and map what it returned, reporting every failure as a result.

        A raised error becomes an ``isError`` result and is logged here, where
        its traceback is still useful.
        """
        tool = await self._find_tool(name)
        if tool is None:
            return error_result(f'Tool {name!r} is not registered')

        context = self._request_context()
        try:
            return multipart_to_call_result(await _run(tool, arguments, context), name)
        except Exception as error:
            logger.exception('an MCP tool call failed', tool=name)
            return error_result(_client_message(name, error))

    def _request_context(self) -> dict[str, Any] | None:
        """Pass the request's MCP ``_meta`` in the shape the client's handler reads."""
        try:
            meta = self._server.request_context.meta
        except LookupError:
            return None
        if meta is None:
            return None
        return {'mcp': {'_meta': meta.model_dump(by_alias=True, exclude_none=True)}}

    async def _find_tool(self, name: str) -> Tool | None:
        """Find a tool by the name it was listed under."""
        for tool, described in await self._served_tools():
            if described.name == name:
                return tool
        return None


def create_mcp_server(
    ai: Genkit,
    name: str,
    *,
    version: str = DEFAULT_VERSION,
    instructions: str | None = None,
    redirect_stdout: bool = True,
) -> McpServer:
    """Build a server over a Genkit app, registering nothing.

    Args:
        ai: The app whose tools are served.
        name: Name the server advertises.
        version: Version the server advertises.
        instructions: How a client should use this server.
        redirect_stdout: Whether ``start`` moves this process's stdout to stderr.

    Returns:
        A server ready for ``start`` or ``serve``.
    """
    return McpServer(
        ai,
        name,
        version=version,
        instructions=instructions,
        redirect_stdout=redirect_stdout,
    )


async def _run(tool: Tool, arguments: dict[str, Any], context: dict[str, Any] | None) -> MultipartToolResponse:
    """Run a tool and return what it produced as a tool response envelope.

    A tool built as a ``Tool`` has a ``tool.v2`` action that returns the
    envelope already. A plain tool action returns its function's value, which
    becomes the envelope's output.
    """
    wire = tool.wire_action
    if wire is not None:
        return (await wire.run(arguments, context=context)).response
    output = (await tool.run(arguments, context=context)).response
    if isinstance(output, MultipartToolResponse):
        return output
    return MultipartToolResponse(output=_JSON.dump_python(output, mode='json'))


def _client_message(tool_name: str, error: Exception) -> str:
    """Say why a tool failed without passing on text the app never chose to show.

    A ``GenkitError`` keeps its status and message but not its cause's text,
    which can carry a request URL or a credential.
    """
    if isinstance(error, GenkitError):
        return f'{error.status}: {error.message}'
    return f'Tool {tool_name!r} failed'


def _redirect_stdout_to_stderr() -> None:
    """Point this process's stdout at stderr, so nothing can corrupt the wire.

    Rebinding ``sys.stdout`` is enough to move a logger that was built before
    this call: structlog's default ``PrintLogger`` passes ``file=None`` to
    ``print``, which reads ``sys.stdout`` at call time. A logger bound to an
    explicit stream is the app's own choice and is left where it points.
    """
    sys.stdout = sys.stderr
