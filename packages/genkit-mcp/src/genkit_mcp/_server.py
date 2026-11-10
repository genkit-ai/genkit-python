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

import os
import sys
from collections.abc import AsyncIterator, Sequence
from contextlib import asynccontextmanager
from typing import Any, TextIO

import anyio
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server
from mcp.server.streamable_http import EventStore
from mcp.server.streamable_http_manager import StreamableHTTPSessionManager
from mcp.server.transport_security import TransportSecuritySettings
from mcp.shared.exceptions import McpError
from mcp.types import INTERNAL_ERROR, CallToolResult, ErrorData, Tool as McpTool
from pydantic import TypeAdapter
from starlette.applications import Starlette
from starlette.routing import Route
from starlette.types import Receive, Scope, Send

from genkit import Genkit, GenkitError, Interrupt, MultipartToolResponse, PublicError, Tool, get_logger
from genkit.web import error_body, error_status
from genkit_mcp._export import error_result, multipart_to_call_result, tool_to_mcp

logger = get_logger(__name__)

_JSON = TypeAdapter(object)

DEFAULT_VERSION = '1.0.0'
"""The version a server advertises when the app names none."""

_LISTING_FAILED = 'The server could not list its tools'

LOCAL_HOSTS = ('localhost', 'localhost:*', '127.0.0.1', '127.0.0.1:*', '[::1]', '[::1]:*')
"""``Host`` values the HTTP endpoint accepts unless told otherwise."""

LOCAL_ORIGINS = tuple(f'{scheme}://{host}' for scheme in ('http', 'https') for host in LOCAL_HOSTS)
"""``Origin`` values the HTTP endpoint accepts unless told otherwise."""


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
                stderr and its stdin to the null device. Leave it on unless the
                app already keeps both clear.
        """
        self._ai = ai
        self._name = name
        self._version = version
        self._redirect_stdout = redirect_stdout
        self._described: dict[str, tuple[tuple[object, ...], McpTool | None]] = {}
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

        stdin and stdout are the wire. The transport reads and writes copies of
        descriptors 0 and 1, taken whatever ``sys.stdin`` and ``sys.stdout``
        are at the time. With ``redirect_stdout`` on, descriptor 1 and
        ``sys.stdout`` then point at stderr and descriptor 0 at the null device
        for the rest of the process's life, so ``print``, loggers and child
        processes can neither write between frames nor read them.
        """
        stdin, stdout = _wire_files()
        with stdin, stdout:
            if self._redirect_stdout:
                _move_stdio_off_the_wire()
            async with stdio_server(stdin=anyio.wrap_file(stdin), stdout=anyio.wrap_file(stdout)) as streams:
                await self.serve(*streams)

    def _session_manager(
        self,
        *,
        stateless: bool = False,
        json_response: bool = False,
        event_store: EventStore | None = None,
        security: TransportSecuritySettings,
    ) -> StreamableHTTPSessionManager:
        """Build a new streamable HTTP session manager, since one can run only once."""
        return StreamableHTTPSessionManager(
            app=self._server,
            event_store=event_store,
            json_response=json_response,
            stateless=stateless,
            security_settings=security,
        )

    def asgi_app(
        self,
        path: str = '/mcp',
        *,
        stateless: bool = False,
        json_response: bool = False,
        event_store: EventStore | None = None,
        allowed_hosts: Sequence[str] = LOCAL_HOSTS,
        allowed_origins: Sequence[str] = LOCAL_ORIGINS,
    ) -> Starlette:
        """Build an ASGI app serving this server over streamable HTTP.

        The app owns a manager of its own and opens it in its lifespan, so
        whatever runs the app has to run that lifespan, and can run it only
        once; call ``asgi_app`` again for an app to start afresh. Starlette does
        not run a mounted app's lifespan, so mounting this one means chaining
        its ``router.lifespan_context`` into the host app's.

        The endpoint has no authentication: anyone who can reach it can call
        every tool. A request whose ``Host`` is not in ``allowed_hosts`` gets a
        421, and one whose ``Origin`` is not in ``allowed_origins`` gets a 403,
        so a web page cannot reach a local server through DNS rebinding.

        stdout is untouched here, unlike ``start``: HTTP carries no wire on it.

        Args:
            path: Route the endpoint answers on, within this app.
            stateless: Whether each request gets a fresh transport.
            json_response: Whether a response is one JSON body rather than an
                SSE stream.
            event_store: Where to record events for a reconnecting client.
                Stateful only.
            allowed_hosts: ``Host`` header values to accept. ``name:*``
                accepts any port. Defaults to localhost on any port.
            allowed_origins: ``Origin`` header values to accept from a
                browser. ``scheme://name:*`` accepts any port. A request
                with no ``Origin`` is accepted. Defaults to localhost on any
                port, over http or https.

        Returns:
            A Starlette app to serve, mount, or wrap in middleware.

        Raises:
            TypeError: If ``allowed_hosts`` or ``allowed_origins`` is a single
                string rather than a sequence of them.
            ValueError: If ``event_store`` is given with ``stateless=True``,
                where there is no session to replay.
        """
        for argument, value in (('allowed_hosts', allowed_hosts), ('allowed_origins', allowed_origins)):
            if isinstance(value, str):
                raise TypeError(f'{argument} takes a sequence of strings, not one string: [{value!r}]')
        if stateless and event_store is not None:
            raise ValueError('event_store needs a stateful app; a stateless one would never use it')
        manager = self._session_manager(
            stateless=stateless,
            json_response=json_response,
            event_store=event_store,
            security=TransportSecuritySettings(
                enable_dns_rebinding_protection=True,
                allowed_hosts=list(allowed_hosts),
                allowed_origins=list(allowed_origins),
            ),
        )

        @asynccontextmanager
        async def lifespan(app: Starlette) -> AsyncIterator[None]:
            async with manager.run():
                yield

        return Starlette(routes=[Route(path, endpoint=_StreamableHttpEndpoint(manager))], lifespan=lifespan)

    async def _list_tools(self) -> list[McpTool]:
        """Describe every tool this server offers, read fresh from the app."""
        return [described for _, described in await self._served_tools()]

    async def _served_tools(self) -> list[tuple[Tool, McpTool]]:
        """Pair each tool this server serves with how it is described.

        Listing and calling read the same pairs, so a tool MCP cannot describe
        is not reachable by name either.
        """
        try:
            tools = await self._ai.list_tools()
        except Exception as error:
            logger.exception("could not list the app's tools for MCP")
            raise McpError(ErrorData(code=INTERNAL_ERROR, message=_LISTING_FAILED)) from error
        served: list[tuple[Tool, McpTool]] = []
        for tool in tools:
            # A plain tool action comes back as a new view on every listing, so
            # the description is keyed by name and by everything it is built from.
            source = _description_source(tool)
            cached = self._described.get(tool.name)
            if cached is None or cached[0] != source:
                cached = (source, tool_to_mcp(tool))
                self._described[tool.name] = cached
            described = cached[1]
            if described is not None:
                served.append((tool, described))
        return served

    async def _call_tool(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Run one tool and map what it returned, reporting every failure as a result.

        A result that cannot be encoded, such as text with a lone surrogate,
        fails here rather than in the transport, where it would end the session.
        """
        result = await self._answer(name, arguments)
        try:
            result.model_dump_json(by_alias=True, exclude_none=True)
        except ValueError:
            logger.exception('an MCP tool result could not be encoded', tool=name)
            return error_result(f'Tool {name!r} failed')
        return result

    async def _answer(self, name: str, arguments: dict[str, Any]) -> CallToolResult:
        """Run one tool and map what it returned or raised.

        A raised error becomes an ``isError`` result and is logged here, where
        its traceback is still useful.
        """
        tool = await self._find_tool(name)
        if tool is None:
            return error_result(f'Tool {name!r} is not registered')

        context = self._request_context()
        try:
            return multipart_to_call_result(await _run(tool, arguments, context), name)
        except Interrupt:
            logger.warning('an MCP tool call paused for input, which MCP cannot ask for', tool=name)
            return error_result(f'Tool {name!r} paused for input from the user, which this server cannot ask for')
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
        redirect_stdout: Whether ``start`` moves this process's stdout to stderr
            and its stdin to the null device.

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


def _description_source(tool: Tool) -> tuple[object, ...]:
    """Everything ``tool_to_mcp`` reads, so an unchanged tool reuses its description."""
    return (
        tool.description,
        tool.input_schema,
        tool.output_schema,
        tool.model_output_schema,
        tool.wire_action is None,
        tool.metadata.get('mcp'),
    )


async def _run(tool: Tool, arguments: dict[str, Any], context: dict[str, Any] | None) -> MultipartToolResponse:
    """Run a tool and return what it produced as a tool response envelope.

    A tool built as a ``Tool`` has a ``tool.v2`` action that returns the
    envelope already. A plain tool action returns its function's value, which
    becomes the envelope's output, dumped by alias as JSON.
    """
    wire = tool.wire_action
    if wire is not None:
        return (await wire.run(arguments, context=context)).response
    output = (await tool.run(arguments, context=context)).response
    if isinstance(output, MultipartToolResponse):
        return MultipartToolResponse(
            output=_as_json(output.output),
            content=output.content,
            metadata=_as_json(output.metadata),
        )
    return MultipartToolResponse(output=_as_json(output))


def _as_json(value: object) -> Any:  # noqa: ANN401 - whatever JSON the value dumps to
    """Dump a tool's value to JSON data, by alias as Genkit's own tools do."""
    return _JSON.dump_python(value, mode='json', by_alias=True)


def _client_message(tool_name: str, error: Exception) -> str:
    """Say why a tool failed without passing on text the app never chose to show.

    The rule is a served flow's: a ``PublicError`` sends its status and
    message, and arguments the tool's own schema rejects send the failing
    fields without their values. Any other ``GenkitError`` sends only its
    status, since a provider's error text can name a project, a resource path
    or a URL.
    """
    if isinstance(error, PublicError) or error_status(error) < 500:
        body = error_body(error)
        return f'{body["status"]}: {body["message"]}'
    if isinstance(error, GenkitError):
        return f'Tool {tool_name!r} failed: {error.status}'
    return f'Tool {tool_name!r} failed'


def _wire_files() -> tuple[TextIO, TextIO]:
    """Open copies of descriptors 0 and 1 for the transport, as UTF-8 whatever the locale.

    Text still buffered for stdout is flushed first, so it cannot land between frames.
    """
    for stream in (sys.stdout, sys.__stdout__):
        if stream is not None:
            stream.flush()
    return open(os.dup(0), encoding='utf-8'), open(os.dup(1), 'w', encoding='utf-8')


def _move_stdio_off_the_wire() -> None:
    """Point stdout at stderr and stdin at the null device, so nothing else can use the wire.

    Moving the descriptors covers child processes, C code and any stream still
    bound to the originals. Rebinding ``sys.stdout`` covers ``print`` and
    structlog's default logger, which reads ``sys.stdout`` on every call.
    """
    null = os.open(os.devnull, os.O_RDONLY)
    try:
        os.dup2(2, 1)
        os.dup2(null, 0)
    finally:
        os.close(null)
    sys.stdout = sys.stderr


class _StreamableHttpEndpoint:
    """A streamable HTTP session manager as a plain ASGI app.

    Starlette hands a callable instance every method as raw ASGI, so the
    manager writes its own response to GET, POST and DELETE alike. On a
    ``Route`` the exact path is served with no trailing-slash redirect.
    """

    def __init__(self, manager: StreamableHTTPSessionManager) -> None:
        """Wrap one session manager.

        Args:
            manager: The manager to serve, whose ``run()`` context has to be
                open before this app handles anything.
        """
        self.manager = manager

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Hand one ASGI request to the session manager.

        Args:
            scope: The ASGI connection scope.
            receive: Where the request body arrives.
            send: Where the response goes.
        """
        await self.manager.handle_request(scope, receive, send)
