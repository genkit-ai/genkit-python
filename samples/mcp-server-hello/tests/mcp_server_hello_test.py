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

"""The sample's server, launched as a child process and driven by the MCP SDK's own client."""

import asyncio
import os
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

SRC = Path(__file__).resolve().parents[1] / 'src'

# Every await on the child process is bounded by this, so a hang fails the test instead.
LIMIT = 60


@asynccontextmanager
async def bookshop() -> AsyncIterator[ClientSession]:
    """Launch the sample's server and open an initialized session on it."""
    server = StdioServerParameters(command=sys.executable, args=[str(SRC / 'main.py')])
    # The server logs to stderr, including a traceback for the stock lookup that fails on purpose.
    with open(os.devnull, 'w') as errlog:
        async with (
            stdio_client(server, errlog=errlog) as (read_stream, write_stream),
            ClientSession(read_stream, write_stream) as session,
        ):
            await asyncio.wait_for(session.initialize(), timeout=LIMIT)
            yield session


@pytest.mark.asyncio
async def test_lists_every_tool_with_an_object_input() -> None:
    """All three tools are listed, each taking the fields of its input model."""
    async with bookshop() as session:
        tools = {tool.name: tool for tool in (await asyncio.wait_for(session.list_tools(), timeout=LIMIT)).tools}

    assert sorted(tools) == ['check_stock', 'opening_hours', 'search_books']
    assert tools['opening_hours'].inputSchema['properties'].keys() == {'name'}
    assert tools['check_stock'].inputSchema['required'] == ['title']
    assert tools['check_stock'].outputSchema is not None
    assert tools['check_stock'].outputSchema['properties'].keys() == {'title', 'copies'}


@pytest.mark.asyncio
async def test_a_typed_output_arrives_as_structured_content() -> None:
    """A tool returning a model answers with that model's fields."""
    async with bookshop() as session:
        result = await asyncio.wait_for(session.call_tool('check_stock', {'title': 'Dune'}), timeout=LIMIT)

    assert not result.isError
    assert result.structuredContent == {'title': 'Dune', 'copies': 1}


@pytest.mark.asyncio
async def test_a_public_error_reaches_the_client_as_an_error_result() -> None:
    """The client reads the error's status and message, and the session stays usable."""
    async with bookshop() as session:
        failed = await asyncio.wait_for(session.call_tool('check_stock', {'title': 'Ulysses'}), timeout=LIMIT)
        after = await asyncio.wait_for(session.call_tool('opening_hours', {'name': 'Sunday'}), timeout=LIMIT)

    assert failed.isError
    assert [block.text for block in failed.content if isinstance(block, TextContent)] == [
        "NOT_FOUND: We do not carry 'Ulysses'"
    ]
    assert [block.text for block in after.content if isinstance(block, TextContent)] == ['closed']


@pytest.mark.asyncio
async def test_the_client_script_runs_against_the_server() -> None:
    """`src/client.py` connects, lists the tools and prints each call's answer."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(SRC / 'client.py'),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.DEVNULL,
    )
    stdout, _ = await asyncio.wait_for(process.communicate(), timeout=LIMIT)
    lines = stdout.decode().splitlines()

    assert process.returncode == 0
    assert lines[0] == 'Connected to bookshop: Ask about books, stock and opening hours.'
    assert "search_books({'topic': 'fantasy'}) [ok]: {'titles': ['The Hobbit', 'A Wizard of Earthsea']}" in lines
    assert "check_stock({'title': 'Ulysses'}) [error]: NOT_FOUND: We do not carry 'Ulysses'" in lines
