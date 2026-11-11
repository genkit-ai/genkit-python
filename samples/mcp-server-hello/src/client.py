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

"""Launch the bookshop server and use it the way any MCP client would."""

import asyncio
import sys
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import CallToolResult, TextContent

SERVER = StdioServerParameters(command=sys.executable, args=[str(Path(__file__).with_name('main.py'))])


def describe(result: CallToolResult) -> str:
    """Show a result's structured content when it has some, else its text."""
    if result.structuredContent is not None:
        return str(result.structuredContent)
    return ' '.join(block.text for block in result.content if isinstance(block, TextContent))


async def call(session: ClientSession, tool: str, arguments: dict[str, Any]) -> None:
    result = await session.call_tool(tool, arguments)
    status = 'error' if result.isError else 'ok'
    print(f'{tool}({arguments}) [{status}]: {describe(result)}')


async def main() -> None:
    async with stdio_client(SERVER) as (read_stream, write_stream), ClientSession(read_stream, write_stream) as session:
        info = await session.initialize()
        print(f'Connected to {info.serverInfo.name}: {info.instructions}')

        print('Tools:')
        for tool in (await session.list_tools()).tools:
            print(f'  {tool.name}: {tool.description}')

        # The arguments are the fields of the tool's input model, not the Python parameter name.
        await call(session, 'opening_hours', {'name': 'Sunday'})
        await call(session, 'search_books', {'topic': 'fantasy'})
        await call(session, 'check_stock', {'title': 'Dune'})
        await call(session, 'check_stock', {'title': 'Ulysses'})


if __name__ == '__main__':
    asyncio.run(main())
