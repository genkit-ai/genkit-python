# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""A Genkit app served over MCP stdio, run as a child process by the stdio tests.

``MCP_TEST_PRE_REDIRECT=1`` points ``sys.stdout`` at stderr before serving, as
an app following the usual MCP advice does. ``MCP_TEST_NO_REDIRECT=1`` serves
with ``redirect_stdout=False``.
"""

import asyncio
import os
import sys

from genkit_mcp import create_mcp_server

from genkit import Genkit, get_logger

logger = get_logger(__name__)

ai = Genkit()


@ai.tool()
async def plain_text() -> str:
    """Return a string."""
    logger.info('serving plain text')
    return 'plain text'


@ai.tool()
async def accented() -> str:
    """Return text outside ASCII."""
    return 'café 東京'


@ai.tool()
async def raw_writes() -> str:
    """Use stdio below sys.stdout: a child process, the descriptor, and the original stream."""
    script = 'import sys; print("written by a child process"); print("child read", repr(sys.stdin.read()))'
    child = await asyncio.create_subprocess_exec(sys.executable, '-c', script)
    await child.wait()
    os.write(1, b'written to descriptor 1\n')
    assert sys.__stdout__ is not None
    sys.__stdout__.write('written to the original stdout\n')
    sys.__stdout__.flush()
    return 'wrote'


@ai.tool()
async def prints() -> str:
    """Print the way app code does."""
    sys.stdout.write('printed by a tool\n')
    sys.stdout.flush()
    return 'printed'


server = create_mcp_server(ai, 'stdio-test', redirect_stdout=os.environ.get('MCP_TEST_NO_REDIRECT') != '1')

sys.stdout.write('written before start\n')

if __name__ == '__main__':
    if os.environ.get('MCP_TEST_PRE_REDIRECT') == '1':
        sys.stdout = sys.stderr
    asyncio.run(server.start())
    sys.stdout.write('printed after the server stopped\n')
    sys.stdout.flush()
