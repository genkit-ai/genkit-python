# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# SPDX-License-Identifier: Apache-2.0

"""Tests for serving over this process's stdin and stdout, with no MCP client in the way."""

import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

import pytest

SERVER = Path(__file__).with_name('genkit_server.py')

# Every await on the child process is bounded by this, so a lifecycle regression
# fails the suite rather than hanging it. There is no global pytest timeout.
LIMIT = 60


def requests(tools: list[str]) -> list[dict[str, Any]]:
    """Open a session and call each tool, as a client would write them to stdin."""
    return [
        {
            'jsonrpc': '2.0',
            'id': 0,
            'method': 'initialize',
            'params': {
                'protocolVersion': '2025-06-18',
                'capabilities': {},
                'clientInfo': {'name': 'stdio-test', 'version': '1.0.0'},
            },
        },
        {'jsonrpc': '2.0', 'method': 'notifications/initialized'},
        *(
            {'jsonrpc': '2.0', 'id': call_id, 'method': 'tools/call', 'params': {'name': tool, 'arguments': {}}}
            for call_id, tool in enumerate(tools, start=1)
        ),
    ]


async def call_over_stdio(tools: list[str], **env: str) -> tuple[list[str], str]:
    """Run the fixture app as a child, call each tool, and return its stdout lines and stderr."""
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        str(SERVER),
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        env={**os.environ, 'GENKIT_LOG': 'debug', **env},
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    try:
        for request in requests(tools):
            process.stdin.write((json.dumps(request) + '\n').encode())
        await asyncio.wait_for(process.stdin.drain(), timeout=LIMIT)
        lines = await asyncio.wait_for(_read_until_answered(process.stdout, len(tools)), timeout=LIMIT)
        process.stdin.close()
        rest, stderr = await asyncio.wait_for(process.communicate(), timeout=LIMIT)
    except BaseException as error:
        process.kill()
        _, stderr = await process.communicate()
        if isinstance(error, (AssertionError, asyncio.TimeoutError)):
            raise AssertionError(f'{error!r}\nserver stderr:\n{stderr.decode()}') from error
        raise
    return lines + rest.decode().splitlines(), stderr.decode()


@pytest.mark.asyncio
async def test_stdout_carries_only_json_rpc_once_the_server_is_serving() -> None:
    """Logging, a child process, descriptor 1 and the original stdout all land on stderr while serving.

    A child process cannot read the client's requests either: its stdin is the null device.

    What the app writes before ``start()`` runs is another matter: stdout is not
    the wire yet, so it is flushed there ahead of the first frame.
    """
    lines, stderr = await call_over_stdio(['plain_text', 'raw_writes'])

    first = _first_json_line(lines)
    assert 'written before start' in lines[:first]
    assert [line for line in lines[first:] if _decode(line) is None] == []
    assert _results(lines) == {1: 'plain text', 2: 'wrote'}
    for text in (
        'serving plain text',
        'written by a child process',
        "child read ''",
        'written to descriptor 1',
        'written to the original stdout',
    ):
        assert text in stderr


@pytest.mark.asyncio
async def test_text_outside_ascii_survives_a_server_whose_locale_cannot_encode_it() -> None:
    lines, _ = await call_over_stdio(
        ['accented'], LC_ALL='C', LANG='C', PYTHONUTF8='0', PYTHONCOERCECLOCALE='0', PYTHONIOENCODING='ascii'
    )

    assert _results(lines) == {1: 'café 東京'}


@pytest.mark.asyncio
async def test_an_app_that_already_moved_sys_stdout_is_still_answered_on_stdout() -> None:
    """The wire is descriptor 1, not whatever ``sys.stdout`` happens to be when ``start()`` runs."""
    lines, _ = await call_over_stdio(['plain_text'], MCP_TEST_PRE_REDIRECT='1')

    assert _results(lines) == {1: 'plain text'}


@pytest.mark.asyncio
async def test_redirect_stdout_false_leaves_this_process_stdout_alone() -> None:
    """Opting out keeps stdout where it was, and still usable once the server stops."""
    lines, stderr = await call_over_stdio(['prints'], MCP_TEST_NO_REDIRECT='1')

    assert _results(lines) == {1: 'printed'}
    assert 'printed by a tool' in lines
    assert lines[-1] == 'printed after the server stopped'
    assert 'printed by a tool' not in stderr


async def _read_until_answered(stdout: asyncio.StreamReader, calls: int) -> list[str]:
    """Collect stdout lines up to and including the answer to the last tool call."""
    lines: list[str] = []
    while True:
        raw = await stdout.readline()
        if not raw:
            raise AssertionError(f'the server closed stdout without answering: {lines}')
        line = raw.decode().rstrip('\n')
        lines.append(line)
        message = _decode(line)
        if message is not None and message.get('id') == calls:
            return lines


def _results(lines: list[str]) -> dict[int, str]:
    """Read the text each tool call answered with, by call id."""
    return {
        message['id']: message['result']['content'][0]['text']
        for line in lines
        if (message := _decode(line)) and message.get('id', 0) > 0
    }


def _decode(line: str) -> dict[str, Any] | None:
    """Read one stdout line as a JSON-RPC message, or nothing when it is not one."""
    try:
        message = json.loads(line)
    except json.JSONDecodeError:
        return None
    return message if isinstance(message, dict) and message.get('jsonrpc') == '2.0' else None


def _first_json_line(lines: list[str]) -> int:
    """Find where the wire opens, which is the first line the transport wrote."""
    for index, line in enumerate(lines):
        if _decode(line) is not None:
            return index
    raise AssertionError(f'the server wrote no JSON-RPC at all: {lines}')
