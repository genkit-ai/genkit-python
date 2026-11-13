# Copyright 2025 Google LLC
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

"""Tests for Filesystem middleware."""

import copy
import tempfile
from pathlib import Path

import pytest
from genkit_middleware import Filesystem

from genkit import Interrupt, MultipartToolResponse, Part
from genkit.middleware import MiddlewareContext, ToolHookParams
from genkit.model import ToolRequest

# ---------------------------------------------------------------------------
# Construction / validation
# ---------------------------------------------------------------------------


def test_filesystem_validates_root_dir() -> None:
    """Filesystem must reject an empty root_dir."""
    with pytest.raises(ValueError, match='root_dir'):
        Filesystem(root_dir='')


def test_filesystem_resolves_root() -> None:
    """root_dir is resolved to an absolute path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        assert fs._root_abs == str(Path(tmpdir).resolve())


# ---------------------------------------------------------------------------
# _resolve_safe
# ---------------------------------------------------------------------------


def test_resolve_safe_allows_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        assert fs._resolve_safe('') == fs._root_abs


def test_resolve_safe_allows_child() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        child = Path(tmpdir) / 'sub' / 'file.txt'
        child.parent.mkdir(parents=True)
        assert fs._resolve_safe('sub/file.txt').endswith('sub/file.txt')


def test_resolve_safe_blocks_escape() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        with pytest.raises(ValueError, match='escapes'):
            fs._resolve_safe('../../../etc/passwd')


# ---------------------------------------------------------------------------
# _list_files
# ---------------------------------------------------------------------------


def test_list_files_returns_paths_relative_to_queried_dir() -> None:
    """list_files paths should be relative to the requested sub-dir, not root."""
    with tempfile.TemporaryDirectory() as tmpdir:
        sub = Path(tmpdir) / 'docs'
        sub.mkdir()
        (sub / 'api.md').write_text('hello')
        fs = Filesystem(root_dir=tmpdir)
        entries = fs._list_files('docs')
        names = [e['path'] for e in entries]
        assert 'api.md' in names
        assert 'docs/api.md' not in names


def test_list_files_root() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        (Path(tmpdir) / 'a.txt').write_text('a')
        (Path(tmpdir) / 'b.txt').write_text('b')
        fs = Filesystem(root_dir=tmpdir)
        entries = fs._list_files()
        names = {e['path'] for e in entries}
        assert 'a.txt' in names
        assert 'b.txt' in names


# ---------------------------------------------------------------------------
# _read_file_impl (text files)
# ---------------------------------------------------------------------------


def test_read_file_queues_content() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'hello.txt'
        f.write_text('hello world\n')
        fs = Filesystem(root_dir=tmpdir)
        result = fs._read_file_impl('hello.txt', 0, 0)
        assert 'queued' in result.lower() or 'read' in result.lower()
        assert len(fs._message_queue) == 1
        assert len(fs._message_queue[0].content) == 1


def test_read_file_rereads_each_time() -> None:
    """No dedup cache — each read queues content again."""
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'hello.txt'
        f.write_text('hello world\n')
        fs = Filesystem(root_dir=tmpdir)
        fs._read_file_impl('hello.txt', 0, 0)
        fs._message_queue.clear()
        result = fs._read_file_impl('hello.txt', 0, 0)
        assert 'read' in result.lower()
        assert len(fs._message_queue) == 1


# ---------------------------------------------------------------------------
# _write_file_impl and _edit_file_impl
# ---------------------------------------------------------------------------


def test_write_file_overwrites_without_prior_read() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'existing.txt'
        f.write_text('original\n')
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        result = fs._write_file_impl('existing.txt', 'new content\n')
        assert 'written' in result.lower()
        assert f.read_text() == 'new content\n'


def test_write_new_file_succeeds() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        result = fs._write_file_impl('new.txt', 'content\n')
        assert 'written' in result.lower()
        assert (Path(tmpdir) / 'new.txt').read_text() == 'content\n'


def test_edit_file_reads_from_disk() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        f = Path(tmpdir) / 'edit_me.txt'
        f.write_text('hello world\n')
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        result = fs._edit_file_impl('edit_me.txt', [{'old_string': 'hello', 'new_string': 'hi'}])
        assert 'edited' in result.lower()
        assert f.read_text() == 'hi world\n'


# ---------------------------------------------------------------------------
# tools() — dynamic tool registration
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_tools_returns_read_and_list(ctx) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        tools = fs.tools(ctx)
        names = {t.name for t in tools}
        assert 'list_files' in names
        assert 'read_file' in names
        assert 'write_file' not in names


@pytest.mark.asyncio
async def test_tools_returns_write_when_allowed(ctx) -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        tools = fs.tools(ctx)
        names = {t.name for t in tools}
        assert 'write_file' in names
        assert 'edit_file' in names


@pytest.mark.asyncio
async def test_tools_have_nonempty_descriptions(ctx) -> None:
    """Model-facing tool descriptions must be set — nested defs have no docstring."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        for t in fs.tools(ctx):
            assert t.description, f'{t.name} has empty description'


@pytest.mark.asyncio
async def test_wrap_tool_lets_interrupt_through_without_queueing_an_error(ctx) -> None:
    """A filesystem tool that pauses still pauses; the error catch-all can't turn it into a message."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir, allow_write_access=True)
        write_file = next(t for t in fs.tools(ctx) if t.name == 'write_file')
        params = ToolHookParams(
            tool_request_part=Part(tool_request=ToolRequest(name='write_file', input={'path': 'menu.md'})),
            tool=write_file,
        )

        async def needs_approval(_params: ToolHookParams, _ctx: MiddlewareContext) -> MultipartToolResponse:
            raise Interrupt({'reason': 'confirm overwrite', 'path': 'menu.md'})

        with pytest.raises(Interrupt) as raised:
            await fs.wrap_tool(params, ctx, needs_approval)

        assert raised.value.metadata == {'reason': 'confirm overwrite', 'path': 'menu.md'}
        assert fs._message_queue == []  # noqa: SLF001


# ---------------------------------------------------------------------------
# Redaction boundaries, edit size caps, and special files (FIFOs)
# ---------------------------------------------------------------------------


def test_redact_does_not_mangle_sibling_directory_sharing_root_prefix(tmp_path: Path) -> None:
    """`_redact` replaces `root` only on path boundaries, never turning `/tmp/workspace_evil` into `._evil`."""
    root = tmp_path / 'workspace'
    root.mkdir()
    evil = tmp_path / 'workspace_evil' / 'sibling.txt'
    dashed = tmp_path / 'workspace-prod' / 'secret.txt'
    fs = Filesystem(root_dir=str(root))

    assert fs._redact(f"Absolute path '{evil}' escapes the root directory.") == (  # noqa: SLF001
        f"Absolute path '{evil}' escapes the root directory."
    )
    assert fs._redact(f"Absolute path '{dashed}' escapes the root directory.") == (  # noqa: SLF001
        f"Absolute path '{dashed}' escapes the root directory."
    )
    assert fs._redact(f"[Errno 21] Is a directory: '{fs._root_abs}'") == "[Errno 21] Is a directory: '.'"  # noqa: SLF001
    assert fs._redact(f"No such file: '{fs._root_abs}/orders/a.txt'") == "No such file: 'orders/a.txt'"  # noqa: SLF001


def test_edit_file_enforces_size_cap_before_read_and_before_write(tmp_path: Path) -> None:
    """`_edit_file_impl` refuses files > 10 MiB on disk and edits whose result exceeds 10 MiB."""
    big = tmp_path / 'huge.txt'
    with open(big, 'wb') as fh:
        fh.truncate(10 * 1024 * 1024 + 1)
    fs = Filesystem(root_dir=str(tmp_path), allow_write_access=True)

    with pytest.raises(ValueError, match='File too large'):
        fs._edit_file_impl('huge.txt', [{'old_string': 'a', 'new_string': 'b'}])

    seed = tmp_path / 'seed.txt'
    seed.write_text('x' * 1024, encoding='utf-8')
    chunk = 'y' * (11 * 1024)  # 1024 * 11 KiB > 10 MiB
    with pytest.raises(ValueError, match='Edited file too large'):
        fs._edit_file_impl('seed.txt', [{'old_string': 'x', 'new_string': chunk, 'replace_all': True}])
    assert seed.read_text(encoding='utf-8') == 'x' * 1024


@pytest.mark.skipif(not hasattr(os := __import__('os'), 'mkfifo'), reason='needs mkfifo')
def test_fifo_is_hidden_from_list_files_and_rejected_by_write_without_blocking(tmp_path: Path) -> None:
    """FIFOs are omitted from `list_files` (matching `read_file`) and refused by `write_file` before `open()`."""
    import os

    (tmp_path / 'menu.md').write_text('Soup\n', encoding='utf-8')
    os.mkfifo(tmp_path / 'kitchen.pipe')
    sub = tmp_path / 'orders'
    sub.mkdir()
    os.mkfifo(sub / 'nested.pipe')

    fs = Filesystem(root_dir=str(tmp_path), allow_write_access=True)

    assert [e['path'] for e in fs._list_files()] == ['menu.md', 'orders']
    assert [e['path'] for e in fs._list_files(recursive=True)] == ['menu.md', 'orders']

    with pytest.raises(ValueError, match='File not found'):
        fs._read_file_impl('kitchen.pipe', 0, 0)
    with pytest.raises(ValueError, match='Cannot write to non-regular file'):
        fs._write_file_impl('kitchen.pipe', 'blocked forever if opened')


def test_per_call_copies_do_not_share_the_message_queue() -> None:
    """Two generate() calls reusing one instance must not see each other's file reads."""
    with tempfile.TemporaryDirectory() as tmpdir:
        fs = Filesystem(root_dir=tmpdir)
        # The engine shallow-copies middleware once per generate() call.
        first_call = copy.copy(fs)
        second_call = copy.copy(fs)

        first_call._enqueue_parts([Part.from_text('contents of allergens.csv')])  # noqa: SLF001

        assert second_call._message_queue == []  # noqa: SLF001
        assert fs._message_queue == []  # noqa: SLF001
        assert second_call.config is fs.config
