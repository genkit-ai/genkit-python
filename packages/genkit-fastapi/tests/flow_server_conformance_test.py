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

"""Runs the shared tests/flow_server_tests.yaml spec against a flow served by serve_flow.

The spec lives in genkit-ai/genkit and is not copied here. JS runs it through
tests/src/flow_server_test.ts against go/tests/test_app, reading the stream
with the ``streamFlow`` client. This harness serves the same ``streamy`` flow
and reads the response the way that client's ``__flowRunEnvelope`` does.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
import urllib.request
from typing import Any

import pytest
import yaml
from fastapi import FastAPI
from genkit_fastapi import serve_flow
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from genkit import FlowContext, Genkit

SPEC_UPSTREAM = 'https://raw.githubusercontent.com/genkit-ai/genkit/main/tests/'

# Same delimiter as __flowStreamDelimiter in js/genkit/src/client/client.ts.
FLOW_STREAM_DELIMITER = '\n\n'


def spec_path(relative: str, *, start: pathlib.Path | None = None) -> pathlib.Path:
    """Resolve ``tests/<relative>`` from genkit-ai/genkit.

    Lookup order: ``GENKIT_SPECS_DIR/<basename>`` (CI fetches the files flat),
    then ``tests/<relative>`` in any parent or in a sibling ``genkit`` checkout,
    then a download from the upstream main branch into a temp-dir cache.
    """
    name = pathlib.PurePosixPath(relative).name
    env_dir = os.environ.get('GENKIT_SPECS_DIR')
    if env_dir:
        candidate = pathlib.Path(env_dir) / name
        if candidate.is_file():
            return candidate

    for parent in (start or pathlib.Path(__file__)).resolve().parents:
        candidate = parent / 'tests' / relative
        if candidate.is_file():
            return candidate
        sibling = parent.parent / 'genkit' / 'tests' / relative
        if sibling.is_file():
            return sibling

    cache_dir = pathlib.Path(tempfile.gettempdir()) / 'genkit-specs'
    cache_dir.mkdir(parents=True, exist_ok=True)
    cached = cache_dir / name
    if not cached.is_file():
        tmp_file = cache_dir / f'{name}.{os.getpid()}.tmp'
        urllib.request.urlretrieve(SPEC_UPSTREAM + relative, tmp_file)  # noqa: S310 - fixed https URL
        os.replace(tmp_file, cached)
    return cached


class StreamyChunk(BaseModel):
    """Chunk shape of streamy in go/tests/test_app (``chunk{Count int `json:"count"`}``)."""

    count: int


def build_flow_server() -> FastAPI:
    """The Python twin of the flow server in go/tests/test_app."""
    ai = Genkit()

    @ai.flow()
    async def streamy(count: int, ctx: FlowContext[StreamyChunk]) -> str:
        i = 0
        while i < count:
            ctx.send_chunk(StreamyChunk(count=i))
            i += 1
        return f'done {count}, streamed: {i} times'

    app = FastAPI()
    # Default path is /<flow name>, the same route Go mounts with "POST /"+a.Name().
    app.include_router(serve_flow(streamy))
    return app


def parse_flow_stream(body: str) -> tuple[list[Any], Any]:
    """Split an SSE body into (chunks, result) exactly like __flowRunEnvelope.

    Each frame is ``data: <json>`` followed by a blank line. ``message`` frames
    are chunks, the ``result`` frame ends the stream, and an ``error`` frame
    fails it. A body that ends without a result is an error too.
    """
    chunks: list[Any] = []
    buffer = body
    while FLOW_STREAM_DELIMITER in buffer:
        frame, buffer = buffer.split(FLOW_STREAM_DELIMITER, 1)
        payload = json.loads(frame[len('data: ') :])
        if 'message' in payload:
            chunks.append(payload['message'])
        elif 'result' in payload:
            return chunks, payload['result']
        elif 'error' in payload:
            error = payload['error']
            raise AssertionError(f'{error.get("status")}: {error.get("message")}\n{error.get("details")}')
        else:
            raise AssertionError(f'unknown chunk format: {json.dumps(payload)}')
    raise AssertionError('stream did not terminate correctly')


def _load_spec() -> dict[str, Any]:
    with spec_path('flow_server_tests.yaml').open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


SPEC = _load_spec()


@pytest.mark.asyncio
@pytest.mark.parametrize('test', SPEC['tests'], ids=[f'{t["path"]} data={t["post"]["data"]}' for t in SPEC['tests']])
async def test_flow_server_spec(test: dict[str, Any]) -> None:
    """Each flow in flow_server_tests.yaml streams the spec's chunks, then the spec's result."""
    data = test['post']['data']
    want_message: str = test['response']['message']
    want_result: str = test['response']['result']

    async with AsyncClient(transport=ASGITransport(app=build_flow_server()), base_url='http://test') as client:
        response = await client.post(
            f'/{test["path"]}',
            json={'data': data},
            headers={'Accept': 'text/event-stream', 'Content-Type': 'application/json'},
        )

    assert response.status_code == 200, f'Server returned: {response.status_code}: {response.text}'
    chunks, result = parse_flow_stream(response.text)

    for count, chunk in enumerate(chunks):
        # The JS runner swaps only the first {count} in the message template.
        expected = want_message.replace('{count}', str(count), 1)
        assert chunk == json.loads(expected), f'chunk {count}: got {chunk!r}, want {expected}'
        # The spec pins a string, so also match JSON.stringify output byte for byte.
        assert json.dumps(chunk, separators=(',', ':'), ensure_ascii=False) == expected
    assert len(chunks) == data, f'unexpected number of stream chunks received: got {len(chunks)}, want: {data}'
    assert result == want_result.replace('{count}', str(len(chunks)))


def test_parse_flow_stream_follows_client_envelope() -> None:
    """Message frames become chunks, the result frame ends the read, error and truncated streams fail."""
    body = 'data: {"message":{"count":0}}\n\ndata: {"result":"done 1, streamed: 1 times"}\n\n'
    assert parse_flow_stream(body) == ([{'count': 0}], 'done 1, streamed: 1 times')

    with pytest.raises(AssertionError, match='INTERNAL: kitchen offline'):
        parse_flow_stream('data: {"error":{"status":"INTERNAL","message":"kitchen offline"}}\n\n')
    with pytest.raises(AssertionError, match='did not terminate'):
        parse_flow_stream('data: {"message":{"count":0}}\n\n')
    with pytest.raises(AssertionError, match='unknown chunk format'):
        parse_flow_stream('data: {"order":"ramen"}\n\n')


def test_spec_path_downloads_flow_server_spec_from_tests_root(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """flow_server_tests.yaml sits at tests/, not tests/specs/, upstream; the fallback URL keeps that path."""
    fetched: list[str] = []

    def fake_urlretrieve(src: str, dest: pathlib.Path) -> None:
        fetched.append(src)
        pathlib.Path(dest).write_text('tests: []\n', encoding='utf-8')

    monkeypatch.delenv('GENKIT_SPECS_DIR', raising=False)
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path / 'tmp'))
    monkeypatch.setattr(urllib.request, 'urlretrieve', fake_urlretrieve)

    path = spec_path('flow_server_tests.yaml', start=tmp_path / 'isolated' / 'repo' / 'conformance_test.py')

    assert fetched == ['https://raw.githubusercontent.com/genkit-ai/genkit/main/tests/flow_server_tests.yaml']
    assert path == tmp_path / 'tmp' / 'genkit-specs' / 'flow_server_tests.yaml'
