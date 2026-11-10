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

"""Runs the shared tests/specs/reflection_api.yaml spec against the Python reflection server.

The spec lives in genkit-ai/genkit and is not copied here. JS runs it through
tests/src/reflection_api_test.ts and Go through go/tests/api_test.go. This
harness mirrors both: the app under test defines the same ``customReflector``
echo model, and responses are compared the way Go's ``compare1`` does (every
key in the spec must match, extra keys in the response are ignored), skipping
the keys the JS runner excludes.
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
from httpx import ASGITransport, AsyncClient

from genkit import FinishReason, Genkit, Message, ModelResponse, Part
from genkit._core._reflection import create_reflection_asgi_app
from genkit.model import GenerationUsage, ModelContext, ModelRequest

SPEC_UPSTREAM = 'https://raw.githubusercontent.com/genkit-ai/genkit/main/tests/'

# Same list as diffJSON's excludeKeys in tests/src/reflection_api_test.ts.
# Schemas and descriptions differ by SDK on purpose; telemetry, latency and
# usage are runtime values the JS runner never pins.
EXCLUDED_KEYS = frozenset({
    'outputSchema',
    'inputSchema',
    'description',
    'telemetry',
    'latencyMs',
    'usage',
})


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
        try:
            urllib.request.urlretrieve(SPEC_UPSTREAM + relative, tmp_file)  # noqa: S310 - fixed https URL
            os.replace(tmp_file, cached)
        except BaseException:
            tmp_file.unlink(missing_ok=True)
            raise
    return cached


def go_compare(got: Any, want: Any, path: str = '', excluded: frozenset[str] = EXCLUDED_KEYS) -> list[str]:  # noqa: ANN401 - decoded JSON
    """Diff ``got`` against ``want`` with go/tests/api_test.go ``compare1`` semantics.

    Every key in ``want`` must be in ``got`` and match. Keys only in ``got`` are
    ignored. Lists must have the same length. Keys in ``excluded`` are skipped
    at any depth, as in the JS runner.
    """
    where = path or 'top level'
    if isinstance(want, dict):
        if not isinstance(got, dict):
            return [f'{where}: want object, got {type(got).__name__}']
        msgs: list[str] = []
        for key, want_value in want.items():
            if key in excluded:
                continue
            if key not in got:
                msgs.append(f'{where}: missing key {key!r}')
                continue
            msgs.extend(go_compare(got[key], want_value, f'{path}.{key}' if path else key, excluded))
        return msgs
    if isinstance(want, list):
        if not isinstance(got, list):
            return [f'{where}: want list, got {type(got).__name__}']
        if len(got) != len(want):
            return [f'{where}: got list length {len(got)}, want {len(want)}']
        msgs = []
        for i, (got_item, want_item) in enumerate(zip(got, want, strict=True)):
            msgs.extend(go_compare(got_item, want_item, f'{path}.{i}' if path else str(i), excluded))
        return msgs
    # bool is an int subclass in Python; JSON true must not equal 1.
    if isinstance(want, bool) or isinstance(got, bool):
        same = type(got) is type(want) and got == want
    else:
        same = got == want
    return [] if same else [f'{where}: got {got!r}, want {want!r}']


def go_marshal(request: ModelRequest) -> str:
    """Serialize a ModelRequest the way go/tests/test_app's echo does with json.Marshal.

    - Go struct fields marshal in declaration order, and the generated Go types
      declare them alphabetically, so keys are sorted.
    - Go uses compact separators and ``omitempty`` drops nil fields.
    - ``exclude_unset`` drops ``output``: Python's ModelRequest fills it with
      an empty ``OutputConfig`` default, while Go keeps a nil pointer that
      ``omitempty`` omits and JS leaves it undefined. Without this the echo
      would read ``{"messages":[...],"output":{}}``.
    """
    body = request.model_dump(mode='json', by_alias=True, exclude_none=True, exclude_unset=True)
    return json.dumps(body, sort_keys=True, separators=(',', ':'), ensure_ascii=False)


def build_test_app() -> Genkit:
    """The Python twin of tests/test_js_app and go/tests/test_app."""
    ai = Genkit()

    async def custom_reflector(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        message = Message(role='model', content=[Part.from_text(go_marshal(request))])
        # Same usage the JS app reports through getBasicUsageStats.
        return ModelResponse(
            finish_reason=FinishReason.STOP,
            message=message,
            usage=GenerationUsage.from_messages(request.messages, message),
        )

    ai.define_model('customReflector', custom_reflector)
    return ai


def _load_spec() -> dict[str, Any]:
    with spec_path('specs/reflection_api.yaml').open(encoding='utf-8') as stream:
        return yaml.safe_load(stream)


def _test_id(test: dict[str, Any]) -> str:
    post = test.get('post')
    if post is None:
        return f'GET {test["path"]}'
    key = post.get('key') if isinstance(post, dict) else None
    return f'POST {test["path"]} {key}' if key else f'POST {test["path"]}'


SPEC = _load_spec()


@pytest.mark.asyncio
@pytest.mark.parametrize('test', SPEC['tests'], ids=[_test_id(t) for t in SPEC['tests']])
async def test_reflection_api_spec(test: dict[str, Any]) -> None:
    """Each request in reflection_api.yaml gets 200 and a body matching the spec."""
    app = create_reflection_asgi_app(build_test_app()._registry)
    async with AsyncClient(transport=ASGITransport(app=app), base_url='http://test') as client:
        health = await client.get('/api/__health')
        assert health.status_code == 200
        if 'post' in test:
            response = await client.post(test['path'], json=test['post'])
        else:
            response = await client.get(test['path'])

    assert response.status_code == 200, f'{test["path"]}: got status {response.status_code}: {response.text}'
    got = response.json()
    msgs = go_compare(got, test['body'])
    assert not msgs, '\n'.join(msgs) + '\n\ngot:\n' + json.dumps(got, indent=2, sort_keys=True)


def test_go_compare_matches_go_semantics() -> None:
    """Missing and mismatched spec keys fail; extra response keys and excluded keys don't."""
    want = {'result': {'finishReason': 'stop', 'usage': {'inputCharacters': 5}, 'flags': [True]}}

    assert go_compare({'result': {'finishReason': 'stop', 'flags': [True], 'custom': {}}}, want) == []
    assert go_compare({'result': {'finishReason': 'length', 'flags': [True]}}, want) == [
        "result.finishReason: got 'length', want 'stop'"
    ]
    assert go_compare({'result': {'flags': [True]}}, want) == ["result: missing key 'finishReason'"]
    assert go_compare({'result': {'finishReason': 'stop', 'flags': [1]}}, want) == ['result.flags.0: got 1, want True']
    assert go_compare({'result': {'finishReason': 'stop', 'flags': []}}, want) == [
        'result.flags: got list length 0, want 1'
    ]


def test_go_marshal_matches_go_test_app_bytes() -> None:
    """The echo text is byte-for-byte what Go's json.Marshal produces for the spec input."""
    request = ModelRequest.model_validate({'messages': [{'role': 'user', 'content': [{'text': 'hello'}]}]})

    assert go_marshal(request) == '{"messages":[{"content":[{"text":"hello"}],"role":"user"}]}'


@pytest.mark.parametrize(
    ('relative', 'url'),
    [
        ('specs/reflection_api.yaml', SPEC_UPSTREAM + 'specs/reflection_api.yaml'),
        ('flow_server_tests.yaml', SPEC_UPSTREAM + 'flow_server_tests.yaml'),
    ],
    ids=['tests/specs', 'tests root'],
)
def test_spec_path_downloads_when_no_checkout(
    relative: str, url: str, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With no env dir and no genkit checkout nearby, the spec is fetched once from upstream main."""
    fetched: list[str] = []

    def fake_urlretrieve(src: str, dest: pathlib.Path) -> None:
        fetched.append(src)
        pathlib.Path(dest).write_text('tests: []\n', encoding='utf-8')

    monkeypatch.delenv('GENKIT_SPECS_DIR', raising=False)
    monkeypatch.setattr(tempfile, 'gettempdir', lambda: str(tmp_path / 'tmp'))
    monkeypatch.setattr(urllib.request, 'urlretrieve', fake_urlretrieve)
    start = tmp_path / 'isolated' / 'repo' / 'conformance_test.py'

    first = spec_path(relative, start=start)
    second = spec_path(relative, start=start)

    assert fetched == [url]
    assert first == second == tmp_path / 'tmp' / 'genkit-specs' / pathlib.PurePosixPath(relative).name
    assert first.read_text(encoding='utf-8') == 'tests: []\n'
    assert list(first.parent.glob('*.tmp')) == []


def test_spec_path_prefers_env_dir_then_sibling_checkout(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """GENKIT_SPECS_DIR wins; without it a sibling genkit checkout is used before any download."""
    monkeypatch.setattr(urllib.request, 'urlretrieve', lambda *_: pytest.fail('should not download'))
    start = tmp_path / 'genkit-python' / 'packages' / 'conformance_test.py'
    sibling = tmp_path / 'genkit' / 'tests' / 'flow_server_tests.yaml'
    sibling.parent.mkdir(parents=True)
    sibling.write_text('app: flow_server\n', encoding='utf-8')
    env_dir = tmp_path / 'ci-specs'
    env_dir.mkdir()
    (env_dir / 'flow_server_tests.yaml').write_text('app: ci\n', encoding='utf-8')

    monkeypatch.setenv('GENKIT_SPECS_DIR', str(env_dir))
    assert spec_path('flow_server_tests.yaml', start=start) == env_dir / 'flow_server_tests.yaml'

    monkeypatch.delenv('GENKIT_SPECS_DIR')
    assert spec_path('flow_server_tests.yaml', start=start) == sibling
