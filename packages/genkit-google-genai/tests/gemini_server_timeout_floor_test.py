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

"""Timeouts under 10s: the Gemini API rejects X-Server-Timeout below 10 with a 400."""

import json
from unittest.mock import MagicMock

import httpx
import pytest
from genkit_google_genai import AntigravityConfig, DeepResearchConfig, GeminiConfig, GoogleAI
from genkit_google_genai._google import _plugin_http_options
from genkit_google_genai._models._gemini import GeminiModel
from google import genai

from genkit import Genkit, Message, Part, Role
from genkit.model import ModelRequest


def _model(vertexai: bool) -> GeminiModel:
    client = MagicMock(spec=genai.Client)
    client.vertexai = vertexai
    return GeminiModel('gemini-flash-latest', client)


def _request(timeout: float) -> ModelRequest:
    return ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])],
        config=GeminiConfig(timeout=timeout),
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('vertexai', 'timeout', 'header', 'ms'),
    [
        (False, 5, '10', 5000),
        (False, 0.25, '10', 250),
        (False, 30, '30', 30000),
        (True, 5, '5', 5000),
    ],
    ids=['googleai-5s', 'googleai-250ms', 'googleai-30s', 'vertex-5s'],
)
async def test_request_timeout_floors_server_header_on_gemini_api(
    vertexai: bool, timeout: float, header: str, ms: int
) -> None:
    """GeminiConfig(timeout=5) keeps a 5s local deadline but sends X-Server-Timeout: 10 to the Gemini API."""
    cfg = await _model(vertexai)._genkit_to_googleai_cfg(_request(timeout))

    assert cfg is not None and cfg.http_options is not None
    assert cfg.http_options.timeout == ms
    assert cfg.http_options.headers is not None
    assert cfg.http_options.headers['X-Server-Timeout'] == header


@pytest.mark.parametrize(('timeout', 'header'), [(3, '10'), (10, None), (60, None)])
def test_plugin_timeout_floors_server_header(timeout: float, header: str | None) -> None:
    """GoogleAI(timeout=3) pins X-Server-Timeout to 10; google-genai would otherwise derive 3."""
    opts = _plugin_http_options(timeout=timeout, min_server_timeout=10)

    assert opts.timeout == timeout * 1000
    assert (opts.headers or {}).get('X-Server-Timeout') == header


def test_plugin_timeout_without_floor_sets_no_header() -> None:
    """VertexAI passes no floor; google-genai derives the header from timeout as before."""
    opts = _plugin_http_options(timeout=3)

    assert 'X-Server-Timeout' not in (opts.headers or {})


@pytest.mark.asyncio
async def test_short_plugin_timeout_floor_does_not_leak_into_interactions_models() -> None:
    """GoogleAI(timeout=5) + DeepResearchConfig(timeout=300) omits synthetic X-Server-Timeout: 10."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if '/interactions' in request.url.path:
            return httpx.Response(
                200,
                json={
                    'id': 'int-1',
                    'status': 'completed',
                    'outputs': [{'type': 'text', 'text': 'Done.'}],
                },
            )
        return httpx.Response(200, json={'models': []})

    plugin = GoogleAI(
        api_key='test-key',
        timeout=5,
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    ai = Genkit(plugins=[plugin])

    await ai.generate_operation(
        model='googleai/deep-research-pro-preview-12-2025',
        prompt='Compare espresso grinders.',
        config=DeepResearchConfig(timeout=300),
    )
    await ai.generate(
        model='googleai/antigravity-preview-05-2026',
        prompt='Build a menu card.',
        config=AntigravityConfig(timeout=60),
    )

    posts = [r for r in captured if r.method == 'POST' and '/interactions' in r.url.path]
    assert len(posts) == 2
    for req in posts:
        assert 'x-server-timeout' not in req.headers
    assert [req.extensions['timeout']['read'] for req in posts] == [300.0, 60.0]


@pytest.mark.asyncio
async def test_lowercase_caller_server_timeout_header_does_not_duplicate() -> None:
    """headers={'x-server-timeout': '20'} normalizes to X-Server-Timeout so google-genai sends one header."""
    captured: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        captured.append(request)
        if request.method == 'GET':
            return httpx.Response(200, json={'models': []})
        body = {'candidates': [{'content': {'role': 'model', 'parts': [{'text': 'ok'}]}, 'finishReason': 'STOP'}]}
        return httpx.Response(
            200, content=json.dumps(body).encode('utf-8'), headers={'content-type': 'application/json'}
        )

    plugin = GoogleAI(
        api_key='test-key',
        timeout=60,
        headers={'x-server-timeout': '20'},
        http_client_factory=lambda: httpx.AsyncClient(transport=httpx.MockTransport(handler)),
    )
    ai = Genkit(plugins=[plugin])

    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.')
    await ai.generate(model='googleai/gemini-2.5-flash', prompt='Suggest a dish.', config=GeminiConfig(timeout=15))

    posts = [r for r in captured if r.method == 'POST']
    assert len(posts) == 2
    first_raw = [v for k, v in posts[0].headers.raw if k.lower() == b'x-server-timeout']
    second_raw = [v for k, v in posts[1].headers.raw if k.lower() == b'x-server-timeout']
    assert first_raw == [b'20']
    assert second_raw == [b'15']
