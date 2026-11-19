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
"""Table-driven tests for list_actions against a full /v1/models listing."""

import json
from pathlib import Path
from typing import Any, cast
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from genkit_openai._openai_plugin import OpenAI
from openai.types import Model

from genkit.plugin_api import Action, ActionKind

_FIXTURE = json.loads((Path(__file__).parent / 'testdata' / 'openai_listing_fixture.json').read_text())
_CHAT: list[str] = _FIXTURE['chat']
_HIDDEN: list[str] = _FIXTURE['hidden']
_OTHER: list[str] = _FIXTURE['other']
_NO_TOOLS = set(_FIXTURE['chat_without_tools'])

# Not in the catalog, so discovery gives them generic info with no tools flag.
_UNCATALOGED_CHAT_IDS = {'gpt-6-cyber', 'gpt-6-cyber-2026-10-06', 'ft:gpt-4o-mini-2024-07-18:acme::abc123'}


async def _listed() -> dict[str, Action]:
    plugin = OpenAI(api_key='test-key')
    client = MagicMock()
    client.base_url = httpx.URL('https://api.openai.com/v1')
    result = MagicMock()
    result.data = [
        Model(id=model_id, created=1790000000, object='model', owned_by='system')
        for model_id in _CHAT + _HIDDEN + _OTHER
    ]
    client.models.list = AsyncMock(return_value=result)
    plugin._runtime_client = lambda: client
    return {action.name.removeprefix('openai/'): action for action in await plugin.list_actions()}


_HIDDEN_ON_EVERY_HOST = [
    'gpt-4o-realtime-preview',
    'gpt-live-1',
    'omni-moderation-latest',
    'sora-2',
    'babbage-002',
    'davinci-002',
    'davinci',
    'davinci-instruct-beta',
    'curie-instruct-beta',
    'text-similarity-davinci-001',
    'text-search-ada-doc-001',
    'gpt-3.5-turbo-instruct',
]
_HIDDEN_ON_OPENAI = [
    'gpt-oss-120b',
    'gpt-5-codex',
    'o3-pro',
    'qwen-audio-turbo',
    'perplexity/sonar-deep-research',
    'dall-e-3',
    'gpt-image-1',
]
_LISTED_ON_EVERY_HOST = [
    'my-chat-model',
    'my-realtime-llama',
    'llama-guard-moderation',
    'text-embedding-3-small',
    'text-embedding-ada-002',
    'tts-1',
    'whisper-1',
]


@pytest.mark.asyncio
@pytest.mark.parametrize('source', ['constructor', 'environment', 'client_options'])
@pytest.mark.parametrize(
    ('base_url', 'on_openai'),
    [
        ('https://api.openai.com/v1', True),
        ('https://API.OPENAI.COM/v1', True),
        ('https://us.api.openai.com/v1', True),
        ('https://eu.api.openai.com/v1', True),
        ('https://compatible.example/v1', False),
        ('http://localhost:8000/v1', False),
        ('https://api.openai.com.proxy.example/v1', False),
        ('https://notapi.openai.com/v1', False),
    ],
)
async def test_discovery_filters_by_the_effective_host(
    monkeypatch: pytest.MonkeyPatch, source: str, base_url: str, on_openai: bool
) -> None:
    """Every host hides non-chat ids; only OpenAI's hosts also hide ids OpenAI serves on other APIs."""
    ids = _HIDDEN_ON_EVERY_HOST + _HIDDEN_ON_OPENAI + _LISTED_ON_EVERY_HOST

    def list_models(request: httpx.Request) -> httpx.Response:
        assert request.method == 'GET'
        assert request.url == httpx.URL(base_url + '/models')
        return httpx.Response(
            200,
            json={
                'object': 'list',
                'data': [{'id': name, 'created': 0, 'object': 'model', 'owned_by': 'host'} for name in ids],
            },
        )

    monkeypatch.setenv('OPENAI_BASE_URL', 'https://ignored.example/v1')
    async with httpx.AsyncClient(transport=httpx.MockTransport(list_models)) as http_client:
        options: dict[str, Any] = {'http_client': http_client}
        kwargs: dict[str, Any] = {}
        if source == 'constructor':
            kwargs['base_url'] = base_url
        elif source == 'environment':
            monkeypatch.setenv('OPENAI_BASE_URL', base_url)
        else:
            options['base_url'] = base_url
        plugin = OpenAI(api_key='test-key', client_options=options, **kwargs)
        actions = {action.name.removeprefix('openai/'): action for action in await plugin.list_actions()}

    expected = _LISTED_ON_EVERY_HOST if on_openai else _HIDDEN_ON_OPENAI + _LISTED_ON_EVERY_HOST
    assert set(actions) == set(expected)
    assert _is_chat(actions['my-chat-model'])
    assert actions['text-embedding-3-small'].kind == ActionKind.EMBEDDER
    assert _model_meta(actions['tts-1'])['customOptions']['title'] == 'OpenAITtsConfig'
    assert _model_meta(actions['whisper-1'])['customOptions']['title'] == 'OpenAISttConfig'
    if not on_openai:
        assert _is_chat(actions['gpt-oss-120b'])
        assert _is_chat(actions['perplexity/sonar-deep-research'])
        assert _model_meta(actions['dall-e-3'])['customOptions']['title'] == 'OpenAIDalleConfig'
        assert _model_meta(actions['gpt-image-1'])['customOptions']['title'] == 'OpenAIGptImageConfig'


def _model_meta(action: Action) -> dict[str, Any]:
    assert action.metadata is not None
    return cast(dict[str, Any], action.metadata['model'])


def _is_chat(action: Action) -> bool:
    if action.kind != ActionKind.MODEL:
        return False
    return 'frequencyPenalty' in _model_meta(action)['customOptions']['properties']


def test_fixture_classes_are_disjoint() -> None:
    """Each fixture id belongs to exactly one class."""
    ids = _CHAT + _HIDDEN + _OTHER
    assert len(ids) == len(set(ids))


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id', _CHAT)
async def test_chat_ids_are_listed_as_chat_models(model_id: str) -> None:
    """Every id Chat Completions serves with text output is listed as a chat model."""
    listed = await _listed()
    assert model_id in listed
    assert _is_chat(listed[model_id])


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id', _HIDDEN)
async def test_hidden_ids_are_not_listed(model_id: str) -> None:
    """Ids no Chat Completions, Images, Audio or Embeddings action in this plugin can serve are not listed."""
    assert model_id not in await _listed()


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id', _OTHER)
async def test_other_ids_are_not_listed_as_chat_models(model_id: str) -> None:
    """Embedding, speech, transcription and image ids are never listed as chat models."""
    listed = await _listed()
    if model_id in listed:
        assert not _is_chat(listed[model_id])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('model_id', 'kind'),
    [
        ('text-embedding-3-small', ActionKind.EMBEDDER),
        ('text-embedding-ada-002', ActionKind.EMBEDDER),
        ('tts-1-hd-1106', ActionKind.MODEL),
        ('gpt-4o-mini-transcribe-2025-12-15', ActionKind.MODEL),
        ('gpt-image-2.5-flare-2026-09-08', ActionKind.MODEL),
        ('chatgpt-image-latest', ActionKind.MODEL),
    ],
)
async def test_non_chat_ids_are_listed_under_their_own_kind(model_id: str, kind: ActionKind) -> None:
    """Embedders, speech, transcription and GPT Image ids are listed with their own config."""
    listed = await _listed()
    assert listed[model_id].kind == kind


@pytest.mark.asyncio
async def test_gpt_4o_transcribe_diarize_is_listed_as_a_transcription_model() -> None:
    """gpt-4o-transcribe-diarize is listed with the transcription config, not hidden or listed as chat."""
    action = (await _listed())['gpt-4o-transcribe-diarize']

    assert action.kind == ActionKind.MODEL
    assert _model_meta(action)['customOptions']['title'] == 'OpenAISttConfig'


@pytest.mark.asyncio
async def test_chatgpt_image_latest_takes_the_gpt_image_config() -> None:
    """chatgpt-image-latest is served by the Images API with the GPT Image options."""
    props = _model_meta((await _listed())['chatgpt-image-latest'])['customOptions']['properties']
    assert 'background' in props
    assert 'frequencyPenalty' not in props


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id', ['gpt-image-1-mini', 'gpt-image-1.5', 'chatgpt-image-latest'])
async def test_image_ids_with_announced_shutdown_are_listed_as_deprecated(model_id: str) -> None:
    """GPT Image ids with an announced shutdown carry stage 'deprecated' in discovery."""
    assert _model_meta((await _listed())[model_id]).get('stage') == 'deprecated'


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id', sorted(set(_CHAT) - _UNCATALOGED_CHAT_IDS))
async def test_listed_chat_models_declare_tool_support(model_id: str) -> None:
    """supports.tools is false only where Chat Completions rejects function tools."""
    supports = _model_meta((await _listed())[model_id])['supports']
    assert supports['tools'] is (model_id not in _NO_TOOLS)
