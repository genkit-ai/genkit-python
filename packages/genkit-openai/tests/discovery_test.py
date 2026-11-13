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

import pytest
from genkit_openai._openai_plugin import OpenAI
from openai.types import Model

from genkit.plugin_api import Action, ActionKind

_FIXTURE = json.loads((Path(__file__).parent / 'testdata' / 'openai_listing_fixture.json').read_text())
_CHAT: list[str] = _FIXTURE['chat']
_HIDDEN: list[str] = _FIXTURE['hidden']
_OTHER: list[str] = _FIXTURE['other']
_NO_TOOLS = set(_FIXTURE['tools']['never_on_chat_completions'])

# Not in the catalog, so discovery gives it generic info with no tools flag.
_UNCATALOGED_CHAT_IDS = {'gpt-6-cyber'}


async def _listed() -> dict[str, Action]:
    plugin = OpenAI(api_key='test-key')
    client = MagicMock()
    result = MagicMock()
    result.data = [
        Model(id=model_id, created=1790000000, object='model', owned_by='system')
        for model_id in _CHAT + _HIDDEN + _OTHER
    ]
    client.models.list = AsyncMock(return_value=result)
    plugin._runtime_client = lambda: client
    return {action.name.removeprefix('openai/'): action for action in await plugin.list_actions()}


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
    """Realtime, live, audio, moderation, Responses-only and legacy base ids are not listed."""
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
async def test_chatgpt_image_latest_takes_the_gpt_image_config() -> None:
    """chatgpt-image-latest is served by the Images API with the GPT Image options."""
    props = _model_meta((await _listed())['chatgpt-image-latest'])['customOptions']['properties']
    assert 'background' in props
    assert 'frequencyPenalty' not in props


@pytest.mark.asyncio
@pytest.mark.parametrize('model_id', sorted(set(_CHAT) - _UNCATALOGED_CHAT_IDS))
async def test_listed_chat_models_declare_tool_support(model_id: str) -> None:
    """supports.tools is false only where Chat Completions rejects function tools."""
    supports = _model_meta((await _listed())[model_id])['supports']
    assert supports['tools'] is (model_id not in _NO_TOOLS)
