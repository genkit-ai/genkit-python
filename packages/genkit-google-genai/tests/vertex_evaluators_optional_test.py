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

"""The plugin loads with or without genkit.evaluator.

genkit-google-genai allows genkit<2, and a later genkit release removes
genkit.evaluator. Each test re-imports ``_google`` against a patched
``sys.modules`` and restores the original modules on teardown.
"""

import importlib
import sys
import traceback
from types import ModuleType
from unittest.mock import AsyncMock, MagicMock

import genkit_google_genai
import genkit_google_genai._evaluators
import pytest

from genkit.plugin_api import ActionKind

_GOOGLE = 'genkit_google_genai._google'
_EVALUATION = 'genkit_google_genai._evaluators._evaluation'


def _forget_google(monkeypatch: pytest.MonkeyPatch) -> None:
    """Drop ``_google`` and ``_evaluation`` so the next import re-runs them; teardown restores both."""
    # setattr to the current value records it, so teardown restores the package attributes
    # that the fresh import rebinds and later `patch('genkit_google_genai._google...')` calls see.
    monkeypatch.setattr(genkit_google_genai, '_google', sys.modules[_GOOGLE])
    monkeypatch.setattr(genkit_google_genai._evaluators, '_evaluation', sys.modules[_EVALUATION])
    monkeypatch.delitem(sys.modules, _GOOGLE)
    monkeypatch.delitem(sys.modules, _EVALUATION)


def _reimport_google(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    """Import a fresh ``_google`` against the current ``sys.modules``."""
    _forget_google(monkeypatch)
    return importlib.import_module(_GOOGLE)


def _offline(monkeypatch: pytest.MonkeyPatch, google: ModuleType) -> None:
    """Stub the genai client and model listing so init and list_actions make no calls."""
    monkeypatch.setattr(google.genai.client, 'Client', MagicMock())
    monkeypatch.setattr(google, '_list_genai_models', AsyncMock(return_value=google.GenaiModels()))


@pytest.mark.asyncio
async def test_vertexai_registers_evaluators_when_genkit_evaluator_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    """With genkit.evaluator importable, VertexAI registers and resolves its evaluators as before."""
    google = _reimport_google(monkeypatch)
    _offline(monkeypatch, google)
    assert google._HAS_VERTEX_EVALUATORS

    plugin = google.VertexAI(project='p', location='us-central1')
    actions = await plugin.init()
    resolved = await plugin.resolve(ActionKind.EVALUATOR, 'fluency')

    evaluators = {a.name for a in actions if a.kind == ActionKind.EVALUATOR}
    assert 'vertexai/fluency' in evaluators
    assert resolved is not None
    assert resolved.name == 'vertexai/fluency'


@pytest.mark.asyncio
async def test_plugins_load_without_genkit_evaluator(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without genkit.evaluator, VertexAI and GoogleAI still init, list and resolve models, with no evaluators."""
    # None in sys.modules makes `import genkit.evaluator` raise ModuleNotFoundError(name='genkit.evaluator'),
    # the same error a genkit release without the module raises.
    monkeypatch.setitem(sys.modules, 'genkit.evaluator', None)
    google = _reimport_google(monkeypatch)
    _offline(monkeypatch, google)
    assert not google._HAS_VERTEX_EVALUATORS

    vertex = google.VertexAI(project='p', location='us-central1')
    init_actions = await vertex.init()
    listed = await vertex.list_actions()
    model = await vertex.resolve(ActionKind.MODEL, 'gemini-2.5-flash')

    assert not [a for a in init_actions if a.kind == ActionKind.EVALUATOR]
    assert not [a for a in listed if a.kind == ActionKind.EVALUATOR]
    assert await vertex.resolve(ActionKind.EVALUATOR, 'fluency') is None
    assert model is not None
    assert model.name == 'vertexai/gemini-2.5-flash'

    googleai = google.GoogleAI(api_key='k')
    assert await googleai.init()
    googleai_model = await googleai.resolve(ActionKind.MODEL, 'gemini-2.5-flash')
    assert googleai_model is not None
    assert googleai_model.name == 'googleai/gemini-2.5-flash'


def test_unrelated_import_error_in_evaluation_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """A missing module other than genkit.evaluator is a real bug, so importing _google fails loudly."""
    # _evaluation imports _provider_errors before _google does, so the error is raised inside the guarded import.
    monkeypatch.setitem(sys.modules, 'genkit_google_genai._provider_errors', None)
    _forget_google(monkeypatch)

    with pytest.raises(ModuleNotFoundError) as exc_info:
        importlib.import_module(_GOOGLE)
    assert exc_info.value.name == 'genkit_google_genai._provider_errors'
    frames = traceback.extract_tb(exc_info.value.__traceback__)
    assert any(f.filename.endswith('_evaluation.py') for f in frames)


def test_missing_name_in_genkit_evaluator_propagates(monkeypatch: pytest.MonkeyPatch) -> None:
    """genkit.evaluator present but missing a name is API drift, not removal, so it is not swallowed."""
    monkeypatch.setitem(sys.modules, 'genkit.evaluator', ModuleType('genkit.evaluator'))
    _forget_google(monkeypatch)

    with pytest.raises(ImportError, match="cannot import name 'BaseDataPoint'"):
        importlib.import_module(_GOOGLE)
