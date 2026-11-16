# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Config keys on the generate and render spans use the wire spelling.

The Dev UI's "Open in model runner" pre-fills its form from the generate
span's ``config`` by exact camelCase key, the same keys the model span and
the JS SDK use. A ``max_output_tokens`` key there shows up as "default".
"""

import json
from typing import Any

import pytest
from pydantic import ConfigDict

from genkit import Genkit, Message, ModelResponse, Part, Role
from genkit._core._telemetry._attrs import Attr
from genkit._core._telemetry._http import ActiveSpan
from genkit.model import ModelConfig, ModelContext, ModelRequest

MODEL = 'kitchen'


class KitchenConfig(ModelConfig):
    """Common settings only; GenkitConfig aliases them to camelCase."""


class PassThroughConfig(ModelConfig):
    """Lets provider-only keys through, like a plugin that forwards extras."""

    model_config = ConfigDict(extra='allow')


def _app(config_class: type[ModelConfig] = KitchenConfig) -> tuple[Genkit, list[Any]]:
    ai = Genkit()
    seen: list[Any] = []

    if config_class is KitchenConfig:

        async def kitchen(request: ModelRequest[KitchenConfig], ctx: ModelContext) -> ModelResponse:
            seen.append(request.config)
            return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('Smoked Salmon Tartine')]))

        ai.define_model(name=MODEL, fn=kitchen)
    else:

        async def passthrough(request: ModelRequest[PassThroughConfig], ctx: ModelContext) -> ModelResponse:
            seen.append(request.config)
            return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('Smoked Salmon Tartine')]))

        ai.define_model(name=MODEL, fn=passthrough)
    return ai, seen


def _span_json(spans: list[ActiveSpan], name: str, attr: str) -> dict[str, Any]:
    span = next(s for s in spans if s.name == name)
    return json.loads(str(span.attributes[attr]))


@pytest.mark.asyncio
async def test_generate_span_input_config_uses_camel_case(exporter) -> None:
    """`config={'max_output_tokens': 64}` shows as maxOutputTokens on the generate span."""
    ai, seen = _app()

    await ai.generate(model=MODEL, prompt='Suggest a dish.', config={'temperature': 0.3, 'max_output_tokens': 64})

    spans = exporter.get_finished_spans()
    assert _span_json(spans, 'generate', Attr.INPUT)['config'] == {'temperature': 0.3, 'maxOutputTokens': 64}
    # The model span already used the wire spelling; both now agree.
    assert _span_json(spans, MODEL, Attr.INPUT)['config'] == {'temperature': 0.3, 'maxOutputTokens': 64}
    assert seen == [KitchenConfig(temperature=0.3, max_output_tokens=64)]


@pytest.mark.asyncio
async def test_generate_span_input_config_from_typed_object(exporter) -> None:
    """A KitchenConfig instance shows the same camelCase keys as a dict."""
    ai, seen = _app()

    await ai.generate(
        model=MODEL,
        prompt='Suggest a dish.',
        config=KitchenConfig(max_output_tokens=64, top_k=5, stop_sequences=['END']),
    )

    spans = exporter.get_finished_spans()
    assert _span_json(spans, 'generate', Attr.INPUT)['config'] == {
        'maxOutputTokens': 64,
        'topK': 5,
        'stopSequences': ['END'],
    }
    assert seen == [KitchenConfig(max_output_tokens=64, top_k=5, stop_sequences=['END'])]


@pytest.mark.asyncio
async def test_generate_span_keeps_unknown_keys_as_written(exporter) -> None:
    """Keys the config class doesn't declare pass through unrenamed."""
    ai, seen = _app(PassThroughConfig)

    await ai.generate(
        model=MODEL,
        prompt='Suggest a dish.',
        config={'max_output_tokens': 64, 'allergy_mode': 'strict'},
    )

    spans = exporter.get_finished_spans()
    assert _span_json(spans, 'generate', Attr.INPUT)['config'] == {'maxOutputTokens': 64, 'allergy_mode': 'strict'}
    assert seen[0].max_output_tokens == 64


@pytest.mark.asyncio
async def test_prompt_call_override_wins_over_frontmatter_in_one_key(exporter) -> None:
    """Prompt maxOutputTokens=256 with call max_output_tokens=64 yields one key, 64, on render and generate."""
    ai, seen = _app()
    menu = ai.define_prompt(
        name='menu',
        model=MODEL,
        prompt='Suggest a dish.',
        config={'maxOutputTokens': 256, 'temperature': 0.4},
    )

    await menu(config={'max_output_tokens': 64})

    spans = exporter.get_finished_spans()
    expected = {'maxOutputTokens': 64, 'temperature': 0.4}
    assert _span_json(spans, 'render', Attr.OUTPUT)['config'] == expected
    assert _span_json(spans, 'generate', Attr.INPUT)['config'] == expected
    assert seen == [KitchenConfig(max_output_tokens=64, temperature=0.4)]


@pytest.mark.asyncio
async def test_generate_without_config_class_keeps_keys_as_written(exporter) -> None:
    """A model with no config class owns its keys; Genkit doesn't rename them."""
    ai = Genkit()
    seen: list[Any] = []

    async def bare(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        seen.append(request.config)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))

    ai.define_model(name='bare', fn=bare)

    await ai.generate(model='bare', prompt='hi', config={'max_output_tokens': 64})

    spans = exporter.get_finished_spans()
    assert _span_json(spans, 'generate', Attr.INPUT)['config'] == {'max_output_tokens': 64}
    assert seen == [{'max_output_tokens': 64}]
