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

"""Anthropic Models for Genkit."""

from typing import Literal, TypeAlias, cast

from genkit.model import ConstrainedSupport, ModelInfo, ModelSupports

# Every model in the catalog takes media input, tools, a system prompt and
# multi-turn history, and supports native structured output (output_config).
_CLAUDE_SUPPORTS = ModelSupports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text', 'json'],
    constrained=ConstrainedSupport.ALL,
)

CLAUDE_FABLE_5_2 = ModelInfo(
    label='Anthropic - Claude Fable 5.2',
    versions=['claude-fable-5-2'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_FABLE_5_1 = ModelInfo(
    label='Anthropic - Claude Fable 5.1',
    versions=['claude-fable-5-1'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_FABLE_5 = ModelInfo(
    label='Anthropic - Claude Fable 5',
    versions=['claude-fable-5'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_OPUS_5_5 = ModelInfo(
    label='Anthropic - Claude Opus 5.5',
    versions=['claude-opus-5-5'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_OPUS_5 = ModelInfo(
    label='Anthropic - Claude Opus 5',
    versions=['claude-opus-5'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_OPUS_4_8 = ModelInfo(
    label='Anthropic - Claude Opus 4.8',
    versions=['claude-opus-4-8'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_OPUS_4_7 = ModelInfo(
    label='Anthropic - Claude Opus 4.7',
    versions=['claude-opus-4-7'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_OPUS_4_6 = ModelInfo(
    label='Anthropic - Claude Opus 4.6',
    versions=['claude-opus-4-6'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_OPUS_4_5 = ModelInfo(
    label='Anthropic - Claude Opus 4.5',
    versions=['claude-opus-4-5-20251101'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_SONNET_5_5 = ModelInfo(
    label='Anthropic - Claude Sonnet 5.5',
    versions=['claude-sonnet-5-5'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_SONNET_5 = ModelInfo(
    label='Anthropic - Claude Sonnet 5',
    versions=['claude-sonnet-5'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_SONNET_4_6 = ModelInfo(
    label='Anthropic - Claude Sonnet 4.6',
    versions=['claude-sonnet-4-6'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_HAIKU_5_5 = ModelInfo(
    label='Anthropic - Claude Haiku 5.5',
    versions=['claude-haiku-5-5'],
    supports=_CLAUDE_SUPPORTS,
)

CLAUDE_HAIKU_4_5 = ModelInfo(
    label='Anthropic - Claude Haiku 4.5',
    versions=['claude-haiku-4-5-20251001'],
    supports=_CLAUDE_SUPPORTS,
)

# Mythos models are limited to verified organizations, so they are not
# advertised; their ids still resolve through get_model_info.

# Quote autocomplete needs a Literal. The catalog below is what you edit when
# a Claude ships; a test requires these members and the dict keys to be the
# same set, and the dict is typed so a new key that is not in the Literal
# does not type-check.
KnownClaude: TypeAlias = Literal[
    'claude-fable-5-2',
    'claude-fable-5-1',
    'claude-fable-5',
    'claude-opus-5-5',
    'claude-opus-5',
    'claude-opus-4-8',
    'claude-opus-4-7',
    'claude-opus-4-6',
    'claude-opus-4-5',
    'claude-sonnet-5-5',
    'claude-sonnet-5',
    'claude-sonnet-4-6',
    'claude-haiku-5-5',
    'claude-haiku-4-5',
]

SUPPORTED_ANTHROPIC_MODELS: dict[KnownClaude, ModelInfo] = {
    'claude-fable-5-2': CLAUDE_FABLE_5_2,
    'claude-fable-5-1': CLAUDE_FABLE_5_1,
    'claude-fable-5': CLAUDE_FABLE_5,
    'claude-opus-5-5': CLAUDE_OPUS_5_5,
    'claude-opus-5': CLAUDE_OPUS_5,
    'claude-opus-4-8': CLAUDE_OPUS_4_8,
    'claude-opus-4-7': CLAUDE_OPUS_4_7,
    'claude-opus-4-6': CLAUDE_OPUS_4_6,
    'claude-opus-4-5': CLAUDE_OPUS_4_5,
    'claude-sonnet-5-5': CLAUDE_SONNET_5_5,
    'claude-sonnet-5': CLAUDE_SONNET_5,
    'claude-sonnet-4-6': CLAUDE_SONNET_4_6,
    'claude-haiku-5-5': CLAUDE_HAIKU_5_5,
    'claude-haiku-4-5': CLAUDE_HAIKU_4_5,
}

# Fallback for ids that are not in the catalog, such as Mythos ids, dated ids
# and Vertex snapshot ids (`<model>@<date>`). Native structured output can't be
# assumed for them: a model without it rejects output_config, so the request
# would fail instead of using prompt-based JSON.
DEFAULT_SUPPORTS = ModelSupports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['text'],
)


def get_model_info(name: str) -> ModelInfo:
    """Get model info for a given model name.

    Args:
        name: Model name.

    Returns:
        Model information.
    """
    if name in SUPPORTED_ANTHROPIC_MODELS:
        return SUPPORTED_ANTHROPIC_MODELS[cast(KnownClaude, name)]
    return ModelInfo(
        label=f'Anthropic - {name}',
        supports=DEFAULT_SUPPORTS,
    )
