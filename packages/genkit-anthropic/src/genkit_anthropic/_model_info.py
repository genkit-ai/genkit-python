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

# Source: https://platform.claude.com/docs/en/models/overview and
# https://platform.claude.com/docs/en/about-claude/model-deprecations
# (checked 2026-11-25). The catalog holds every model the Claude API serves
# today, active or deprecated; retired models are dropped. Any other id still
# resolves on demand, with DEFAULT_SUPPORTS.


def _claude(label: str, version: str) -> ModelInfo:
    """A cataloged Claude: text, images, tools, and native structured output.

    ``version`` is the id sent to the API. From the 4.6 generation on, the
    dateless id is itself a pinned snapshot; older models send the dated id.
    """
    return ModelInfo(
        label=f'Anthropic - {label}',
        versions=[version],
        supports=ModelSupports(
            multiturn=True,
            media=True,
            tools=True,
            system_role=True,
            output=['text', 'json'],
            constrained=ConstrainedSupport.ALL,
        ),
    )


# Latest models.
CLAUDE_FABLE_5_2 = _claude('Claude Fable 5.2', 'claude-fable-5-2')
CLAUDE_OPUS_5_5 = _claude('Claude Opus 5.5', 'claude-opus-5-5')
CLAUDE_SONNET_5_5 = _claude('Claude Sonnet 5.5', 'claude-sonnet-5-5')
CLAUDE_HAIKU_5_5 = _claude('Claude Haiku 5.5', 'claude-haiku-5-5')

# Earlier models, still served.
CLAUDE_FABLE_5_1 = _claude('Claude Fable 5.1', 'claude-fable-5-1')
CLAUDE_FABLE_5 = _claude('Claude Fable 5', 'claude-fable-5')
CLAUDE_OPUS_5 = _claude('Claude Opus 5', 'claude-opus-5')
CLAUDE_OPUS_4_8 = _claude('Claude Opus 4.8', 'claude-opus-4-8')
CLAUDE_OPUS_4_7 = _claude('Claude Opus 4.7', 'claude-opus-4-7')
CLAUDE_OPUS_4_6 = _claude('Claude Opus 4.6', 'claude-opus-4-6')
CLAUDE_SONNET_5 = _claude('Claude Sonnet 5', 'claude-sonnet-5')
CLAUDE_SONNET_4_6 = _claude('Claude Sonnet 4.6', 'claude-sonnet-4-6')
CLAUDE_HAIKU_4_5 = _claude('Claude Haiku 4.5', 'claude-haiku-4-5-20251001')

# Deprecated: Opus 4.5 retires 2027-01-22, Sonnet 4.5 retires 2026-11-30.
CLAUDE_OPUS_4_5 = _claude('Claude Opus 4.5', 'claude-opus-4-5-20251101')
CLAUDE_SONNET_4_5 = _claude('Claude Sonnet 4.5', 'claude-sonnet-4-5-20250929')

# Quote autocomplete needs a Literal. The catalog below is what you edit when
# a Claude ships or retires; a test requires these members and the dict keys
# to be the same set, and the dict is typed so a new key that is not in the
# Literal does not type-check.
KnownClaude: TypeAlias = Literal[
    'claude-fable-5-2',
    'claude-opus-5-5',
    'claude-sonnet-5-5',
    'claude-haiku-5-5',
    'claude-fable-5-1',
    'claude-fable-5',
    'claude-opus-5',
    'claude-opus-4-8',
    'claude-opus-4-7',
    'claude-opus-4-6',
    'claude-sonnet-5',
    'claude-sonnet-4-6',
    'claude-haiku-4-5',
    'claude-opus-4-5',
    'claude-sonnet-4-5',
]

SUPPORTED_ANTHROPIC_MODELS: dict[KnownClaude, ModelInfo] = {
    'claude-fable-5-2': CLAUDE_FABLE_5_2,
    'claude-opus-5-5': CLAUDE_OPUS_5_5,
    'claude-sonnet-5-5': CLAUDE_SONNET_5_5,
    'claude-haiku-5-5': CLAUDE_HAIKU_5_5,
    'claude-fable-5-1': CLAUDE_FABLE_5_1,
    'claude-fable-5': CLAUDE_FABLE_5,
    'claude-opus-5': CLAUDE_OPUS_5,
    'claude-opus-4-8': CLAUDE_OPUS_4_8,
    'claude-opus-4-7': CLAUDE_OPUS_4_7,
    'claude-opus-4-6': CLAUDE_OPUS_4_6,
    'claude-sonnet-5': CLAUDE_SONNET_5,
    'claude-sonnet-4-6': CLAUDE_SONNET_4_6,
    'claude-haiku-4-5': CLAUDE_HAIKU_4_5,
    'claude-opus-4-5': CLAUDE_OPUS_4_5,
    'claude-sonnet-4-5': CLAUDE_SONNET_4_5,
}

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
