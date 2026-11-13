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


"""OpenAI Compatible Models for Genkit."""

from typing import Literal, TypeAlias

from genkit.model import ConstrainedSupport, ModelInfo, ModelStage, ModelSupports

# 'json' in output means the endpoint accepts schema-less JSON mode.
# constrained=ALL means it also accepts response_format json_schema, so Genkit
# leaves the JSON format instructions out of the prompt. Models without it keep them.
MULTIMODAL_MODEL_SUPPORTS = ModelSupports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['json', 'text'],
    constrained=ConstrainedSupport.ALL,
)

O_SERIES_MODEL_SUPPORTS = ModelSupports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=False,
    output=['json', 'text'],
    constrained=ConstrainedSupport.ALL,
)

GPT_REASONING_MODEL_SUPPORTS = ModelSupports(
    multiturn=True,
    media=True,
    tools=True,
    system_role=True,
    output=['json', 'text'],
    constrained=ConstrainedSupport.ALL,
)

GPT_SEARCH_MODEL_SUPPORTS = ModelSupports(
    multiturn=True,
    media=False,
    tools=False,
    system_role=True,
    output=['json', 'text'],
    constrained=ConstrainedSupport.ALL,
)

# Chat Completions has no function tools for gpt-6-astra, gpt-6.1-sol and gpt-6.1-terra; they need Responses.
GPT_6_NO_TOOLS_MODEL_SUPPORTS = ModelSupports(
    multiturn=True,
    media=True,
    tools=False,
    system_role=True,
    output=['json', 'text'],
    constrained=ConstrainedSupport.ALL,
)

# Quote autocomplete needs a Literal. The catalog below is what you edit when
# a chat model ships; a test requires these members and the dict keys to be the
# same set, and the dict is typed so a new key that is not in the Literal
# does not type-check. Image, TTS, STT, and embedding ids live in other
# catalogs — they do not take OpenAIConfig.
KnownGpt: TypeAlias = Literal[
    'chat-latest',
    'gpt-4o',
    'gpt-4o-mini',
    'gpt-4o-mini-2024-07-18',
    'gpt-4.1',
    'gpt-4.1-mini',
    'o3',
    'gpt-5',
    'gpt-5-mini',
    'gpt-5-nano',
    'gpt-5-search-api',
    'gpt-5.1',
    'gpt-5.2',
    'gpt-5.4',
    'gpt-5.4-mini',
    'gpt-5.4-nano',
    'gpt-5.5',
    'gpt-5.6-sol',
    'gpt-5.6-terra',
    'gpt-5.6-luna',
    'gpt-6-astra',
    'gpt-6-sol',
    'gpt-6-luna',
    'gpt-6.1-sol',
    'gpt-6.1-terra',
]


def _info(name: str, supports: ModelSupports, *, deprecated: bool = False) -> ModelInfo:
    return ModelInfo(
        label=f'OpenAI - {name}',
        supports=supports,
        stage=ModelStage.DEPRECATED if deprecated else None,
    )


# Only ids that serve /v1/chat/completions are listed; Codex, -pro and gpt-oss
# ids are Responses-only. deprecated=True marks ids OpenAI is shutting down, with
# the date beside each group. The bare gpt-4o and gpt-4o-mini aliases have no
# date of their own and follow their dated snapshots.
SUPPORTED_OPENAI_MODELS: dict[KnownGpt, ModelInfo] = {
    # --- GPT-4o / GPT-4.1 (shutdown 2027-01-14) ---
    'gpt-4o': _info('gpt-4o', MULTIMODAL_MODEL_SUPPORTS, deprecated=True),
    'gpt-4o-mini': _info('gpt-4o-mini', MULTIMODAL_MODEL_SUPPORTS, deprecated=True),
    'gpt-4o-mini-2024-07-18': _info('gpt-4o-mini-2024-07-18', MULTIMODAL_MODEL_SUPPORTS, deprecated=True),
    'gpt-4.1': _info('gpt-4.1', MULTIMODAL_MODEL_SUPPORTS, deprecated=True),
    'gpt-4.1-mini': _info('gpt-4.1-mini', MULTIMODAL_MODEL_SUPPORTS, deprecated=True),
    # --- O-series (o3: shutdown 2026-12-11) ---
    'o3': _info('o3', O_SERIES_MODEL_SUPPORTS, deprecated=True),
    # --- GPT-5 (shutdown 2026-12-11) ---
    'gpt-5': _info('gpt-5', GPT_REASONING_MODEL_SUPPORTS, deprecated=True),
    'gpt-5-mini': _info('gpt-5-mini', GPT_REASONING_MODEL_SUPPORTS, deprecated=True),
    'gpt-5-nano': _info('gpt-5-nano', GPT_REASONING_MODEL_SUPPORTS, deprecated=True),
    # --- GPT-5.1 to GPT-5.6 (gpt-5.1 and gpt-5.4-nano: shutdown 2027-04-01) ---
    'gpt-5.1': _info('gpt-5.1', GPT_REASONING_MODEL_SUPPORTS, deprecated=True),
    'gpt-5.2': _info('gpt-5.2', GPT_REASONING_MODEL_SUPPORTS),
    # gpt-5.4*, gpt-5.5, gpt-5.6-* and gpt-6-sol/luna take function tools on Chat
    # Completions only at reasoning_effort='none'.
    'gpt-5.4': _info('gpt-5.4', GPT_REASONING_MODEL_SUPPORTS),
    'gpt-5.4-mini': _info('gpt-5.4-mini', GPT_REASONING_MODEL_SUPPORTS),
    'gpt-5.4-nano': _info('gpt-5.4-nano', GPT_REASONING_MODEL_SUPPORTS, deprecated=True),
    'gpt-5.5': _info('gpt-5.5', GPT_REASONING_MODEL_SUPPORTS),
    'gpt-5.6-sol': _info('gpt-5.6-sol', GPT_REASONING_MODEL_SUPPORTS),
    'gpt-5.6-terra': _info('gpt-5.6-terra', GPT_REASONING_MODEL_SUPPORTS),
    'gpt-5.6-luna': _info('gpt-5.6-luna', GPT_REASONING_MODEL_SUPPORTS),
    # --- Search and ChatGPT ---
    'gpt-5-search-api': _info('gpt-5-search-api', GPT_SEARCH_MODEL_SUPPORTS),
    'chat-latest': _info('chat-latest', MULTIMODAL_MODEL_SUPPORTS),
    # --- GPT-6 ---
    'gpt-6-astra': _info('gpt-6-astra', GPT_6_NO_TOOLS_MODEL_SUPPORTS),
    'gpt-6-sol': _info('gpt-6-sol', GPT_REASONING_MODEL_SUPPORTS),
    'gpt-6-luna': _info('gpt-6-luna', GPT_REASONING_MODEL_SUPPORTS),
    'gpt-6.1-sol': _info('gpt-6.1-sol', GPT_6_NO_TOOLS_MODEL_SUPPORTS),
    'gpt-6.1-terra': _info('gpt-6.1-terra', GPT_6_NO_TOOLS_MODEL_SUPPORTS),
}

SUPPORTED_EMBEDDING_MODELS: dict[str, dict] = {
    'text-embedding-3-small': {
        'label': 'OpenAI - text-embedding-3-small',
        'dimensions': 1536,
        'supports': {'input': ['text']},
    },
    'text-embedding-3-large': {
        'label': 'OpenAI - text-embedding-3-large',
        'dimensions': 3072,
        'supports': {'input': ['text']},
    },
    'text-embedding-ada-002': {
        'label': 'OpenAI - text-embedding-ada-002',
        'dimensions': 1536,
        'supports': {'input': ['text']},
    },
}


def get_default_openai_model_info(name: str) -> ModelInfo:
    """Gets the default model info given a name."""
    return ModelInfo(label=f'OpenAI - {name}', supports=ModelSupports(multiturn=True))
