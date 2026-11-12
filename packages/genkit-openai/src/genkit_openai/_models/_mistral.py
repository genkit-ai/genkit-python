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

"""Request rewrites for Mistral's OpenAI-compatible API.

``OpenAI(base_url='https://api.mistral.ai/v1')`` talks to Mistral. Its chat and
embeddings endpoints take the OpenAI shape with a few fields renamed or
missing, and reject any field they don't declare, so a body built for OpenAI
fails with a 422. The plugin rewrites the few fields that differ; everything
else goes out unchanged.

The target is read from the client's base URL, not the model id as DeepSeek's
JSON handling is. Mistral's ids share no prefix (``mistral-``, ``magistral-``,
``codestral-``, ``pixtral-``, ...), and the same ids are served through other
OpenAI-compatible hosts that take OpenAI's names: Vertex Model Garden reuses
``OpenAIModel`` for ``mistralai/...`` models with a Vertex client, so it never
matches here.

Source: https://docs.mistral.ai/openapi.yaml (``ChatCompletionRequest``,
``EmbeddingRequest``).
"""

from typing import Any

from openai import AsyncOpenAI

# Mistral's API hosts: global, EU and US, the servers in the mistralai SDK's
# sdkconfiguration.py. codestral.mistral.ai, for Codestral-only keys, is not in
# that list or the OpenAPI servers; it is kept since nothing else lives there.
MISTRAL_HOSTS = frozenset({'api.mistral.ai', 'api.eu.mistral.ai', 'api.us.mistral.ai', 'codestral.mistral.ai'})


def is_mistral(client: AsyncOpenAI) -> bool:
    """Whether ``client`` sends its requests to one of Mistral's API hosts."""
    return client.base_url.host in MISTRAL_HOSTS


def mistral_chat_body(body: dict[str, Any], *, top_k: int | None = None) -> dict[str, Any]:
    """Rewrite ``chat.completions.create()`` kwargs into a body Mistral accepts.

    - ``stream_options`` is dropped. Mistral has no such field; it reports
      usage on the last streamed chunk without being asked.
    - ``max_completion_tokens`` goes out as ``max_tokens``, Mistral's only
      reply cap, including when ``reasoning_effort`` is set.
    - ``seed`` goes out as ``random_seed``.
    - ``top_k``, which OpenAI has no field for, goes out as is.

    The SDK has no ``random_seed`` or ``top_k`` kwarg, so both ride in
    ``extra_body``; the same key in the caller's ``extra`` wins, as every
    ``extra`` key does.

    Args:
        body: The kwargs the plugin built for OpenAI. Not modified.
        top_k: Genkit's ``top_k`` setting, which the OpenAI kwargs leave out.

    Returns:
        A copy with Mistral's field names.
    """
    out = dict(body)
    out.pop('stream_options', None)
    if 'max_completion_tokens' in out:
        out['max_tokens'] = out.pop('max_completion_tokens')
    renamed: dict[str, Any] = {}
    if 'seed' in out:
        renamed['random_seed'] = out.pop('seed')
    if top_k is not None:
        renamed['top_k'] = top_k
    if renamed:
        out['extra_body'] = {**renamed, **(out.get('extra_body') or {})}
    return out
