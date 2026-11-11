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
fails with a 422. The plugin derives the target from the client's base URL
and rewrites the few fields that differ; everything else goes out unchanged.

Source: https://docs.mistral.ai/openapi.yaml (``ChatCompletionRequest``,
``EmbeddingRequest``).
"""

from typing import Any

from openai import AsyncOpenAI

MISTRAL_HOST = 'api.mistral.ai'


def is_mistral(client: AsyncOpenAI) -> bool:
    """Whether ``client`` sends its requests to Mistral's API."""
    return getattr(client.base_url, 'host', None) == MISTRAL_HOST


def mistral_chat_body(body: dict[str, Any]) -> dict[str, Any]:
    """Rewrite ``chat.completions.create()`` kwargs into a body Mistral accepts.

    - ``stream_options`` is dropped. Mistral has no such field; it reports
      usage on the last streamed chunk without being asked.
    - ``max_completion_tokens`` goes out as ``max_tokens``, Mistral's only
      reply cap, including when ``reasoning_effort`` is set.
    - ``seed`` goes out as ``random_seed``. The SDK has no ``random_seed``
      kwarg, so it rides in ``extra_body``; a ``random_seed`` the caller put in
      ``extra`` wins, as every ``extra`` key does.

    Args:
        body: The kwargs the plugin built for OpenAI. Not modified.

    Returns:
        A copy with Mistral's field names.
    """
    out = dict(body)
    out.pop('stream_options', None)
    if 'max_completion_tokens' in out:
        out['max_tokens'] = out.pop('max_completion_tokens')
    if 'seed' in out:
        seed = out.pop('seed')
        out['extra_body'] = {'random_seed': seed, **(out.get('extra_body') or {})}
    return out
