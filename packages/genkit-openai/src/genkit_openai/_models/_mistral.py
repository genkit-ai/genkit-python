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
JSON handling is, or set with ``OpenAI(provider='mistral')`` for a gateway in
front of Mistral on another host. Mistral's ids share no prefix (``mistral-``, ``magistral-``,
``codestral-``, ``pixtral-``, ...), and the same ids are served through other
OpenAI-compatible hosts that take OpenAI's names: Vertex Model Garden reuses
``OpenAIModel`` for ``mistralai/...`` models with a Vertex client, so it never
matches here.

Source: https://docs.mistral.ai/openapi.yaml (``ChatCompletionRequest``,
``EmbeddingRequest``).
"""

from typing import Any

import httpx
from genkit_openai._models._utils import parse_data_uri_content_type
from openai import AsyncOpenAI

from genkit import GenkitError, Message

# Mistral's API hosts: global, EU and US, the servers in the mistralai SDK's
# sdkconfiguration.py. codestral.mistral.ai, for Codestral-only keys, is not in
# that list or the OpenAPI servers; it is kept since nothing else lives there.
MISTRAL_HOSTS = frozenset({'api.mistral.ai', 'api.eu.mistral.ai', 'api.us.mistral.ai', 'codestral.mistral.ai'})


# Where OpenAI(provider='mistral') sends calls when no base_url is given, as
# JS's and Dart's mistral() do.
MISTRAL_BASE_URL = 'https://api.mistral.ai/v1'

# The key Mistral's own SDKs, and Genkit's JS and Dart plugins, read.
MISTRAL_API_KEY_ENV = 'MISTRAL_API_KEY'


def is_mistral_url(url: str | httpx.URL) -> bool:
    """Whether ``url`` is on one of Mistral's API hosts."""
    return httpx.URL(str(url)).host in MISTRAL_HOSTS


def is_mistral(client: AsyncOpenAI) -> bool:
    """Whether ``client`` sends its requests to one of Mistral's API hosts."""
    return is_mistral_url(client.base_url)


def _rename(body: dict[str, Any], old: str, new: str) -> None:
    """Move ``body[old]`` to ``body[new]``, unless ``new`` is already set there."""
    if old in body:
        value = body.pop(old)
        body.setdefault(new, value)


def mistral_chat_body(body: dict[str, Any], *, top_k: int | None = None) -> dict[str, Any]:
    """Rewrite ``chat.completions.create()`` kwargs into a body Mistral accepts.

    - ``stream_options`` is dropped. Mistral reports usage on the last
      streamed chunk without being asked, so it isn't needed.
    - ``max_completion_tokens`` goes out as ``max_tokens``, Mistral's only
      reply cap, including when ``reasoning_effort`` is set.
    - ``seed`` goes out as ``random_seed``.
    - ``top_k`` is refused. Mistral's schema declares it, but its models
      answer "top_k sampling is not enabled for this model".

    The SDK has no ``random_seed`` kwarg, so it rides in ``extra_body``; the
    same key in the caller's ``extra`` wins, as every ``extra`` key does.
    ``extra`` gets the same renames: its ``seed``, ``max_completion_tokens``
    and ``stream_options`` would otherwise reach Mistral as given and fail
    with a 422. Where ``extra`` has both names, Mistral's wins.

    Args:
        body: The kwargs the plugin built for OpenAI. Not modified.
        top_k: Genkit's ``top_k`` setting, which the OpenAI kwargs leave out.

    Returns:
        A copy with Mistral's field names.

    Raises:
        GenkitError: ``INVALID_ARGUMENT`` when ``top_k`` is set.
    """
    if top_k is not None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message="Mistral doesn't support top_k: its models reject top_k sampling. Remove top_k from the config.",
        )
    out = dict(body)
    out.pop('stream_options', None)
    if 'max_completion_tokens' in out:
        out['max_tokens'] = out.pop('max_completion_tokens')
    renamed: dict[str, Any] = {}
    if 'seed' in out:
        renamed['random_seed'] = out.pop('seed')
    extra = dict(out.pop('extra_body', None) or {})
    extra.pop('stream_options', None)
    _rename(extra, 'max_completion_tokens', 'max_tokens')
    _rename(extra, 'seed', 'random_seed')
    if renamed or extra:
        out['extra_body'] = {**renamed, **extra}
    return out


def refuse_non_image_media(messages: list[Message]) -> None:
    """Refuse media Mistral's chat endpoint can't take as an image.

    Every media part goes out as an ``image_url`` chunk, so a PDF or audio
    clip would reach Mistral as a broken image and fail there. The type is the
    part's ``content_type``, else the one a ``data:`` URL declares; a remote
    URL with neither is sent as is, since only the host knows what it serves.

    Raises:
        GenkitError: ``INVALID_ARGUMENT`` naming the first non-image type.
    """
    for message in messages:
        for part in message.content:
            media = part.media
            if media is None:
                continue
            declared = media.content_type or parse_data_uri_content_type(media.url)
            content_type = declared.split(';', 1)[0].strip().lower()
            if content_type and not content_type.startswith('image/'):
                raise GenkitError(
                    status='INVALID_ARGUMENT',
                    message=f'Mistral only supports image media in chat requests, got {content_type}.',
                )
