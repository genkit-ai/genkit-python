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

``Mistral()`` (``OpenAI(provider='mistral')``, or ``OpenAI`` with a Mistral
``base_url``) talks to Mistral. Its chat and embeddings endpoints take the
OpenAI shape with a few fields renamed or missing, and reject any field they
don't declare, so a body built for OpenAI fails with a 422. The plugin
renames the fields that differ and leaves out the OpenAI-only ones. This
module also holds the catalog and ``/models`` reading a Mistral target uses.

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

from collections.abc import Mapping
from typing import Any, cast

import httpx
from genkit_openai._models._utils import parse_data_uri_content_type
from openai import AsyncOpenAI

from genkit import GenkitError, Message, get_logger
from genkit.model import ConstrainedSupport, ModelInfo, ModelStage, ModelSupports

logger = get_logger(__name__)

# Mistral's API hosts: global, EU and US, the servers in the mistralai SDK's
# sdkconfiguration.py. codestral.mistral.ai, for Codestral-only keys, is not in
# that list or the OpenAPI servers; it is kept since nothing else lives there.
MISTRAL_HOSTS = frozenset({'api.mistral.ai', 'api.eu.mistral.ai', 'api.us.mistral.ai', 'codestral.mistral.ai'})


# Where OpenAI(provider='mistral') sends calls when no base_url is given, as
# JS's and Dart's mistral() do.
MISTRAL_BASE_URL = 'https://api.mistral.ai/v1'

# The key Mistral's own SDKs, and Genkit's JS and Dart plugins, read.
MISTRAL_API_KEY_ENV = 'MISTRAL_API_KEY'

# The plugin namespace for a Mistral target, as in JS and Dart: mistral/<id>.
MISTRAL_NAMESPACE = 'mistral'

# The top-level fields Mistral's ChatCompletionRequest declares
# (https://docs.mistral.ai/openapi.yaml). Its schema is closed: anything else
# is a 422 "Extra inputs are not permitted". ``extra_body`` is the SDK's
# passthrough, not a wire field, and is kept.
MISTRAL_CHAT_FIELDS = frozenset({
    'model',
    'messages',
    'temperature',
    'top_p',
    'max_tokens',
    'stream',
    'stop',
    'random_seed',
    'metadata',
    'response_format',
    'tools',
    'tool_choice',
    'presence_penalty',
    'frequency_penalty',
    'min_tokens',
    'repetition_penalty',
    'top_k',
    'logprobs',
    'top_logprobs',
    'prompt_logprobs',
    'top_prompt_logprobs',
    'n',
    'prediction',
    'parallel_tool_calls',
    'reasoning_effort',
    'prompt_mode',
    'guardrails',
    'prompt_cache_key',
    'service_tier',
    'safe_prompt',
})

# Chat models with vision, tools and structured output (JS's
# MISTRAL_MODEL_INFO, Dart's multimodalSupports).
_MISTRAL_MULTIMODAL_SUPPORTS = ModelSupports(
    multiturn=True,
    media=True,
    tools=True,
    tool_choice=True,
    system_role=True,
    output=['json', 'text'],
    constrained=ConstrainedSupport.ALL,
)

# Codestral: the same without image input.
_MISTRAL_TEXT_SUPPORTS = ModelSupports(
    multiturn=True,
    media=False,
    tools=True,
    tool_choice=True,
    system_role=True,
    output=['json', 'text'],
    constrained=ConstrainedSupport.ALL,
)

# A model outside the catalog with no /models capabilities to go by: text in,
# text out, as JS's and Dart's unknown-model info.
MISTRAL_UNKNOWN_MODEL_SUPPORTS = ModelSupports(
    multiturn=True,
    media=False,
    tools=False,
    tool_choice=False,
    system_role=True,
    output=['text'],
)

# The chat models a Mistral target registers at init, the set JS's
# SUPPORTED_MISTRAL_MODELS and Dart's KnownMistralModel register.
SUPPORTED_MISTRAL_MODELS: dict[str, ModelInfo] = {
    'mistral-large-latest': ModelInfo(label='Mistral Large', supports=_MISTRAL_MULTIMODAL_SUPPORTS),
    'mistral-large-4': ModelInfo(label='Mistral Large 4', supports=_MISTRAL_MULTIMODAL_SUPPORTS),
    'mistral-small-latest': ModelInfo(label='Mistral Small', supports=_MISTRAL_MULTIMODAL_SUPPORTS),
    'ministral-14b-latest': ModelInfo(label='Ministral 14B', supports=_MISTRAL_MULTIMODAL_SUPPORTS),
    'ministral-8b-latest': ModelInfo(label='Ministral 8B', supports=_MISTRAL_MULTIMODAL_SUPPORTS),
    'ministral-3b-latest': ModelInfo(label='Ministral 3B', supports=_MISTRAL_MULTIMODAL_SUPPORTS),
    'codestral-latest': ModelInfo(label='Codestral', supports=_MISTRAL_TEXT_SUPPORTS),
    # Mistral Medium 3.5, which Mistral has deprecated (retiring 2027-04-30).
    'mistral-medium-latest': ModelInfo(
        label='Mistral Medium', supports=_MISTRAL_MULTIMODAL_SUPPORTS, stage=ModelStage.LEGACY
    ),
}

# The embedders a Mistral target registers at init.
SUPPORTED_MISTRAL_EMBEDDERS: dict[str, dict[str, Any]] = {
    'mistral-embed': {'label': 'Mistral Embed', 'dimensions': 1024, 'supports': {'input': ['text']}},
    'codestral-embed': {'label': 'Codestral Embed', 'dimensions': 1536, 'supports': {'input': ['text']}},
}


def mistral_model_supports(name: str, capabilities: Mapping[str, object] | None = None) -> ModelSupports:
    """What a Mistral chat model supports.

    The catalog's entry when ``name`` is in it; else read from the ``/models``
    entry's ``capabilities`` (``function_calling`` for tools, ``vision`` for
    media), with text-only output since capabilities don't say whether the
    model takes a JSON schema; else :data:`MISTRAL_UNKNOWN_MODEL_SUPPORTS`.
    """
    if (known := SUPPORTED_MISTRAL_MODELS.get(name)) is not None and known.supports is not None:
        return known.supports
    if capabilities is None:
        return MISTRAL_UNKNOWN_MODEL_SUPPORTS
    tools = capabilities.get('function_calling') is True
    return MISTRAL_UNKNOWN_MODEL_SUPPORTS.model_copy(
        update={'tools': tools, 'tool_choice': tools, 'media': capabilities.get('vision') is True}
    )


def listed_capabilities(entry: object) -> Mapping[str, object] | None:
    """The ``capabilities`` object of a Mistral ``/models`` entry, or None when it has none.

    It is a Mistral field, not part of OpenAI's ``Model``; the SDK keeps it in
    ``model_extra``.
    """
    extra = getattr(entry, 'model_extra', None) or {}
    capabilities = extra.get('capabilities') if isinstance(extra, Mapping) else None
    return cast(Mapping[str, object], capabilities) if isinstance(capabilities, Mapping) else None


def mistral_serves_chat(model_id: str, capabilities: Mapping[str, object] | None) -> bool:
    """Whether a ``/models`` entry is a chat model, from ``capabilities.completion_chat``.

    That leaves out OCR, moderation, transcription and speech models. An
    entry with no ``capabilities`` counts as chat unless its id names OCR,
    which ``/v1/ocr`` serves, as Dart's plugin does.
    """
    if capabilities is None:
        return 'ocr' not in model_id.lower()
    return capabilities.get('completion_chat') is True


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
    - Any other field Mistral's schema doesn't declare is dropped, as Dart's
      ``mistralChatBody`` drops ``user`` and ``verbosity``: OpenAI-only
      settings such as ``verbosity``, ``safety_identifier``, ``store``,
      ``logit_bias``, ``prompt_cache_retention``, ``web_search_options``,
      ``modalities`` and ``audio`` would otherwise fail the call with a 422.
    - ``logprobs=True`` and any ``top_logprobs`` are refused, as in JS:
      Mistral's schema declares them, but its models answer "Logprobs are not
      enabled for this model". ``logprobs=False`` is left out.

    The SDK has no ``random_seed`` kwarg, so it rides in ``extra_body``; the
    same key in the caller's ``extra`` wins, as every ``extra`` key does.
    ``extra`` gets the same renames: its ``seed``, ``max_completion_tokens``
    and ``stream_options`` would otherwise reach Mistral as given and fail
    with a 422. Where ``extra`` has both names, Mistral's wins. Nothing else
    in ``extra`` is dropped: it is sent as the caller wrote it.

    Args:
        body: The kwargs the plugin built for OpenAI. Not modified.
        top_k: Genkit's ``top_k`` setting, which the OpenAI kwargs leave out.

    Returns:
        A copy with Mistral's field names.

    Raises:
        GenkitError: ``INVALID_ARGUMENT`` when ``top_k``, ``logprobs=True``
            or ``top_logprobs`` is set.
    """
    if top_k is not None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message="Mistral doesn't support top_k: its models reject top_k sampling. Remove top_k from the config.",
        )
    # top_logprobs=0 is still a request for logprobs; only logprobs=False isn't.
    asked = [('logprobs', body.get('logprobs') is True), ('top_logprobs', body.get('top_logprobs') is not None)]
    for field, set_ in asked:
        if set_:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f"Mistral doesn't support {field}: its models answer 'Logprobs are not enabled for this model'. "
                    f'Remove {field} from the config.'
                ),
            )
    out = dict(body)
    out.pop('logprobs', None)
    out.pop('top_logprobs', None)
    out.pop('stream_options', None)
    if 'max_completion_tokens' in out:
        out['max_tokens'] = out.pop('max_completion_tokens')
    renamed: dict[str, Any] = {}
    if 'seed' in out:
        renamed['random_seed'] = out.pop('seed')
    if dropped := sorted(k for k in out if k not in MISTRAL_CHAT_FIELDS and k != 'extra_body'):
        logger.warning('Mistral does not define these OpenAI settings; not sending them', fields=dropped)
        for key in dropped:
            del out[key]
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
