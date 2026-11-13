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

"""OpenAI-compatible image generation model for Genkit.

Provides image generation capabilities via the OpenAI Images API,
supporting the GPT Image models (gpt-image-2, gpt-image-2.5 and gpt-image-2.6).
"""

from __future__ import annotations

from typing import Any, Literal

from genkit_openai._models._utils import (
    OPENAI_CALL_ERRORS,
    _extract_text,
    extract_config_dict,
    pop_extra_body,
    reject_version,
    reraise_openai_error,
)
from openai import AsyncOpenAI
from openai.types.images_response import ImagesResponse
from pydantic import Field

from genkit import FinishReason, Message, ModelResponse, Part, Role
from genkit.model import ModelContext, ModelInfo, ModelRequest, ModelStage, ModelSupports
from genkit.plugin_api import GenkitConfig

_IMAGE_EXTRA_DESCRIPTION = (
    'Images API fields this class does not declare, in API wire names, sent as the request body. '
    'prompt and model are built by Genkit and rejected here. Not checked; do not put API keys here.'
)


# The image configs extend GenkitConfig, not ModelConfig: the Images API has
# no temperature, token cap, top_k/top_p, stop strings, or version pin, so
# those keys raise instead of being accepted and dropped.
class OpenAIDalleConfig(GenkitConfig):
    """Settings the DALL-E image endpoint reads from config."""

    size: str | None = None
    quality: Literal['standard', 'hd'] | None = None
    style: Literal['vivid', 'natural'] | None = None
    n: int | None = None
    user: str | None = None
    response_format: str | None = None
    extra: dict[str, Any] | None = Field(default=None, description=_IMAGE_EXTRA_DESCRIPTION)


class OpenAIGptImageConfig(GenkitConfig):
    """Settings the GPT Image endpoint reads from config."""

    size: Literal['1024x1024', '1536x1024', '1024x1536', 'auto'] | None = None
    style: Literal['vivid', 'natural'] | None = None
    user: str | None = None
    n: int | None = Field(default=None, ge=1, le=10)
    quality: Literal['low', 'medium', 'high', 'xhigh', 'max', 'auto'] | None = Field(
        default=None,
        json_schema_extra={'enum': ['low', 'medium', 'high', 'xhigh', 'max', 'auto']},
    )
    background: Literal['transparent', 'opaque', 'auto'] | None = None
    moderation: Literal['low', 'auto'] | None = None
    output_compression: int | None = Field(default=None, ge=1, le=100)
    output_format: Literal['png', 'jpeg', 'webp'] | None = None
    extra: dict[str, Any] | None = Field(default=None, description=_IMAGE_EXTRA_DESCRIPTION)


# Supported image generation models with their metadata. Other gpt-image
# ids still resolve by name.
_GPT_IMAGE_SUPPORTS = ModelSupports(
    media=False,
    output=['media'],
    multiturn=False,
    system_role=False,
    tools=False,
)

SUPPORTED_IMAGE_MODELS: dict[str, ModelInfo] = {
    # Shutdown 2027-02-10.
    'gpt-image-2': ModelInfo(label='OpenAI - GPT Image 2', supports=_GPT_IMAGE_SUPPORTS, stage=ModelStage.DEPRECATED),
    'gpt-image-2.5-sunburst': ModelInfo(label='OpenAI - GPT Image 2.5 Sunburst', supports=_GPT_IMAGE_SUPPORTS),
    'gpt-image-2.5-flare': ModelInfo(label='OpenAI - GPT Image 2.5 Flare', supports=_GPT_IMAGE_SUPPORTS),
    'gpt-image-2.6-sunburst': ModelInfo(label='OpenAI - GPT Image 2.6 Sunburst', supports=_GPT_IMAGE_SUPPORTS),
    'gpt-image-2.6-flare': ModelInfo(label='OpenAI - GPT Image 2.6 Flare', supports=_GPT_IMAGE_SUPPORTS),
}


# Re-export _extract_text as _extract_prompt_text for backward compatibility.
_extract_prompt_text = _extract_text


def _to_image_generate_params(
    model_name: str,
    request: ModelRequest,
) -> dict[str, Any]:
    """Convert a ModelRequest into OpenAI image generation parameters.

    Extracts the text prompt and maps Genkit config options to OpenAI's
    image generation API parameters.

    Args:
        model_name: The OpenAI model name (e.g., 'gpt-image-2.5-flare').
        request: The Genkit generate request.

    Returns:
        A dictionary of parameters for client.images.generate().
    """
    prompt = _extract_prompt_text(request)
    config = extract_config_dict(request)
    extra_body = pop_extra_body(config, managed=('prompt', 'model'), label='openai image')
    reject_version(config, label='openai image')

    # Start with required params.
    params: dict[str, Any] = {
        'model': model_name,
        'prompt': prompt,
    }

    # GPT Image models reject response_format and always return base64 data.
    if 'gpt-image' not in model_name.lower() and 'chatgpt-image' not in model_name.lower():
        params['response_format'] = config.pop('response_format', 'b64_json')
    else:
        config.pop('response_format', None)

    # Pass remaining config through (size, quality, style, n, etc.).
    params.update(config)
    if extra_body:
        params['extra_body'] = extra_body

    # Remove None values.
    return {k: v for k, v in params.items() if v is not None}


def _to_generate_response(result: ImagesResponse) -> ModelResponse:
    """Convert an OpenAI ImagesResponse to a Genkit ModelResponse.

    Each generated image becomes a media part in the response message.

    Args:
        result: The OpenAI images.generate() response object.

    Returns:
        A ModelResponse with media parts for each generated image.
    """
    images = result.data
    if not images:
        return ModelResponse(
            message=Message(role=Role.MODEL, content=[]),
            finish_reason=FinishReason.STOP,
        )

    content: list[Part] = []
    for image in images:
        url = image.url
        if not url and image.b64_json:
            url = f'data:image/png;base64,{image.b64_json}'

        if url:
            content.append(Part.from_media(url, content_type='image/png'))

    return ModelResponse(
        message=Message(role=Role.MODEL, content=content),
        finish_reason=FinishReason.STOP,
    )


class OpenAIImageModel:
    """Handles image generation via the OpenAI Images API.

    Args:
        model_name: The image model to use (e.g., 'gpt-image-2.5-flare').
        client: An async OpenAI client instance.
    """

    def __init__(self, model_name: str, client: AsyncOpenAI) -> None:
        """Initialize the image model.

        Args:
            model_name: The image model to use (e.g., 'gpt-image-2.5-flare').
            client: An async OpenAI client instance.
        """
        self._model_name = model_name
        self._client = client

    @property
    def name(self) -> str:
        """The name of the image model."""
        return self._model_name

    async def generate(self, request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        """Generate images from the request.

        Args:
            request: The generate request containing the text prompt.
            ctx: The action run context.

        Returns:
            A ModelResponse containing generated image media parts.
        """
        try:
            params = _to_image_generate_params(self._model_name, request)
            result = await self._client.images.generate(**params)
            return _to_generate_response(result)
        except OPENAI_CALL_ERRORS as e:
            reraise_openai_error(e)
