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

"""Tests for OpenAI-compatible image generation model."""

from __future__ import annotations

from typing import Any, Literal, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from genkit_openai import OpenAI
from genkit_openai._models._image import (
    SUPPORTED_IMAGE_MODELS,
    OpenAIDalleConfig,
    OpenAIGptImageConfig,
    OpenAIImageModel,
    _extract_prompt_text,
    _to_generate_response,
    _to_image_generate_params,
)
from genkit_openai._openai_plugin import _ModelType
from pydantic import BaseModel, ValidationError

from genkit import FinishReason, Genkit, GenkitError, Message, Part, Role
from genkit.model import ModelRequest, ModelStage
from genkit.plugin_api import ActionKind

_Quality = Literal['low', 'medium', 'high', 'xhigh', 'max', 'auto']
_MAX_QUALITY = ['low', 'medium', 'high', 'xhigh', 'max', 'auto']
_STANDARD_QUALITY = ['low', 'medium', 'high', 'auto']
_MAX_QUALITY_MODELS = [
    'gpt-image-2.5-sunburst',
    'gpt-image-2.5-sunburst-2026-09-08',
    'gpt-image-2.5-flare',
    'gpt-image-2.5-flare-2026-09-08',
    'gpt-image-2.6-sunburst',
    'gpt-image-2.6-flare',
]
_STANDARD_QUALITY_MODELS = [
    'gpt-image-2',
    'gpt-image-2-2026-04-21',
    'gpt-image-1.5',
    'gpt-image-1-mini',
    'gpt-image-1',
    'chatgpt-image-latest',
]


def _genkit_with_image_client() -> tuple[Genkit, MagicMock]:
    plugin = OpenAI(api_key='test-key')
    client = MagicMock()
    client.images.generate = AsyncMock(return_value=MagicMock(data=[]))
    plugin._runtime_client = lambda: client
    return Genkit(plugins=[plugin]), client


class TestExtractPromptText:
    """Tests for extracting text from ModelRequest messages."""

    def test_extracts_text_from_first_message(self) -> None:
        """Verify text extraction from a simple single-message request."""
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part.from_text('a sunset')]),
            ],
        )
        got = _extract_prompt_text(request)
        assert got == 'a sunset'

    def test_raises_on_empty_messages(self) -> None:
        """Verify ValueError when messages list is empty."""
        request = ModelRequest(messages=[])
        with pytest.raises(ValueError, match='No messages found'):
            _extract_prompt_text(request)

    def test_raises_on_no_text_content(self) -> None:
        """Verify ValueError when message has no text parts."""
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[]),
            ],
        )
        with pytest.raises(ValueError, match='No text content found'):
            _extract_prompt_text(request)


class TestToImageGenerateParams:
    """Tests for converting ModelRequest to OpenAI image params."""

    def test_basic_params(self) -> None:
        """Verify required params are set with correct defaults."""
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part.from_text('a cat')]),
            ],
        )
        got = _to_image_generate_params('dall-e-3', request)
        assert got['model'] == 'dall-e-3'
        assert got['prompt'] == 'a cat'
        assert got['response_format'] == 'b64_json'

    def test_gpt_image_1_omits_response_format(self) -> None:
        """Verify GPT Image 1 requests omit the unsupported response format."""
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part.from_text('a cat')]),
            ],
            config={'response_format': 'url'},
        )

        got = _to_image_generate_params('gpt-image-1', request)

        assert 'response_format' not in got

    @pytest.mark.parametrize('model_name', ['GPT-IMAGE-1', 'gpt-image-1-mini', 'gpt-image-1.5', 'chatgpt-image-latest'])
    def test_gpt_image_variants_omit_response_format(self, model_name: str) -> None:
        """Verify GPT Image variants use the API's base64-only response shape."""
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part.from_text('a cat')]),
            ],
            config={'response_format': 'url'},
        )

        got = _to_image_generate_params(model_name, request)

        assert 'response_format' not in got

    def test_gpt_image_config_passthrough(self) -> None:
        """Verify GPT Image-specific options are forwarded unchanged."""
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part.from_text('a cat')]),
            ],
            config={
                'background': 'transparent',
                'moderation': 'low',
                'output_compression': 80,
                'output_format': 'png',
                'size': '1024x1024',
                'quality': 'high',
            },
        )

        got = _to_image_generate_params('gpt-image-1', request)

        assert got['background'] == 'transparent'
        assert got['moderation'] == 'low'
        assert got['output_compression'] == 80
        assert got['output_format'] == 'png'
        assert got['size'] == '1024x1024'
        assert got['quality'] == 'high'

    def test_config_passthrough(self) -> None:
        """Verify image-specific config options pass through."""
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part.from_text('a dog')]),
            ],
            config={'size': '1024x1024', 'quality': 'hd', 'n': 2},
        )
        got = _to_image_generate_params('dall-e-3', request)
        assert got.get('size') == '1024x1024'
        assert got.get('quality') == 'hd'
        assert got.get('n') == 2

    @pytest.mark.parametrize('config_cls', [OpenAIDalleConfig, OpenAIGptImageConfig])
    @pytest.mark.parametrize(
        'key', ['temperature', 'max_output_tokens', 'maxOutputTokens', 'top_k', 'top_p', 'stop_sequences', 'version']
    )
    def test_image_config_rejects_generic_knob(self, config_cls: type[BaseModel], key: str) -> None:
        """The Images API takes none of the text-generation knobs, so the config rejects them by name."""
        with pytest.raises(ValidationError, match=key):
            config_cls.model_validate({key: 1})


class TestToModelResponse:
    """Tests for converting OpenAI ImagesResponse to ModelResponse."""

    def test_empty_data(self) -> None:
        """Verify empty image data produces empty content."""
        mock_result = MagicMock()
        mock_result.data = []
        got = _to_generate_response(mock_result)
        assert got.message is not None
        assert len(got.message.content) == 0

    def test_url_response(self) -> None:
        """Verify URL-based image response is preserved."""
        mock_image = MagicMock()
        mock_image.url = 'https://example.com/image.png'
        mock_image.b64_json = None
        mock_result = MagicMock()
        mock_result.data = [mock_image]

        got = _to_generate_response(mock_result)
        assert got.message is not None
        assert len(got.message.content) == 1

        part = got.message.content[0]
        assert part.media is not None
        assert str(part.media.url) == 'https://example.com/image.png'

    def test_b64_response(self) -> None:
        """Verify base64-encoded image is wrapped in a data URI."""
        mock_image = MagicMock()
        mock_image.url = None
        mock_image.b64_json = 'aGVsbG8='
        mock_result = MagicMock()
        mock_result.data = [mock_image]

        got = _to_generate_response(mock_result)
        assert got.message is not None
        part = got.message.content[0]
        assert part.media is not None
        assert str(part.media.url) == 'data:image/png;base64,aGVsbG8='

    def test_multiple_images(self) -> None:
        """Verify multiple images produce multiple content parts."""
        images = []
        for i in range(3):
            img = MagicMock()
            img.url = f'https://example.com/{i}.png'
            img.b64_json = None
            images.append(img)
        mock_result = MagicMock()
        mock_result.data = images

        got = _to_generate_response(mock_result)
        assert got.message is not None
        assert len(got.message.content) == 3


class TestSupportedImageModels:
    """Tests that the model info registry is correct."""

    def test_registry_lists_current_gpt_image_models(self) -> None:
        """The registry holds the served GPT Image models, not the retired dall-e-3 or gpt-image-1."""
        assert set(SUPPORTED_IMAGE_MODELS) == {
            'gpt-image-1-mini',
            'gpt-image-1.5',
            'chatgpt-image-latest',
            'gpt-image-2',
            'gpt-image-2.5-sunburst',
            'gpt-image-2.5-flare',
            'gpt-image-2.6-sunburst',
            'gpt-image-2.6-flare',
        }

    @pytest.mark.parametrize('name', ['gpt-image-1-mini', 'gpt-image-1.5', 'chatgpt-image-latest', 'gpt-image-2'])
    def test_models_with_announced_shutdown_are_deprecated(self, name: str) -> None:
        """Ids with an announced shutdown date carry stage=deprecated."""
        assert SUPPORTED_IMAGE_MODELS[name].stage == ModelStage.DEPRECATED

    def test_current_gpt_image_models_have_no_stage(self) -> None:
        """The 2.5 and 2.6 models have no announced shutdown."""
        for name in ('gpt-image-2.5-sunburst', 'gpt-image-2.5-flare', 'gpt-image-2.6-sunburst', 'gpt-image-2.6-flare'):
            assert SUPPORTED_IMAGE_MODELS[name].stage is None

    def test_image_models_support_media_output(self) -> None:
        """Verify all image models declare 'media' output support."""
        for name, info in SUPPORTED_IMAGE_MODELS.items():
            assert info.supports is not None, f'{name} has no supports metadata'
            assert 'media' in (info.supports.output or []), f"{name} should support 'media' output"

    def test_gpt_image_exposes_config_schema(self) -> None:
        """A current GPT Image model advertises the constraints of its config class."""
        plugin = OpenAI(api_key='test-key')
        action = plugin._create_multimodal_action(
            'gpt-image-2.5-flare', OpenAIImageModel, SUPPORTED_IMAGE_MODELS, _ModelType.IMAGE
        )
        model_meta = cast(dict[str, Any], action.metadata['model'])
        schema = model_meta['customOptions']
        properties = schema['properties']

        def _enum(name: str) -> list[str]:
            return properties[name]['anyOf'][0]['enum']

        def _int_bounds(name: str) -> dict[str, Any]:
            return properties[name]['anyOf'][0]

        assert _enum('size') == ['1024x1024', '1536x1024', '1024x1536', 'auto']
        assert _enum('quality') == ['low', 'medium', 'high', 'xhigh', 'max', 'auto']
        assert _enum('background') == ['transparent', 'opaque', 'auto']
        assert _enum('moderation') == ['low', 'auto']
        assert _int_bounds('n') == {'type': 'integer', 'minimum': 1, 'maximum': 10}
        assert _int_bounds('outputCompression') == {'type': 'integer', 'minimum': 1, 'maximum': 100}
        assert _enum('outputFormat') == ['png', 'jpeg', 'webp']
        assert 'responseFormat' not in properties
        assert 'response_format' not in properties

    @pytest.mark.asyncio
    async def test_gpt_image_response_format_raises_before_model_call(self) -> None:
        """`response_format` on a GPT Image model raises INVALID_ARGUMENT before the endpoint runs."""
        ai = Genkit(plugins=[OpenAI(api_key='test-key')])

        with patch.object(OpenAIImageModel, 'generate', new_callable=AsyncMock) as generate:
            with pytest.raises(GenkitError) as err:
                await ai.generate(
                    model='openai/gpt-image-2.5-flare',
                    prompt='a lighthouse at dusk',
                    config={'response_format': 'b64_json'},
                )

        assert err.value.status == 'INVALID_ARGUMENT'
        assert 'response_format' in str(err.value)
        generate.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('name', 'quality'),
        [
            *[(name, _MAX_QUALITY) for name in _MAX_QUALITY_MODELS],
            *[(name, _STANDARD_QUALITY) for name in _STANDARD_QUALITY_MODELS],
        ],
    )
    async def test_gpt_image_advertises_the_quality_values_its_model_takes(self, name: str, quality: list[str]) -> None:
        """Only the 2.5 and 2.6 Sunburst and Flare models advertise xhigh and max quality."""
        action = await OpenAI(api_key='test-key').resolve(ActionKind.MODEL, name)

        assert action is not None
        schema = cast(dict[str, Any], action.metadata['model'])['customOptions']
        assert schema['title'] == 'OpenAIGptImageConfig'
        assert schema['properties']['quality']['enum'] == quality
        assert schema['properties']['quality']['anyOf'][0]['enum'] == quality

    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', _STANDARD_QUALITY_MODELS)
    @pytest.mark.parametrize('quality', ['xhigh', 'max'])
    @pytest.mark.parametrize('typed', [False, True], ids=['dict', 'typed'])
    async def test_gpt_image_without_max_quality_fails_before_the_api_call(
        self, name: str, quality: _Quality, typed: bool
    ) -> None:
        """xhigh and max fail with INVALID_ARGUMENT naming the field and model, and never reach the Images API."""
        ai, client = _genkit_with_image_client()
        config = OpenAIGptImageConfig(quality=quality) if typed else {'quality': quality}

        response = await ai.generate(model=f'openai/{name}', prompt='a lighthouse at dusk', config=config)

        assert response.finish_reason == FinishReason.FAILED
        assert response.error is not None
        assert response.error.status == 'INVALID_ARGUMENT'
        assert f"quality '{quality}'" in response.error.message
        assert name in response.error.message
        client.images.generate.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        ('name', 'quality'),
        [
            *[(name, quality) for name in _MAX_QUALITY_MODELS for quality in ('xhigh', 'max')],
            ('gpt-image-2', 'auto'),
            ('chatgpt-image-latest', 'high'),
        ],
    )
    @pytest.mark.parametrize('typed', [False, True], ids=['dict', 'typed'])
    async def test_gpt_image_quality_the_model_takes_reaches_the_endpoint(
        self, name: str, quality: _Quality, typed: bool
    ) -> None:
        """A quality value the model takes reaches the Images API, as a dict or an OpenAIGptImageConfig."""
        ai, client = _genkit_with_image_client()
        config = OpenAIGptImageConfig(quality=quality) if typed else {'quality': quality}

        await ai.generate(model=f'openai/{name}', prompt='a lighthouse at dusk', config=config)

        assert client.images.generate.call_args.kwargs['quality'] == quality

    @pytest.mark.asyncio
    @pytest.mark.parametrize('name', _MAX_QUALITY_MODELS + _STANDARD_QUALITY_MODELS)
    async def test_gpt_image_config_with_standard_quality_works_on_every_gpt_image_model(self, name: str) -> None:
        """An OpenAIGptImageConfig instance is the config class of every GPT Image model."""
        ai, client = _genkit_with_image_client()

        await ai.generate(
            model=f'openai/{name}', prompt='a lighthouse at dusk', config=OpenAIGptImageConfig(quality='high')
        )

        client.images.generate.assert_called_once()


class TestOpenAIImageModel:
    """Tests for the OpenAIImageModel class."""

    @pytest.mark.asyncio
    async def test_generate_calls_client(self) -> None:
        """Verify generate() calls client.images.generate and returns media."""
        mock_image = MagicMock()
        mock_image.url = 'https://example.com/generated.png'
        mock_image.b64_json = None
        mock_response = MagicMock()
        mock_response.data = [mock_image]

        mock_client = AsyncMock()
        mock_client.images.generate = AsyncMock(return_value=mock_response)

        model = OpenAIImageModel('dall-e-3', mock_client)
        request = ModelRequest(
            messages=[
                Message(role=Role.USER, content=[Part.from_text('a mountain')]),
            ],
        )

        ctx = MagicMock()
        got = await model.generate(request, ctx)

        mock_client.images.generate.assert_called_once()
        call_kwargs = mock_client.images.generate.call_args
        assert call_kwargs.kwargs.get('model') == 'dall-e-3'
        assert call_kwargs.kwargs.get('prompt') == 'a mountain'

        assert got.message is not None
        assert len(got.message.content) == 1
