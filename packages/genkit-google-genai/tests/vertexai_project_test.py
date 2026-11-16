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

"""VertexAI client construction without a resolvable project or credentials."""

import os
from unittest.mock import MagicMock, patch

import pytest
from genkit_google_genai import VertexAI
from genkit_google_genai._google import _plugin_client
from genkit_google_genai._models._gemini import GeminiModel
from google.auth.credentials import AnonymousCredentials
from google.auth.exceptions import DefaultCredentialsError
from google.genai import types as genai_types

from genkit import Genkit, GenkitError, Message, Part, Role
from genkit.model import ModelRequest


def test_vertex_without_project_is_failed_precondition() -> None:
    """google-genai's bare ValueError becomes FAILED_PRECONDITION naming VertexAI(project=...)."""
    kwargs = {
        'vertexai': True,
        'api_key': None,
        'credentials': None,
        'project': None,
        'location': 'us-central1',
        'http_options': genai_types.HttpOptions(),
    }
    with (
        patch(
            'genkit_google_genai._google.genai.client.Client', side_effect=ValueError('Project or API key must be set')
        ),
        pytest.raises(GenkitError) as raised,
    ):
        _plugin_client(kwargs)

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert 'VertexAI(project=...)' in raised.value.message
    assert 'GOOGLE_CLOUD_PROJECT' in raised.value.message


def test_other_client_value_errors_are_not_rewritten() -> None:
    """A ValueError unrelated to a missing Vertex project passes through unchanged."""
    kwargs = {'vertexai': True, 'api_key': None, 'project': 'menu-prod', 'location': 'us-central1'}
    with (
        patch('genkit_google_genai._google.genai.client.Client', side_effect=ValueError('bad location')),
        pytest.raises(ValueError, match='bad location'),
    ):
        _plugin_client(kwargs)


@pytest.mark.asyncio
@pytest.mark.parametrize('location', [None, 'us'], ids=['regional', 'multi-region'])
async def test_vertex_without_project_fails_at_call_time_not_init(location: str | None) -> None:
    """With ADC credentials but no project, VertexAI() and VertexAI(location='us') fail FAILED_PRECONDITION."""
    creds = (AnonymousCredentials(), None)
    with (
        patch.dict(os.environ, {}, clear=True),
        patch('google.auth.default', return_value=creds),
        patch('genkit_google_genai._google.google_auth_default', return_value=creds),
    ):
        plugin = VertexAI(location=location) if location else VertexAI()
        ai = Genkit(plugins=[plugin])
        response = await ai.generate(model='vertexai/gemini-2.5-flash', prompt='Suggest a dish.')

    # init() registers a static catalog without listing, so the failure comes
    # from the model call and generate returns it on response.error.
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert 'VertexAI(project=...)' in response.error.message


@pytest.mark.asyncio
async def test_vertex_without_credentials_fails_unauthenticated_at_call_time_not_init() -> None:
    """When google.auth.default finds no ADC, VertexAI() constructs cleanly and the call fails UNAUTHENTICATED."""
    no_adc = DefaultCredentialsError('Could not automatically determine credentials.')
    with (
        patch.dict(os.environ, {}, clear=True),
        patch('google.auth.default', side_effect=no_adc),
        patch('genkit_google_genai._google.google_auth_default', side_effect=no_adc),
    ):
        ai = Genkit(plugins=[VertexAI()])
        response = await ai.generate(model='vertexai/gemini-2.5-flash', prompt='Suggest a dish.')

    assert response.error is not None
    assert response.error.status == 'UNAUTHENTICATED'


@pytest.mark.asyncio
async def test_gemini_resolve_request_client_missing_project_is_failed_precondition() -> None:
    """GeminiModel._resolve_request_client maps a missing Vertex project ValueError to FAILED_PRECONDITION."""
    client = MagicMock()
    client.vertexai = True
    model = GeminiModel(
        'gemini-2.5-flash',
        client,
        client_kwargs={'vertexai': True, 'api_key': None, 'project': None, 'location': 'us-central1'},
    )
    with (
        patch('genkit_google_genai._models._gemini._adc_project', return_value=None),
        patch(
            'genkit_google_genai._models._gemini.genai.Client',
            side_effect=ValueError('Project or API key must be set'),
        ),
        pytest.raises(GenkitError) as raised,
    ):
        await model._resolve_request_client(
            ModelRequest(
                messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
                config={'apiVersion': 'v1'},
            )
        )

    assert raised.value.status == 'FAILED_PRECONDITION'
    assert 'VertexAI(project=...)' in raised.value.message
