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

"""Tests for the backend seam: what each backend owns, checked without a plugin around it."""

import pytest
from anthropic import AsyncAnthropic, AsyncAnthropicVertex
from genkit_anthropic._anthropic_api import BETA_APIS, AnthropicApiBackend
from genkit_anthropic._backend import client_kwargs


@pytest.mark.asyncio
async def test_anthropic_api_secrets_key_copy_keeps_client_settings_and_transport() -> None:
    """A per-request key yields a request-scoped copy that keeps client settings and transport."""
    backend = AnthropicApiBackend(
        api_key='base-key', base_url=None, client_options={'default_headers': {'X-Custom': 'yes'}}
    )
    base_client = backend.client()

    client = await backend.request_client('request-key')

    assert client is not base_client
    assert client.api_key == 'request-key'
    assert client.default_headers.get('X-Custom') == 'yes'
    assert client._client is base_client._client
    assert await backend.request_client(None) is base_client


def test_anthropic_api_offers_the_default_betas() -> None:
    """Only the Anthropic API backend sends the default beta headers."""
    assert AnthropicApiBackend.default_betas == BETA_APIS


def test_client_kwargs_names_the_client_it_validated_against() -> None:
    """The allowlist and the error both come from the client class, so each backend reports its own client."""
    with pytest.raises(TypeError, match='AsyncAnthropic accepts'):
        client_kwargs(AsyncAnthropic, {'region': 'us-east5'})
    with pytest.raises(TypeError, match='AsyncAnthropicVertex accepts'):
        client_kwargs(AsyncAnthropicVertex, {'region': 'us-east5'}, reserved=frozenset({'region'}))


def test_client_kwargs_rejects_a_setting_passed_twice() -> None:
    """A typed setting and the same client_options key is ambiguous, so it raises naming the setting."""
    with pytest.raises(TypeError, match='base_url both as an argument and in client_options'):
        client_kwargs(AsyncAnthropic, {'base_url': 'https://a'}, base_url='https://b')

    assert client_kwargs(AsyncAnthropic, {'timeout': 5}, base_url=None) == {'timeout': 5}
