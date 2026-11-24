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

"""The Anthropic API as a backend: API keys, per-request keys, and the Models API."""

import os
from collections.abc import Mapping
from typing import Any
from urllib.parse import urlsplit

from anthropic import AsyncAnthropic

from genkit import GenkitError
from genkit_anthropic._backend import Backend, client_kwargs
from genkit_anthropic._models import TASK_BUDGET_BETA
from genkit_anthropic._secrets import missing_key_error

# Sent on every beta-surface call to the Anthropic API, matching the JS plugin.
BETA_APIS: tuple[str, ...] = (
    'files-api-2025-04-14',
    'effort-2025-11-24',
    'structured-outputs-2025-11-13',
    TASK_BUDGET_BETA,
)


class AnthropicApiBackend(Backend[AsyncAnthropic]):
    """Claude on the Anthropic API, billed to an Anthropic key."""

    default_betas = BETA_APIS

    def __init__(self, *, api_key: str | None, base_url: str | None, client_options: Mapping[str, Any] | None) -> None:
        """Validate the client settings at setup, on the user's line.

        Raises:
            TypeError: An unknown or repeated ``client_options`` key.
            ValueError: The base URL ends in ``/v1``.
        """
        self._options = client_kwargs(AsyncAnthropic, client_options, api_key=api_key, base_url=base_url)
        if 'base_url' in self._options:
            _reject_v1_suffix(self._options['base_url'], source='base_url')
        else:
            _reject_v1_suffix(os.environ.get('ANTHROPIC_BASE_URL'), source='ANTHROPIC_BASE_URL')

    def _new_client(self) -> AsyncAnthropic:
        return AsyncAnthropic(**self._options)

    async def request_client(self, api_key: str | None) -> AsyncAnthropic:
        """The plugin's client, or a copy on the caller's key.

        A client that can't swap its credential raises rather than sending the
        call on the plugin's own key, because whoever passed a key expects
        that account to be billed.
        """
        client = self.client()
        if api_key is None:
            # A plugin built without a credential serves only callers who bring a key.
            if not _has_credential(client):
                raise missing_key_error()
            return client

        reason = None
        # copy() cannot unset these, so the override would leave the base credential authenticating the request.
        if client.auth_token is not None:
            reason = 'the plugin client authenticates with an auth token'
        elif (header := _pinned_credential_header(client)) is not None:
            reason = f'the plugin client sets its own {header} header'
        if reason is not None:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=f'A per-request API key from context.secrets cannot be used: {reason}.',
            )
        # copy() keeps every other client setting and shares the pooled HTTP transport.
        return client.copy(api_key=api_key)

    async def list_model_ids(self) -> list[str]:
        """Every model id the Models API lists, in API order.

        Uses the beta models endpoint (matching JS, so both stable and beta
        models are discovered) and fully paginates via ``async for`` (matching
        Go's completeness, rather than JS's first-page-only read).
        """
        return [model.id async for model in self.client().beta.models.list() if model.id]


def _reject_v1_suffix(base_url: str | None, *, source: str) -> None:
    """The SDK appends ``/v1/messages``, so a base URL ending in ``/v1`` would send ``/v1/v1/messages``."""
    if base_url and urlsplit(base_url).path.rstrip('/').endswith('/v1'):
        raise ValueError(
            f'{source} {base_url!r} ends in /v1, but the Anthropic SDK appends /v1/messages itself; '
            'pass the origin, e.g. https://claude-proxy.example.com'
        )


def _pinned_credential_header(client: AsyncAnthropic) -> str | None:
    """The credential header ``client`` pins in ``default_headers``, if any.

    ``copy(api_key=...)`` keeps custom headers, so a pinned ``x-api-key`` or
    ``Authorization`` would still go out next to a tenant key and could
    authenticate the call.
    """
    for name in client._custom_headers:  # noqa: SLF001
        if name.lower() in ('x-api-key', 'authorization'):
            return name
    return None


def _has_credential(client: AsyncAnthropic) -> bool:
    if client.api_key is not None or client.auth_token is not None or client.credentials is not None:
        return True
    return _pinned_credential_header(client) is not None
