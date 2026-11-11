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

"""Raw HTTP helpers for the Google AI Interactions API."""

from __future__ import annotations

import json
from collections.abc import Mapping
from urllib.parse import quote

import httpx
from genkit_google_genai._interactions._errors import api_error_fields, interaction_error_status, unwrap_error_body
from genkit_google_genai._interactions._options import ClientOptions
from genkit_google_genai._interactions._types import GeminiInteraction, parse_interaction
from genkit_google_genai._provider_errors import TRANSPORT_ERRORS, transport_error
from pydantic import ValidationError

from genkit import GenkitError
from genkit.plugin_api import GENKIT_USER_AGENT, StatusName, loop_local_client, provider_error
from genkit.telemetry import SpanContext, run_in_new_span

DEFAULT_API_VERSION = 'v1beta'
DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com'
API_REVISION = '2026-05-20'
# Creates can run for many minutes. No read timeout, but keep a connect
# budget so a hung handshake doesn't sit forever.
NO_TIMEOUT = httpx.Timeout(None, connect=10.0)
RESERVED_HEADERS = ('x-goog-api-key', 'x-goog-api-client')


@loop_local_client
def _http_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(timeout=NO_TIMEOUT)


def google_ai_url(
    resource_path: str,
    *,
    client_options: ClientOptions | None = None,
) -> str:
    """Build a Google AI REST URL for the given resource path."""
    opts = client_options or ClientOptions()
    api_version = opts.api_version or DEFAULT_API_VERSION
    base_url = (opts.base_url or DEFAULT_BASE_URL).rstrip('/')
    return f'{base_url}/{api_version}/{resource_path}'


def headers(*, api_key: str, client_options: ClientOptions | None) -> dict[str, str]:
    """Build request headers; api key and Genkit client attribution win over custom.

    Header names are matched case-insensitively so ``X-Goog-Api-Key`` cannot
    sneak a second key onto the request.
    """
    custom = httpx.Headers((client_options or ClientOptions()).custom_headers or {})
    for name in RESERVED_HEADERS:
        custom.pop(name, None)
    custom['Content-Type'] = 'application/json'
    custom['x-goog-api-client'] = GENKIT_USER_AGENT
    custom['Api-Revision'] = API_REVISION
    custom['x-goog-api-key'] = api_key
    return dict(custom)


def timeout_seconds(client_options: ClientOptions | None) -> float | None:
    """Convert ClientOptions.timeout (milliseconds) to httpx seconds."""
    opts = client_options or ClientOptions()
    if opts.timeout is not None and opts.timeout >= 0:
        return opts.timeout / 1000.0
    return None


async def create_interaction(
    api_key: str,
    body: dict[str, object],
    client_options: ClientOptions | None = None,
) -> GeminiInteraction:
    """POST /interactions and return the parsed Interaction."""
    url = google_ai_url('interactions', client_options=client_options)
    created = await request(
        method='POST',
        url=url,
        api_key=api_key,
        client_options=client_options,
        json_body=body,
    )
    assert created is not None
    return created


async def get_interaction(
    api_key: str,
    interaction_id: str,
    client_options: ClientOptions | None = None,
) -> GeminiInteraction:
    """GET /interactions/{id} and return the parsed Interaction."""
    url = google_ai_url(f'interactions/{quote(interaction_id, safe="")}', client_options=client_options)
    found = await request(
        method='GET',
        url=url,
        api_key=api_key,
        client_options=client_options,
    )
    assert found is not None
    return found


async def cancel_interaction(
    api_key: str,
    interaction_id: str,
    client_options: ClientOptions | None = None,
) -> GeminiInteraction:
    """POST /interactions/{id}/cancel and return a cancelled Interaction."""
    url = google_ai_url(f'interactions/{quote(interaction_id, safe="")}/cancel', client_options=client_options)
    try:
        interaction = await request(
            method='POST',
            url=url,
            api_key=api_key,
            client_options=client_options,
            allow_empty=True,
            # Not traced, as in JS (cancelInteraction calls makeRequest directly).
            trace=False,
        )
    except GenkitError as error:
        if error.status == 'CANCELLED':
            return {'id': interaction_id, 'status': 'cancelled'}
        raise
    if interaction is None:
        return {'id': interaction_id, 'status': 'cancelled'}
    return {**interaction, 'status': 'cancelled'}


def redact(value: str) -> str:
    """``<REDACTED> (N characters)``, as in JS buildTraceMetadataInput."""
    return f'<REDACTED> ({len(value)} characters)'


def trace_input(
    url: str,
    request_headers: Mapping[str, str],
    json_body: dict[str, object] | None,
    client_options: ClientOptions | None,
) -> dict[str, object]:
    """The ``httpRequest`` span input, the same shape as JS buildTraceMetadataInput.

    The API key header is hidden. Interactions requests carry the model in
    the body, so there's no separate ``model`` field (JS adds one only for
    generateContent).
    """
    safe_headers = {
        name: redact(value) if name.lower() in ('x-goog-api-key', 'authorization') else value
        for name, value in request_headers.items()
    }
    traced: dict[str, object] = {'apiEndpoint': url, 'request': json_body, 'headers': safe_headers}
    timeout = client_options.timeout if client_options else None
    if timeout:
        traced['options'] = {'timeout': timeout}
    return traced


async def request(
    *,
    method: str,
    url: str,
    api_key: str,
    client_options: ClientOptions | None,
    json_body: dict[str, object] | None = None,
    allow_empty: bool = False,
    trace: bool = True,
) -> GeminiInteraction | None:
    """Issue one Interactions HTTP call and parse the Interaction body.

    With ``experimental_debug_traces`` on the client options (and ``trace``),
    the call runs in an ``httpRequest`` span whose input is the endpoint,
    the request body and the headers (API key hidden), and whose output is
    the response body, as in JS.
    """
    # Auth/key headers are per-request; the client is just the transport.
    request_headers = headers(api_key=api_key, client_options=client_options)

    async def call(span: SpanContext | None = None) -> object:
        return await send(
            method=method,
            url=url,
            request_headers=request_headers,
            client_options=client_options,
            json_body=json_body,
            allow_empty=allow_empty,
        )

    if trace and client_options is not None and client_options.experimental_debug_traces:
        # The span output is the response body as received, like JS.
        body = await run_in_new_span(
            'httpRequest',
            call,
            input=trace_input(url, request_headers, json_body, client_options),
        )
    else:
        body = await call()
    if body is None:
        return None
    try:
        return parse_interaction(body)
    except ValidationError as error:
        raise GenkitError(
            status='INTERNAL',
            message=f'Unable to parse Interaction response from {url}: {error}',
        ) from error


async def send(
    *,
    method: str,
    url: str,
    request_headers: dict[str, str],
    client_options: ClientOptions | None,
    json_body: dict[str, object] | None = None,
    allow_empty: bool = False,
) -> object:
    """Send one Interactions HTTP call; return the JSON body (None for an allowed empty one) or raise."""
    # The timeout always goes on the request so a caller's own client gets the
    # same budget as the default one instead of httpx's 5 second default.
    http_client_factory = client_options.http_client_factory if client_options else None
    client = http_client_factory() if http_client_factory is not None else _http_client()
    timeout = timeout_seconds(client_options)

    try:
        response = await client.request(
            method,
            url,
            headers=request_headers,
            json=json_body,
            timeout=timeout if timeout is not None else NO_TIMEOUT,
        )
    except TRANSPORT_ERRORS as error:
        raise transport_error(error) from error

    if response.is_success:
        if not response.content:
            if allow_empty:
                return None
            raise GenkitError(
                status='INTERNAL',
                message=f'Received an empty response from {url}',
            )
        try:
            body: object = response.json()
        except json.JSONDecodeError as error:
            raise GenkitError(
                status='INTERNAL',
                message=f'Unable to parse Interaction response from {url}: {error}',
            ) from error
        return body

    error_message = response.text
    error_detail: object | None = None
    # An Interactions error code (e.g. `quota_exceeded`, `model_not_found`)
    # is more precise than the HTTP status, so it wins when it's known.
    code_status: StatusName | None = None
    try:
        # The Interactions API sometimes wraps the error in a one-element
        # array (e.g. a bad API key).
        payload: object = unwrap_error_body(response.json())
        error_detail = payload
        api_message, api_code = api_error_fields(payload)
        if api_message is not None:
            error_message = api_message
        code_status = interaction_error_status(api_code)
    except json.JSONDecodeError:
        pass

    message = f'Request to {url} failed with HTTP {response.status_code} {response.reason_phrase}: {error_message}'
    cause = httpx.HTTPStatusError(message, request=httpx.Request(method, url), response=response)
    error = provider_error(
        cause, http_status=response.status_code, status=code_status, headers=response.headers, message=message
    )
    if isinstance(error_detail, Mapping):
        error.details.update(error_detail)
    raise error from cause
