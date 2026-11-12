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

"""Per-request API key on ``context.secrets``.

The ticket and the generate span are not where a tenant key lives. Callers
pass it again on the run: ``context={'secrets': {'api_key': tenant}}``.
"""

from typing import Any, cast

from genkit import GenkitError

SECRETS_SLOT = "Pass the key as context={'secrets': {'api_key': ...}}."

# One line: GoogleAI() logs it at construction and a keyless call raises it, so
# it has to read cleanly both as a log line and inside str()/repr() of an error.
MISSING_API_KEY_MESSAGE = (
    'No Gemini API key found. Set GEMINI_API_KEY (or GOOGLE_API_KEY), pass GoogleAI(api_key=...), '
    "or send context={'secrets': {'api_key': ...}} per request. "
    'Get a key: https://aistudio.google.com/apikey'
)


def missing_api_key_error() -> GenkitError:
    """No plugin key, no credentials, and no ``context.secrets.api_key``.

    ``FAILED_PRECONDITION`` matches JS ``MISSING_API_KEY_ERROR``. A key the
    API rejects is a different failure: the provider's 4xx comes back as
    ``UNAUTHENTICATED`` with reason ``INVALID_CREDENTIALS`` (see ``_provider_errors``).
    """
    return GenkitError(status='FAILED_PRECONDITION', message=MISSING_API_KEY_MESSAGE)


def misplaced_key_error() -> GenkitError:
    return GenkitError(
        status='INVALID_ARGUMENT',
        message=f'API key belongs in context.secrets, not config or the top-level context. {SECRETS_SLOT}',
    )


_KEY_SLOTS = ('api_key', 'apiKey')


def string_secret(value: object, slot: str = 'api_key') -> str:
    # An explicit None or blank key is a tenant that meant to bring one and
    # didn't (request.headers.get('x-gemini-key') returning None, say). Raise
    # instead of falling back, which would bill the plugin key.
    if not isinstance(value, str):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.{slot} must be a string. {SECRETS_SLOT}',
        )
    cleaned = value.strip()
    if not cleaned:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.{slot} is blank. {SECRETS_SLOT}',
        )
    if any(c in cleaned for c in ('\r', '\n', '\0', ' ', '\t')):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets.{slot} contains invalid whitespace or control characters. {SECRETS_SLOT}',
        )
    return cleaned


def context_api_key(context: dict[str, Any]) -> str | None:
    """Read the per-request key from ``context.secrets``.

    Documented slot is ``api_key``; ``apiKey`` is accepted so either
    spelling works. A key on ``context['config']`` or the top-level
    context (even ``None``) is rejected so an explicit override in the
    wrong place cannot silently fall through to the plugin client.
    ``generate(..., config={'api_key': ...})`` is the same leftover —
    reject it on the request too.

    Returns ``None`` only when ``secrets`` omits ``api_key`` / ``apiKey``
    (e.g. holding other app secrets); the call then falls back to the
    plugin key, as in genkit-openai and genkit-anthropic. An explicit
    ``None`` or blank key raises instead of billing the plugin key.
    """
    extra = context.get('config')
    if isinstance(extra, dict) and any(slot in extra for slot in _KEY_SLOTS):
        raise misplaced_key_error()
    if any(slot in context for slot in _KEY_SLOTS):
        raise misplaced_key_error()

    if 'secrets' not in context:
        return None
    secrets = context['secrets']
    if not isinstance(secrets, dict):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'context.secrets must be a dict. {SECRETS_SLOT}',
        )
    for slot in _KEY_SLOTS:
        if slot in secrets:
            return string_secret(secrets[slot], slot)
    return None


def _bag_has_api_key(bag: dict[str, Any]) -> bool:
    return bag.get('api_key') is not None or bag.get('apiKey') is not None


def request_config_has_api_key(config: object) -> bool:
    """True when the generate/start request still carries a tenant key."""
    if config is None:
        return False
    if isinstance(config, dict):
        return _bag_has_api_key(cast(dict[str, Any], config))
    extra = getattr(config, 'model_extra', None)
    if isinstance(extra, dict) and _bag_has_api_key(cast(dict[str, Any], extra)):
        return True
    # config.extra is sent on the wire and lands in traces; a key there is the same leak.
    passthrough = getattr(config, 'extra', None)
    if isinstance(passthrough, dict) and _bag_has_api_key(cast(dict[str, Any], passthrough)):
        return True
    if getattr(config, 'api_key', None) is not None or getattr(config, 'apiKey', None) is not None:
        return True
    dump = getattr(config, 'model_dump', None)
    if callable(dump):
        try:
            dumped = dump(by_alias=True)
        except TypeError:
            dumped = dump()
        return isinstance(dumped, dict) and _bag_has_api_key(cast(dict[str, Any], dumped))
    return False


def reject_request_config_api_key(config: object) -> None:
    """A key on request.config belongs in context.secrets."""
    if request_config_has_api_key(config):
        raise misplaced_key_error()
