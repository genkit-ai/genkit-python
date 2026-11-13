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

"""Utility functions for Google GenAI context caching."""

import hashlib
import json
from collections.abc import Mapping
from typing import Any, cast

from genkit_google_genai._models._context_caching._constants import INVALID_ARGUMENT_MESSAGES
from google.genai import types as genai_types

from genkit import GenkitError, get_logger
from genkit.model import ModelRequest

logger = get_logger(__name__)


def generate_cache_key(contents: list[genai_types.Content], model_name: str) -> str:
    """Generates a cache key by hashing the cached prefix contents and model name.

    Only the prefix slice (messages up to and including the cache marker) is
    hashed, so two requests with the same cached prefix but different trailing
    messages correctly reuse the same cache entry.

    Args:
        contents: The prefix content objects to be cached.
        model_name: Name of the model — included in the key to prevent
            cross-model cache collisions when multiple models are used
            in the same session.

    Returns:
        Generated cache key string
    """
    serialized = [c.model_dump() for c in contents]
    payload = {'model': model_name, 'contents': serialized}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _config_flag(config: object, snake: str, camel: str) -> Any:  # noqa: ANN401
    if isinstance(config, Mapping):
        mapping = cast(Mapping[str, Any], config)
        if snake in mapping:
            return mapping[snake]
        if camel in mapping:
            return mapping[camel]
        return None
    return getattr(config, snake, None)


def _has_nonempty_items(value: object) -> bool:
    if not isinstance(value, list):
        return bool(value)
    return any(bool(item) for item in cast(list[object], value))


_GOOGLE_SEARCH_OPTION_PAIRS: tuple[tuple[str, str], ...] = (
    ('google_search_types', 'googleSearchTypes'),
    ('google_search_exclude_domains', 'googleSearchExcludeDomains'),
    ('google_search_blocking_confidence', 'googleSearchBlockingConfidence'),
    ('google_search_start_time', 'googleSearchStartTime'),
    ('google_search_end_time', 'googleSearchEndTime'),
)


def validate_context_cache_request(request: ModelRequest, model_name: str) -> bool:
    """Verifies that the context cache request could be processed for the request.

    Args:
        request: `ModelRequest` instance to check
        model_name: Name of the generation model to check

    Returns:
        True if the context cache request could be processed for the request
    """
    if not model_name:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=INVALID_ARGUMENT_MESSAGES['modelVersion'],
        )
    if request.tools:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=INVALID_ARGUMENT_MESSAGES['tools'],
        )
    config = request.config
    if _config_flag(config, 'code_execution', 'codeExecution'):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=INVALID_ARGUMENT_MESSAGES['codeExecution'],
        )
    google_search = _config_flag(config, 'google_search', 'googleSearch')
    has_search_options = google_search is not False and any(
        _has_nonempty_items(val)
        for snake, camel in _GOOGLE_SEARCH_OPTION_PAIRS
        if (val := _config_flag(config, snake, camel)) is not None
    )
    if google_search or has_search_options:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'{INVALID_ARGUMENT_MESSAGES["tools"]} (google_search)',
        )
    if _config_flag(config, 'url_context', 'urlContext'):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'{INVALID_ARGUMENT_MESSAGES["tools"]} (url_context)',
        )
    stores = _config_flag(config, 'file_search_store_names', 'fileSearchStoreNames')
    has_file_search_opts = (
        _config_flag(config, 'file_search_metadata_filter', 'fileSearchMetadataFilter') is not None
        or _config_flag(config, 'file_search_top_k', 'fileSearchTopK') is not None
    )
    if _has_nonempty_items(stores) or has_file_search_opts:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=f'{INVALID_ARGUMENT_MESSAGES["tools"]} (file_search_store_names)',
        )
    return True
