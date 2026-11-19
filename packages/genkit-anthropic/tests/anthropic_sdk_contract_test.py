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


"""What the plugin assumes about the installed ``anthropic`` SDK.

anthropic 1.0 dropped ``temperature``/``top_p``/``top_k`` from
``messages.create()`` and moved to httpx2. The plugin reads both off the
installed SDK; these tests pin that it keeps doing so across the declared range.
"""

import inspect
import re
from collections.abc import Callable
from importlib.metadata import requires
from unittest.mock import MagicMock

import pytest
from anthropic import DefaultAsyncHttpxClient
from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
from anthropic.resources.messages import AsyncMessages
from genkit_anthropic import AnthropicConfig
from genkit_anthropic._compat import SDK_MESSAGE_KWARGS, httpx
from genkit_anthropic._models import AnthropicModel
from packaging.requirements import Requirement

from genkit import Message, Part, Role
from genkit.model import ModelRequest

_SAMPLING = {'temperature': 0.2, 'top_p': 0.9, 'top_k': 40, 'stop_sequences': ['END']}

_METHODS = [AsyncMessages.create, AsyncMessages.stream, AsyncBetaMessages.create, AsyncBetaMessages.stream]


def _request() -> ModelRequest:
    return ModelRequest(messages=[Message(role=Role.USER, content=[Part.from_text('Suggest a dish.')])])


def test_sampling_fields_cover_create_kwargs() -> None:
    """The sampling fixture lists every ``CREATE_KWARGS`` key."""
    assert set(_SAMPLING) == AnthropicConfig.CREATE_KWARGS


@pytest.mark.parametrize('method', _METHODS, ids=['messages', 'messages.stream', 'beta', 'beta.stream'])
def test_every_top_level_kwarg_is_named_by_the_sdk(method: Callable[..., object]) -> None:
    """A sampling field goes top-level only when the installed SDK names it, else into ``extra_body``."""
    params = AnthropicModel(model_name='claude-sonnet-4', client=MagicMock())._build_params(
        _request(), config=AnthropicConfig(**_SAMPLING)
    )
    named = inspect.signature(method).parameters.keys()

    top_level = {key: params[key] for key in _SAMPLING if key in params}
    in_body = {key: value for key, value in params.get('extra_body', {}).items() if key in _SAMPLING}

    assert set(top_level) <= named
    assert {**top_level, **in_body} == _SAMPLING
    assert set(top_level).isdisjoint(in_body)


def test_compat_kwargs_match_sdk() -> None:
    """``SDK_MESSAGE_KWARGS`` is what every request method the plugin calls accepts."""
    for method in _METHODS:
        assert SDK_MESSAGE_KWARGS <= inspect.signature(method).parameters.keys()


def test_sdk_transport_matches_compat_httpx() -> None:
    """The ``except`` handlers in ``_models.py`` catch the transport the SDK actually uses."""
    assert issubclass(DefaultAsyncHttpxClient, httpx.AsyncClient)


def test_declared_anthropic_range_excludes_3x() -> None:
    """``genkit-anthropic`` caps ``anthropic`` below 3.0 until a 3.x release is vetted."""
    reqs = [Requirement(r) for r in requires('genkit-anthropic') or [] if re.match(r'anthropic\b', r)]
    assert len(reqs) == 1, reqs
    assert '3.0.0' not in reqs[0].specifier, reqs[0]
