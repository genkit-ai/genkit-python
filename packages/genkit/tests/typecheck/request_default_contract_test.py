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

# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Typed contract of bare ``ModelRequest`` / ``EmbedRequest`` annotations.

A bare request means ``Request[Any]``, the way bare ``list`` means
``list[Any]``. A helper annotated with the bare type must accept a typed
request in pyright, pyrefly and ty with no suppressions; an unneeded ignore
here fails CI.
"""

from typing import Any

from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Document, Message, Part, Role
from genkit.embedder import EmbedRequest
from genkit.model import ModelConfig, ModelRequest


class MenuEmbedConfig(BaseModel):
    """Options for a menu embedder."""

    dimensions: int = 8


class KitchenConfig(ModelConfig):
    """Config for a kitchen model."""

    spice_level: int = 1


def count_dishes(request: EmbedRequest) -> int:
    """A shared helper annotated with bare ``EmbedRequest``."""
    return len(request.input)


def count_turns(request: ModelRequest) -> int:
    """A shared helper annotated with bare ``ModelRequest``."""
    return len(request.messages)


def count_batch(requests: list[EmbedRequest]) -> int:
    """A helper over a list of bare ``EmbedRequest``."""
    return sum(len(r.input) for r in requests)


def test_bare_embed_request_helper_accepts_typed_request() -> None:
    """``EmbedRequest[Cfg]`` passes to a helper typed with bare ``EmbedRequest``."""
    typed = EmbedRequest[MenuEmbedConfig](input=[Document.from_text('Smoked Salmon Tartine')])

    assert count_dishes(typed) == 1
    assert count_batch([typed, typed]) == 2


def test_bare_model_request_helper_accepts_typed_request() -> None:
    """``ModelRequest[Cfg]`` passes to a helper typed with bare ``ModelRequest``."""
    typed = ModelRequest[KitchenConfig](
        messages=[Message(role=Role.USER, content=[Part.from_text('Plate the tartine')])]
    )

    assert count_turns(typed) == 1


def read_options(request: EmbedRequest) -> object:
    """A bare annotation: the default type argument applies here."""
    assert_type(request.options, Any)
    return request.options


def test_bare_embed_request_options_is_any_statically_and_dict_at_runtime() -> None:
    """Bare options is ``Any`` to checkers and a ``dict`` at runtime."""
    bare = EmbedRequest(input=[Document.from_text('tartine')], options={'dimensions': 4})

    options = read_options(bare)
    assert options == {'dimensions': 4}
    assert isinstance(options, dict)
