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

"""Runtime side of the public type contract.

Annotations on public classes resolve with ``typing.get_type_hints`` (FastAPI
and Pydantic introspect them), and the generic session types parametrize.
"""

import importlib
import typing

import pytest

from genkit import Message, Part, Role
from genkit._ai._background_model import BackgroundModel
from genkit._core._background import BackgroundAction
from genkit.model import GenerationUsage
from genkit.plugin_api import Action


@pytest.mark.parametrize(
    'target',
    [
        Action.__call__,
        Action.run,
        Action.stream,
        BackgroundAction.__init__,
        BackgroundAction.lookup,
        BackgroundModel.__init__,
        BackgroundModel.lookup,
        GenerationUsage.from_messages,
    ],
    ids=lambda t: t.__qualname__,
)
def test_public_annotations_resolve_at_runtime(target: object) -> None:
    """get_type_hints() resolves every annotation instead of raising NameError/TypeError."""
    assert typing.get_type_hints(target)


def test_generation_usage_from_messages_counts_characters() -> None:
    """from_messages still counts the request and reply after the signature change."""
    usage = GenerationUsage.from_messages(
        [Message(role=Role.USER, content=[Part.from_text('Table for two')])],
        Message(role=Role.MODEL, content=[Part.from_text('Booked')]),
    )

    assert usage.input_characters == len('Table for two')
    assert usage.output_characters == len('Booked')


@pytest.mark.parametrize(
    ('module', 'names'),
    [
        ('genkit.model', ['ModelFn']),
        ('genkit.plugin_api', ['ActionResponse']),
        ('genkit_mcp', ['McpServerEntry']),
    ],
)
def test_signature_types_are_exported(module: str, names: list[str]) -> None:
    """Types that appear in public signatures import from their canonical module."""
    mod = importlib.import_module(module)
    for name in names:
        assert name in mod.__all__, f'{module}.{name}'
        assert hasattr(mod, name)


@pytest.mark.parametrize(
    ('module', 'names'),
    [
        (
            'genkit',
            [
                'ActionResponse',
                'ModelConfig',
                'ModelConfigDict',
                'ModelStreamingCallback',
                'Resource',
                'StatusName',
                'StoppableStream',
                'ToolRequest',
                'ToolResponse',
            ],
        ),
        ('genkit.middleware', ['MiddlewareRef']),
        ('genkit.plugin_api', ['ActionMetadata', 'custom_options_schema']),
    ],
)
def test_non_app_and_wire_types_stay_out_of_genkit_and_middleware_exports(module: str, names: list[str]) -> None:
    """Internal wire/reflection types do not leak into ``genkit``, ``genkit.middleware``, or ``genkit.plugin_api``."""
    mod = importlib.import_module(module)
    for name in names:
        assert name not in mod.__all__, f'{module}.{name}'
        assert not hasattr(mod, name), f'{module}.{name}'
