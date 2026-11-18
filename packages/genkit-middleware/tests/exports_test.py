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

"""Public surface of ``genkit_middleware``, typed constructors, and wrong-class plugin error."""

from pathlib import Path

import genkit_middleware
import pytest
from genkit_middleware import (
    Fallback,
    Filesystem,
    Middleware,
    Retry,
    Skills,
    ToolApproval,
)
from genkit_middleware._fallback import FallbackConfig, FallbackModelEntry
from genkit_middleware._filesystem import FilesystemConfig
from genkit_middleware._retry import RetryConfig
from genkit_middleware._skills import SkillsConfig
from genkit_middleware._tool_approval import ToolApprovalConfig

from genkit import Genkit


def test_public_exports_stay_tight() -> None:
    """``genkit_middleware`` exports only the plugin and middleware classes, not internal Config types."""
    assert set(genkit_middleware.__all__) == {
        'Fallback',
        'Filesystem',
        'Middleware',
        'Retry',
        'Skills',
        'ToolApproval',
    }
    assert Middleware().name == 'genkit-middleware'


def test_typed_keyword_and_config_constructors(tmp_path: Path) -> None:
    """Each shipped middleware accepts typed keyword arguments and ``config=``, rejecting both together."""
    retry = Retry(max_retries=5, initial_delay_ms=250, max_delay_ms=5000, backoff_factor=1.5, no_jitter=True)
    assert retry.config.max_retries == 5
    assert retry.config.no_jitter is True
    assert Retry(config=RetryConfig(max_retries=2)).config.max_retries == 2
    with pytest.raises(TypeError, match='pass either config= or keyword config fields'):
        Retry(config=RetryConfig(max_retries=2), max_retries=5)  # type: ignore[call-overload]

    entry = FallbackModelEntry(name='googleai/gemini-flash-lite-latest', config={'temperature': 0})
    fallback = Fallback(models=['googleai/gemini-pro-latest', entry], statuses=['UNAVAILABLE'])
    assert len(fallback.config.models) == 2
    assert Fallback(config=FallbackConfig(models=['backup'])).config.models == ['backup']

    fs = Filesystem(root_dir=str(tmp_path), allow_write_access=True, tool_name_prefix='ws_')
    assert fs.config.allow_write_access is True
    assert fs.config.tool_name_prefix == 'ws_'
    assert Filesystem(config=FilesystemConfig(root_dir=str(tmp_path))).config.root_dir == str(tmp_path)

    skills = Skills(skill_paths=[str(tmp_path)])
    assert skills.config.skill_paths == [str(tmp_path)]
    assert Skills(config=SkillsConfig(skill_paths=[str(tmp_path)])).config.skill_paths == [str(tmp_path)]

    approval = ToolApproval(allowed_tools=['check_allergens'])
    assert approval.config.allowed_tools == ['check_allergens']
    assert ToolApproval(config=ToolApprovalConfig(allowed_tools=['check_allergens'])).config.allowed_tools == [
        'check_allergens'
    ]


def test_middleware_class_in_plugins_names_plugin_api() -> None:
    with pytest.raises(ValueError, match=r'genkit\.plugin_api\.Plugin') as exc:
        Genkit(plugins=[Retry()])  # pyright: ignore[reportArgumentType]
    assert 'genkit.ai.Plugin' not in str(exc.value)
    assert 'genkit_middleware.Middleware()' in str(exc.value)
