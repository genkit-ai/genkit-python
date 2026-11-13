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

"""Type-checker contract for shipped middleware constructors in ``genkit_middleware``.

pyright, pyrefly, and ty include ``packages/*/tests/typecheck/`` with strict
argument, overload, and unused-ignore checks enabled.
"""

from pathlib import Path

from genkit_middleware import Fallback, Filesystem, Retry, Skills, ToolApproval
from genkit_middleware._fallback import FallbackConfig
from genkit_middleware._filesystem import FilesystemConfig
from genkit_middleware._retry import RetryConfig
from genkit_middleware._skills import SkillsConfig
from genkit_middleware._tool_approval import ToolApprovalConfig
from typing_extensions import assert_type

from genkit.middleware import Middleware
from genkit.model import ModelConfig, ModelRef


class _MenuModelConfig(ModelConfig):
    pass


def test_middleware_keyword_constructors_typecheck(tmp_path: Path) -> None:
    """Callers get typed keyword constructors without importing separate Config classes."""
    backup_ref = ModelRef(
        name='bistro/backup',
        config_schema=_MenuModelConfig,
        config=_MenuModelConfig(temperature=0.2),
    )

    retry = Retry(max_retries=3, statuses=['UNAVAILABLE'], initial_delay_ms=250, no_jitter=True)
    fallback = Fallback(models=['bistro/primary', backup_ref], statuses=['UNAVAILABLE'])
    filesystem = Filesystem(root_dir=str(tmp_path), allow_write_access=True, tool_name_prefix='ws_')
    skills = Skills(skill_paths=[str(tmp_path)])
    approval = ToolApproval(allowed_tools=['check_allergens'])

    assert_type(retry, Retry)
    assert_type(fallback, Fallback)
    assert_type(filesystem, Filesystem)
    assert_type(skills, Skills)
    assert_type(approval, ToolApproval)

    use_list: list[Middleware] = [retry, fallback, filesystem, skills, approval]
    assert len(use_list) == 5


def test_middleware_config_constructors_typecheck(tmp_path: Path) -> None:
    """Explicit ``config=`` construction keeps working alongside keyword fields."""
    assert_type(Retry(config=RetryConfig(max_retries=2)), Retry)
    assert_type(Fallback(config=FallbackConfig(models=['bistro/backup'])), Fallback)
    assert_type(Filesystem(config=FilesystemConfig(root_dir=str(tmp_path))), Filesystem)
    assert_type(Skills(config=SkillsConfig(skill_paths=[str(tmp_path)])), Skills)
    assert_type(ToolApproval(config=ToolApprovalConfig(allowed_tools=['check_allergens'])), ToolApproval)


def _negative_contract_typos_and_missing_root_are_static_errors() -> None:
    """Misspelled kwargs and missing required ``root_dir`` fail pyright, pyrefly, and ty."""
    Retry(max_retry=3)  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    Filesystem()  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
