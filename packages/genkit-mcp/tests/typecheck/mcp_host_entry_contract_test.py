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

"""Type-checker contract for building ``mcp_servers`` away from the call.

A dict literal assigned to a variable infers as ``dict[str, dict[str, str]]``
and fails every checker against ``Mapping[str, McpServerEntry]``. Annotating
the variable with the exported ``McpServerEntry`` is the typed way to do it.
"""

from genkit_mcp import McpHost, McpServerEntry, define_mcp_host
from typing_extensions import assert_type

from genkit import Genkit


def test_servers_built_in_a_variable_pass_define_mcp_host() -> None:
    """``dict[str, McpServerEntry]`` built ahead of the call type-checks and registers."""
    ai = Genkit()

    # 1. Build the servers block once, e.g. from app settings
    servers: dict[str, McpServerEntry] = {
        'menu': {'command': 'uvx', 'args': ['menu-mcp-server']},
        'orders': {'command': 'npx', 'args': ['orders-mcp'], 'env': {'REGION': 'us'}},
    }

    # 2. Pass it to the host; connection is lazy, so nothing starts here
    host = define_mcp_host(ai, 'bistro', mcp_servers=servers)
    assert_type(host, McpHost)
