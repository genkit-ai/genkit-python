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

"""The parts of the MCP client API that differ between mcp 1.x and 2.x.

Each difference is detected from the installed SDK, not from its version string.
Names that only one major version defines are looked up with ``getattr`` so the
module imports, and type checks, against either.
"""

import importlib
from datetime import timedelta
from typing import Any

from mcp import ClientSession
from mcp.types import ErrorData, ServerCapabilities
from pydantic import BaseModel

_exceptions = importlib.import_module('mcp.shared.exceptions')
_types = importlib.import_module('mcp.types')

McpError: type[Exception] = getattr(_exceptions, 'MCPError', None) or getattr(_exceptions, 'McpError')  # noqa: B009
"""The error the SDK raises for a JSON-RPC error reply. mcp 2 renamed ``McpError`` to ``MCPError``."""

REQUEST_TIMEOUT: int = getattr(_types, 'REQUEST_TIMEOUT', 408)
"""The code on the ``McpError`` the SDK raises when a request outlives its read timeout.

mcp 1.x used the HTTP status 408. mcp 2 uses the JSON-RPC code -32001 and exports it.
"""

_SESSION_HAS_PROPERTIES = hasattr(ClientSession, 'server_capabilities')
"""Whether this is mcp 2's ``ClientSession``, which also takes timeouts as float seconds rather than ``timedelta``."""


def error_data(error: BaseException) -> ErrorData | None:
    """Return the JSON-RPC error an ``McpError`` carries, or ``None`` for any other exception."""
    if not isinstance(error, McpError):
        return None
    data: ErrorData = getattr(error, 'error')  # noqa: B009 - McpError is typed as a plain Exception here.
    return data


def read_timeout(timeout: timedelta | None) -> Any:  # noqa: ANN401 - timedelta on mcp 1.x, float on mcp 2.
    """Convert a request timeout to the type this SDK's ``ClientSession`` takes."""
    if timeout is None or not _SESSION_HAS_PROPERTIES:
        return timeout
    return timeout.total_seconds()


def server_capabilities(session: ClientSession) -> ServerCapabilities | None:
    """Return what the server advertised at initialize, or ``None`` before the handshake."""
    if _SESSION_HAS_PROPERTIES:
        return getattr(session, 'server_capabilities')  # noqa: B009 - absent from mcp 1.x.
    return getattr(session, 'get_server_capabilities')()  # noqa: B009 - removed in mcp 2.


def field(model: BaseModel, name: str) -> Any:  # noqa: ANN401 - the field's own type.
    """Read a protocol field by its mcp 2 snake_case name, falling back to the camelCase name mcp 1.x uses.

    Args:
        model: An ``mcp.types`` model.
        name: The snake_case field name, such as ``next_cursor``.

    Returns:
        The field's value.
    """
    if name in type(model).model_fields:
        return getattr(model, name)
    head, *rest = name.split('_')
    return getattr(model, head + ''.join(part.capitalize() for part in rest))
