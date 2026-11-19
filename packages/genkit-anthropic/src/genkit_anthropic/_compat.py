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

"""What differs across the supported ``anthropic`` range, read off the installed SDK.

anthropic 1.0 changed two things the plugin depends on:

- The transport moved from ``httpx`` to ``httpx2``, an API-compatible fork in
  a different module. Exceptions raised mid-stream come from whichever one the
  SDK uses, so ``httpx`` here is that module.
- ``temperature``, ``top_p`` and ``top_k`` left the ``messages.create()``
  signature. The API still takes them, so they go in ``extra_body`` when the
  installed SDK has no named parameter for them.
"""

import importlib
import inspect
from typing import TYPE_CHECKING

from anthropic import DefaultAsyncHttpxClient
from anthropic.resources.beta.messages import AsyncMessages as AsyncBetaMessages
from anthropic.resources.messages import AsyncMessages

if TYPE_CHECKING:
    # Type checkers run against the locked SDK, which is on httpx2.
    import httpx2 as httpx
else:
    _transport_client = next(c for c in DefaultAsyncHttpxClient.__mro__ if c.__name__ == 'AsyncClient')
    httpx = importlib.import_module(_transport_client.__module__.partition('.')[0])

__all__ = ['SDK_MESSAGE_KWARGS', 'TRANSPORT_DROP_ERRORS', 'TRANSPORT_TIMEOUT_ERRORS', 'httpx']

TRANSPORT_TIMEOUT_ERRORS: tuple[type[Exception], ...] = (httpx.TimeoutException,)
TRANSPORT_DROP_ERRORS: tuple[type[Exception], ...] = (httpx.NetworkError, httpx.RemoteProtocolError)

# Named parameters every request method the plugin calls accepts.
SDK_MESSAGE_KWARGS: frozenset[str] = frozenset.intersection(
    *(
        frozenset(inspect.signature(method).parameters)
        for method in (AsyncMessages.create, AsyncMessages.stream, AsyncBetaMessages.create, AsyncBetaMessages.stream)
    )
)
