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

"""What differs between the places the Anthropic plugin serves Claude from."""

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, ClassVar, Generic, TypeVar

from genkit.plugin_api import loop_local_client
from genkit_anthropic._models import ClaudeClient

ClientT_co = TypeVar('ClientT_co', bound=ClaudeClient, covariant=True)


class Backend(ABC, Generic[ClientT_co]):
    """One place Claude is served from, e.g. the Anthropic API or Google Cloud.

    The plugin and ``AnthropicModel`` are the same for every backend. A backend
    owns only what differs: the SDK client it builds, how a call gets its
    credential, which models it can list, and which beta headers it offers.
    """

    # Sent on beta-surface calls that don't pass their own betas.
    default_betas: ClassVar[tuple[str, ...]] = ()

    @abstractmethod
    def _new_client(self) -> ClientT_co:
        """Build the SDK client from the backend's settings."""

    @loop_local_client
    def client(self) -> ClientT_co:
        """The shared SDK client for the running event loop.

        Calls go through ``request_client``, which runs the backend's checks
        first; only tests reach for this directly.
        """
        return self._new_client()

    @abstractmethod
    async def request_client(self, api_key: str | None) -> ClientT_co:
        """The client for one call, given the caller's ``context.secrets`` key, if any.

        Raises:
            GenkitError: FAILED_PRECONDITION when the call can't be authenticated.
        """

    @abstractmethod
    async def list_model_ids(self) -> list[str]:
        """Model ids the backend reports; the plugin adds its built-in catalog."""


def client_kwargs(
    client_cls: type[ClaudeClient],
    client_options: Mapping[str, Any] | None,
    *,
    reserved: frozenset[str] = frozenset(),
    api_key: str | None = None,
    base_url: str | None = None,
) -> dict[str, Any]:
    """Merge the plugin's typed settings into ``client_options`` for ``client_cls``.

    Args:
        client_cls: The SDK client the options are for.
        client_options: The user's ``client_options``.
        reserved: Constructor keys the backend sets itself, so users can't.
        api_key: The plugin's ``api_key`` argument; None means unset.
        base_url: The plugin's ``base_url`` argument; None means unset.

    Raises:
        TypeError: A ``client_options`` key ``client_cls`` doesn't take, or a
            setting given both as an argument and in ``client_options``.
    """
    allowed = frozenset(inspect.signature(client_cls.__init__).parameters) - {'self'} - reserved
    options = dict(client_options or {})
    for key in options:
        if key not in allowed:
            raise TypeError(
                f'Anthropic got an unexpected client_options key {key!r}; '
                f'{client_cls.__name__} accepts {", ".join(sorted(k for k in allowed if not k.startswith("_")))}'
            )
    for name, value in (('api_key', api_key), ('base_url', base_url)):
        if value is None:
            continue
        if name in options:
            raise TypeError(f'Anthropic got {name} both as an argument and in client_options; pass it once')
        options[name] = value
    return options
