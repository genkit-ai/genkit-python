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

"""Anthropic plugin for Genkit."""

from collections.abc import Mapping
from typing import Any, Literal

from genkit import GenkitError, ModelResponse, get_logger
from genkit.model import Model, ModelContext, ModelRef, ModelRequest
from genkit.plugin_api import (
    Action,
    ActionKind,
    Plugin,
)
from genkit_anthropic._anthropic_api import AnthropicApiBackend
from genkit_anthropic._backend import Backend
from genkit_anthropic._config import AnthropicConfig
from genkit_anthropic._google_cloud import GoogleCloud, GoogleCloudBackend
from genkit_anthropic._model_info import SUPPORTED_ANTHROPIC_MODELS, KnownClaude, get_model_info
from genkit_anthropic._models import AnthropicModel, ClaudeClient
from genkit_anthropic._secrets import context_api_key, reject_config_api_key

logger = get_logger(__name__)

ANTHROPIC_PLUGIN_NAME = 'anthropic'


def _backend_for(
    config: GoogleCloud | None,
    *,
    api_key: str | None,
    base_url: str | None,
    client_options: Mapping[str, Any] | None,
) -> Backend[ClaudeClient]:
    """Turn the public ``backend=`` value into the backend that serves the models.

    ``GoogleCloud`` is the public choice; ``GoogleCloudBackend`` implements it.
    This is the only place a backend is picked; past it, the plugin treats
    every backend the same.

    Every backend takes the same arguments and rejects the ones it can't use.

    Raises:
        TypeError: ``backend`` isn't ``None`` or a ``GoogleCloud``.
    """
    if config is None:
        return AnthropicApiBackend(api_key=api_key, base_url=base_url, client_options=client_options)
    if not isinstance(config, GoogleCloud):
        raise TypeError(f'Anthropic backend must be GoogleCloud(...) or None, got {type(config).__name__}')
    return GoogleCloudBackend(config, api_key=api_key, base_url=base_url, client_options=client_options)


# Only this plugin's namespace. A vertexai/ paste is a different name —
# remapping it here would send the request to the Anthropic API.
_STRIP_PREFIXES = (f'{ANTHROPIC_PLUGIN_NAME}/',)


def _strip_ref_prefixes(name: str) -> str:
    """Peel this plugin's prefix so it is not stamped twice."""
    local = name
    changed = True
    while changed:
        changed = False
        for prefix in _STRIP_PREFIXES:
            if local.startswith(prefix):
                local = local[len(prefix) :]
                changed = True
    return local


def anthropic_name(name: str) -> str:
    """Get Anthropic model name.

    Args:
        name: The name of Anthropic model.

    Returns:
        Fully qualified Anthropic model name.
    """
    return f'{ANTHROPIC_PLUGIN_NAME}/{name}'


class Anthropic(Plugin):
    """Anthropic plugin for Genkit.

    This plugin adds Anthropic models to Genkit for generative AI applications.
    """

    name = ANTHROPIC_PLUGIN_NAME

    @classmethod
    def claude(cls, name: KnownClaude | str, *, config: AnthropicConfig | None = None) -> ModelRef[AnthropicConfig]:
        """Typed ref for a Claude model, e.g. ``Anthropic.claude('claude-sonnet-4-5')``.

        Unknown ids are allowed so new Claude releases work before this
        plugin learns their names; every Anthropic generate model takes
        AnthropicConfig.
        """
        # str(None) is 'None', and unknown ids are allowed, so a non-string
        # would mint a real-looking ref instead of failing here.
        if not isinstance(name, str):
            raise GenkitError(status='INVALID_ARGUMENT', message='Anthropic.claude: model name must be a string.')
        local = _strip_ref_prefixes(name)
        if not local:
            raise GenkitError(status='INVALID_ARGUMENT', message='Anthropic.claude: model name is required.')
        return ModelRef(name=f'{ANTHROPIC_PLUGIN_NAME}/{local}', config_schema=AnthropicConfig, config=config)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        api_version: Literal['stable', 'beta'] | None = None,
        client_options: Mapping[str, Any] | None = None,
        backend: GoogleCloud | None = None,
    ) -> None:
        """Initializes Anthropic plugin with given configuration.

        Any Claude model id resolves on demand, so there is no model list to pass.

        Example:
            ```python
            # 1. Claude on the Anthropic API (ANTHROPIC_API_KEY)
            Anthropic()

            # 2. Claude on Google Cloud, billed to your project (application default credentials)
            Anthropic(backend=GoogleCloud(project='bistro-prod'))

            # 3. Refs, config and .prompt files don't change between the two
            await ai.generate(model=Anthropic.claude('claude-sonnet-4-6'), prompt='Suggest a dish.')
            ```

        Args:
            api_key: Anthropic API key. Defaults to ``ANTHROPIC_API_KEY``. Not
                used with ``backend=GoogleCloud(...)``.
            base_url: Anthropic API origin with no ``/v1`` suffix, e.g.
                ``https://claude-proxy.example.com``; the SDK appends
                ``/v1/messages``. Defaults to ``ANTHROPIC_BASE_URL``, then the
                public API. A value ending in ``/v1`` raises. With
                ``backend=GoogleCloud(...)`` it overrides the Google Cloud endpoint.
            api_version: Default API surface unless overridden by per-request
                config. An explicit config ``apiVersion`` always takes
                precedence. Defaults to stable.
            client_options: Other SDK client settings, such as ``timeout``,
                ``max_retries``, ``default_headers``, or ``http_client``
                (``AsyncAnthropic``, or ``AsyncAnthropicVertex`` with
                ``backend=GoogleCloud(...)``).
            backend: Where the models are served. ``None`` (the default) is the
                Anthropic API; ``GoogleCloud(project=..., location=...)``
                serves them from Google Cloud with application default
                credentials. Needs the ``google-cloud`` extra.

        Raises:
            ValueError: If ``api_version`` is not ``'stable'``, ``'beta'``, or
                ``None``, the base URL ends in ``/v1``, or ``api_key`` is
                passed with ``backend=GoogleCloud(...)``.
            TypeError: If ``backend`` isn't a ``GoogleCloud``, or
                ``client_options`` has a key the SDK client doesn't take or
                repeats ``api_key`` or ``base_url``.
            ImportError: ``backend=GoogleCloud(...)`` without the ``google-cloud`` extra.
        """
        if api_version not in (None, 'stable', 'beta'):
            raise ValueError("api_version must be 'stable', 'beta', or None")

        self._backend = _backend_for(backend, api_key=api_key, base_url=base_url, client_options=client_options)
        self._default_api_version: Literal['stable', 'beta'] | None = api_version
        self._list_actions_cache: list[Action] | None = None

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Returns:
            Empty list (using lazy loading via resolve).
        """
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Args:
            action_type: The kind of action to resolve.
            name: The id without the ``anthropic/`` prefix.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type != ActionKind.MODEL:
            return None

        return self._create_model_action(name)

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for an Anthropic model.

        Args:
            name: The model id as received (no plugin-prefix stripping).

        Returns:
            Action object for the model.
        """
        model_info = get_model_info(name)

        async def _generate(request: ModelRequest[AnthropicConfig], ctx: ModelContext) -> ModelResponse:
            # A key in config is the more specific error, so it wins over a missing plugin key.
            reject_config_api_key(request.config)
            client = await self._backend.request_client(context_api_key(ctx.context))
            model = AnthropicModel(
                model_name=name,
                client=client,
                default_api_version=self._default_api_version,
                default_betas=self._backend.default_betas,
            )
            return await model.generate(request, ctx)

        return Model(
            anthropic_name(name),
            _generate,
            config_schema=AnthropicConfig,
            info=model_info,
            metadata={
                'model': {
                    'supports': (
                        model_info.supports.model_dump(by_alias=True, exclude_none=True) if model_info.supports else {}
                    ),
                },
            },
        )

    async def list_actions(self) -> list[Action]:
        """List available Anthropic models.

        Asks the backend for the models it serves and returns the union of
        those and any statically known models not already covered (backend
        ids first, in backend order, then remaining static ids, deduplicated
        by bare id). A backend without a Models API, such as Google Cloud,
        lists only the static models. The
        successful result is cached for the lifetime of the plugin instance.
        If the API call fails, logs a warning and returns the static model
        list only, without caching the failure, so the next call retries the
        API.

        Returns:
            List of Action objects for all discovered/supported models.
        """
        if self._list_actions_cache is not None:
            return self._list_actions_cache

        try:
            model_ids = await self._backend.list_model_ids()
        except Exception as e:
            logger.warning('Failed to list Anthropic models, using static model list', error=str(e))
            return [self._create_model_action(model_id) for model_id in SUPPORTED_ANTHROPIC_MODELS]

        seen: set[str] = set()
        ordered_ids: list[str] = []
        for model_id in model_ids:
            if model_id and model_id not in seen:
                seen.add(model_id)
                ordered_ids.append(model_id)
        for model_id in SUPPORTED_ANTHROPIC_MODELS:
            if model_id not in seen:
                seen.add(model_id)
                ordered_ids.append(model_id)

        actions = [self._create_model_action(model_id) for model_id in ordered_ids]
        self._list_actions_cache = actions
        return actions
