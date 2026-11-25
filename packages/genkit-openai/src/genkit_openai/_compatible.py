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

"""Any server that speaks the OpenAI API, registered under a name you pick."""

from collections.abc import Mapping
from typing import Any

from openai import APIError, AsyncOpenAI

from genkit import GenkitError, ModelResponse, get_logger
from genkit.embedder import EmbedContext, Embedder, EmbedderInfo, EmbedderSupports, EmbedRequest, EmbedResponse
from genkit.model import Model, ModelContext, ModelInfo, ModelRef, ModelRequest
from genkit.plugin_api import Action, ActionKind, Plugin, loop_local_client
from genkit_openai._models._handler import OpenAIModelHandler
from genkit_openai._models._mistral import is_mistral
from genkit_openai._models._model import OpenAIModel
from genkit_openai._models._model_info import get_default_openai_model_info
from genkit_openai._openai_plugin import ApiKey, _client_kwargs, embed_texts
from genkit_openai._secrets import context_api_key
from genkit_openai._typing import OpenAIConfig

logger = get_logger(__name__)

# Sent when the server needs no key. AsyncOpenAI requires one, and without a
# value it would read OPENAI_API_KEY and send an OpenAI key to another server.
_NO_KEY = 'no-key'


class OpenAICompatible(Plugin):
    """Any OpenAI-compatible server (OpenRouter, Together, Groq, vLLM, LiteLLM) under a name you pick.

    Models and embedders register as ``<name>/<id>``, and ``<id>`` goes to the
    server exactly as written, so ids with slashes (``meta-llama/...``,
    ``openai/gpt-4o``) need no escaping. There's no built-in model list:
    any id resolves when you call it, and the Dev UI lists what the server's
    ``/models`` returns. Several instances, and an ``OpenAI()``, can run side
    by side because each has its own name.

    Example:
        ```python
        from genkit import Genkit
        from genkit_openai import OpenAI, OpenAICompatible

        # 1. OpenAI's API, plus OpenRouter under its own name
        openrouter = OpenAICompatible(name='openrouter', base_url='https://openrouter.ai/api/v1', api_key=key)
        ai = Genkit(plugins=[OpenAI(), openrouter])

        # 2. The id is OpenRouter's own, sent as written
        res = await ai.generate(model=openrouter.model('meta-llama/llama-3.3-70b-instruct'), prompt='Name a bakery.')
        print(res.text)
        # => Rise & Shine Bakehouse

        # 3. A local server that needs no key
        local = OpenAICompatible(name='local', base_url='http://localhost:8000/v1')
        # => models resolve as local/<id>
        ```
    """

    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: ApiKey | None = None,
        client_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Point the plugin at one OpenAI-compatible server.

        Args:
            name: The namespace for this server's models and embedders, e.g.
                ``'openrouter'`` gives ``openrouter/<id>``.
            base_url: The server's API base URL up to and including ``/v1``,
                e.g. ``https://openrouter.ai/api/v1``; the SDK appends
                ``/chat/completions``.
            api_key: The server's key, or an async function that returns one
                before each request (for a token that expires). ``None`` sends
                no real key, for local servers. ``OPENAI_API_KEY`` is never
                read.
            client_options: Other ``AsyncOpenAI`` settings, such as
                ``timeout``, ``max_retries``, ``default_headers``, or
                ``http_client``.

        Raises:
            ValueError: ``name`` is empty or contains ``/``, or ``base_url`` is empty.
            TypeError: A ``client_options`` key ``AsyncOpenAI`` doesn't take,
                or ``api_key``/``base_url`` repeated there.
        """
        if not isinstance(name, str) or not name or '/' in name:
            raise ValueError(f"OpenAICompatible name must be a non-empty string without '/', got {name!r}")
        if not base_url:
            raise ValueError("OpenAICompatible needs base_url, e.g. 'https://openrouter.ai/api/v1'")
        options = _client_kwargs(
            api_key=api_key, base_url=base_url, client_options=client_options, plugin='OpenAICompatible'
        )
        options.setdefault('api_key', _NO_KEY)
        self.name = name
        self._options = options
        self._list_actions_cache: list[Action] | None = None

    def model(self, name: str, *, config: OpenAIConfig | None = None) -> ModelRef[OpenAIConfig]:
        """Typed ref for one of this server's chat models, e.g. ``openrouter.model('openai/gpt-4o')``.

        Args:
            name: The id the server knows the model by, sent as written.
            config: Default config for calls through this ref.
        """
        if not isinstance(name, str) or not name:
            raise GenkitError(
                status='INVALID_ARGUMENT', message=f'{self.name}.model: model name must be a non-empty string.'
            )
        return ModelRef(name=f'{self.name}/{name}', config_schema=OpenAIConfig, config=config)

    @loop_local_client
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(**self._options)

    def _client_for(self, context: Mapping[str, Any] | None) -> AsyncOpenAI:
        """The plugin's client, or a copy on the caller's ``context.secrets`` key."""
        key = context_api_key(context)
        client = self._client()
        return client.with_options(api_key=key) if key else client

    async def init(self) -> list[Action]:
        """Nothing up front: every id resolves on demand."""
        return []

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Build the action for ``name``, by the kind the caller asked for.

        Args:
            action_type: ``MODEL`` for chat, ``EMBEDDER`` for embeddings.
            name: The server's id, without this plugin's prefix.
        """
        if action_type == ActionKind.MODEL:
            return self._model_action(name)
        if action_type == ActionKind.EMBEDDER:
            return self._embedder_action(name)
        return None

    def _model_action(self, model_id: str) -> Action:
        info = ModelInfo(label=f'{self.name} - {model_id}', supports=get_default_openai_model_info(model_id).supports)

        async def _generate(request: ModelRequest[OpenAIConfig], ctx: ModelContext) -> ModelResponse:
            client = self._client_for(ctx.context)
            model = OpenAIModel(model_id, client, supports=info.supports, mistral=is_mistral(client))
            return await OpenAIModelHandler(model).generate(request, ctx)

        return Model(f'{self.name}/{model_id}', _generate, config_schema=OpenAIConfig, info=info)

    def _embedder_action(self, model_id: str) -> Action:
        async def _embed(request: EmbedRequest, ctx: EmbedContext) -> EmbedResponse:
            client = self._client_for(ctx.context)
            return await embed_texts(client, model_id, request, mistral=is_mistral(client))

        return Embedder(
            f'{self.name}/{model_id}',
            _embed,
            info=EmbedderInfo(label=f'{self.name} - {model_id}', supports=EmbedderSupports(input=['text'])),
        )

    async def list_actions(self) -> list[Action]:
        """What the server's ``/models`` lists, as models and embedders.

        Ids containing ``embed`` are listed as embedders, everything else as
        chat models; either can still be called as the other kind. A server
        without ``/models``, or one that rejects the call, lists nothing and
        logs why; that isn't cached, so the next listing tries again.
        """
        if self._list_actions_cache is not None:
            return self._list_actions_cache
        try:
            page = await self._client().models.list()
        except (APIError, GenkitError) as e:
            logger.warning(
                'Could not list models; the Dev UI shows none for this plugin', plugin=self.name, error=str(e)
            )
            return []
        actions = [
            self._embedder_action(m.id) if 'embed' in m.id else self._model_action(m.id) for m in page.data if m.id
        ]
        self._list_actions_cache = actions
        return actions
