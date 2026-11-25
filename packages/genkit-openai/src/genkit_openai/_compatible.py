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

import asyncio
from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Literal, cast, overload

from openai import APIError, AsyncOpenAI

from genkit import GenkitError, ModelResponse, get_logger
from genkit.embedder import EmbedContext, Embedder, EmbedderInfo, EmbedderSupports, EmbedRequest, EmbedResponse
from genkit.model import Model, ModelContext, ModelInfo, ModelRef, ModelRequest
from genkit.plugin_api import Action, ActionKind, Plugin, loop_local_client
from genkit_openai._google_cloud import (
    MODEL_GARDEN_NAME,
    AdcToken,
    GoogleCloud,
    load_adc,
    model_garden_base_url,
    require_google_auth,
    resolve_location,
    resolve_project,
)
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


class _Server(ABC):
    """Where an OpenAICompatible plugin sends its calls, and how they're authenticated."""

    @abstractmethod
    async def request_client(self, api_key: str | None) -> AsyncOpenAI:
        """The client for one call, given the caller's ``context.secrets`` key, if any."""

    @abstractmethod
    async def model_ids(self) -> list[str]:
        """Ids for the Dev UI list.

        Raises:
            APIError: The server rejected the listing.
        """


class _BaseUrlServer(_Server):
    """A server named by ``base_url``, with a static key, a token function, or no key."""

    def __init__(self, options: dict[str, Any]) -> None:
        self._options = options

    @loop_local_client
    def _client(self) -> AsyncOpenAI:
        return AsyncOpenAI(**self._options)

    async def request_client(self, api_key: str | None) -> AsyncOpenAI:
        """The plugin's client, or a copy on the caller's key."""
        client = self._client()
        return client.with_options(api_key=api_key) if api_key else client

    async def model_ids(self) -> list[str]:
        """What the server's ``/models`` lists."""
        page = await self._client().models.list()
        return [m.id for m in page.data if m.id]


class _ModelGardenServer(_Server):
    """Model Garden's OpenAI-compatible endpoint, authenticated with application default credentials."""

    def __init__(self, config: GoogleCloud, *, base_url: str | None, client_options: dict[str, Any]) -> None:
        self._base_url = base_url
        self._options = client_options
        # An explicit setting beats the environment; the credentials' project fills in last.
        self._project = resolve_project(config)
        self._location = resolve_location(config)
        self._token: AdcToken | None = None

    @loop_local_client
    def _client(self) -> AsyncOpenAI:
        # Only request_client builds the client, after _prepare set the project and token.
        project = cast(str, self._project)
        base_url = self._base_url or model_garden_base_url(project=project, location=self._location)
        return AsyncOpenAI(base_url=base_url, api_key=cast(AdcToken, self._token), **self._options)

    async def request_client(self, api_key: str | None) -> AsyncOpenAI:
        """The client, once credentials and project are known.

        Raises:
            GenkitError: FAILED_PRECONDITION for a ``context.secrets`` key,
                missing or stale credentials, or no project.
        """
        if api_key is not None:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    'A per-request API key from context.secrets cannot be used: '
                    'OpenAICompatible(backend=GoogleCloud()) authenticates with Google Cloud credentials.'
                ),
            )
        await self._prepare()
        return self._client()

    async def model_ids(self) -> list[str]:
        """Nothing: Model Garden's endpoint has no ``/models``, and any id resolves when called."""
        return []

    async def _prepare(self) -> None:
        """Load credentials once (in a thread: it can hit the metadata server); a failure isn't cached.

        Two first calls racing both load; the second result is equivalent, so
        there's no lock (an asyncio.Lock would bind to one event loop).
        """
        if self._token is not None:
            return
        credentials, adc_project = await asyncio.to_thread(load_adc)
        project = self._project or adc_project
        if project is None:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    'OpenAICompatible(backend=GoogleCloud()) needs a Google Cloud project: '
                    'pass GoogleCloud(project=...), set GOOGLE_CLOUD_PROJECT, '
                    'or run `gcloud config set project PROJECT_ID`'
                ),
            )
        self._project = project
        self._token = AdcToken(credentials)


class OpenAICompatible(Plugin):
    """Any OpenAI-compatible server (OpenRouter, Together, Groq, vLLM, LiteLLM) under a name you pick.

    Models and embedders register as ``<name>/<id>``, and ``<id>`` goes to the
    server exactly as written, so ids with slashes (``meta-llama/...``,
    ``openai/gpt-4o``) need no escaping. There's no built-in model list:
    any id resolves when you call it, and the Dev UI lists what the server's
    ``/models`` returns. Several instances, and an ``OpenAI()``, can run side
    by side because each has its own name.

    With ``backend=GoogleCloud(...)``, the plugin serves Model Garden's
    OpenAI-compatible models (Llama, Mistral and others) under ``modelgarden/``,
    authenticated with application default credentials.

    Example:
        ```python
        from genkit import Genkit
        from genkit_openai import GoogleCloud, OpenAI, OpenAICompatible

        # 1. OpenAI's API, OpenRouter under its own name, and Model Garden
        openrouter = OpenAICompatible(name='openrouter', base_url='https://openrouter.ai/api/v1', api_key=key)
        garden = OpenAICompatible(backend=GoogleCloud(project='bistro-prod'))
        ai = Genkit(plugins=[OpenAI(), openrouter, garden])

        # 2. Each server's own ids, sent as written
        res = await ai.generate(model=openrouter.model('meta-llama/llama-3.3-70b-instruct'), prompt='Name a bakery.')
        print(res.text)
        # => Rise & Shine Bakehouse

        # 3. Model Garden refs are modelgarden/<publisher>/<model>
        garden.model('meta/llama-3.3-70b-instruct-maas')
        # => ModelRef(name='modelgarden/meta/llama-3.3-70b-instruct-maas', ...)
        ```
    """

    @overload
    def __init__(
        self,
        *,
        name: str,
        base_url: str,
        api_key: ApiKey | None = None,
        client_options: Mapping[str, Any] | None = None,
        backend: None = None,
    ) -> None: ...
    @overload
    def __init__(
        self,
        *,
        backend: GoogleCloud,
        name: Literal['modelgarden'] | None = None,
        base_url: str | None = None,
        client_options: Mapping[str, Any] | None = None,
    ) -> None: ...
    def __init__(
        self,
        *,
        name: str | None = None,
        base_url: str | None = None,
        api_key: ApiKey | None = None,
        client_options: Mapping[str, Any] | None = None,
        backend: GoogleCloud | None = None,
    ) -> None:
        """Point the plugin at one OpenAI-compatible server, or at Model Garden.

        Args:
            name: The namespace for this server's models and embedders, e.g.
                ``'openrouter'`` gives ``openrouter/<id>``. Required without a
                backend. With ``backend=GoogleCloud(...)`` it's always
                ``'modelgarden'``, so it can be left out.
            base_url: The server's API base URL up to and including ``/v1``,
                e.g. ``https://openrouter.ai/api/v1``; the SDK appends
                ``/chat/completions``. Required without a backend. With
                ``backend=GoogleCloud(...)`` it overrides the Model Garden endpoint.
            api_key: The server's key, or an async function that returns one
                before each request (for a token that expires). ``None`` sends
                no real key, for local servers. ``OPENAI_API_KEY`` is never
                read. Not used with ``backend=GoogleCloud(...)``.
            client_options: Other ``AsyncOpenAI`` settings, such as
                ``timeout``, ``max_retries``, ``default_headers``, or
                ``http_client``.
            backend: ``GoogleCloud(project=..., location=...)`` serves Model
                Garden's OpenAI-compatible models with application default
                credentials. Needs the ``google-cloud`` extra.

        Raises:
            ValueError: ``name`` or ``base_url`` missing without a backend, a
                ``name`` that is empty or contains ``/``, a ``name`` other than
                ``'modelgarden'`` with ``GoogleCloud``, or ``api_key`` with
                ``GoogleCloud``.
            TypeError: ``backend`` isn't a ``GoogleCloud``, or a
                ``client_options`` key ``AsyncOpenAI`` doesn't take.
            ImportError: ``backend=GoogleCloud(...)`` without the ``google-cloud`` extra.
        """
        if backend is None:
            if not isinstance(name, str) or not name or '/' in name:
                raise ValueError(f"OpenAICompatible name must be a non-empty string without '/', got {name!r}")
            if not base_url:
                raise ValueError("OpenAICompatible needs base_url, e.g. 'https://openrouter.ai/api/v1'")
            options = _client_kwargs(
                api_key=api_key, base_url=base_url, client_options=client_options, plugin='OpenAICompatible'
            )
            options.setdefault('api_key', _NO_KEY)
            self._server: _Server = _BaseUrlServer(options)
            self.name = name
        else:
            if not isinstance(backend, GoogleCloud):
                raise TypeError(
                    f'OpenAICompatible backend must be GoogleCloud(...) or None, got {type(backend).__name__}'
                )
            if name is not None and name != MODEL_GARDEN_NAME:
                raise ValueError(
                    f"OpenAICompatible(backend=GoogleCloud()) registers under '{MODEL_GARDEN_NAME}'; "
                    f'drop name={name!r} or pass name={MODEL_GARDEN_NAME!r}'
                )
            if api_key is not None:
                raise ValueError(
                    'OpenAICompatible(backend=GoogleCloud()) authenticates with Google Cloud credentials; drop api_key'
                )
            require_google_auth()
            options = _client_kwargs(
                api_key=None, base_url=None, client_options=client_options, plugin='OpenAICompatible'
            )
            self._server = _ModelGardenServer(backend, base_url=base_url, client_options=options)
            self.name = MODEL_GARDEN_NAME
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

    async def _client_for(self, context: Mapping[str, Any] | None) -> AsyncOpenAI:
        """The server's client for one call, given the caller's context."""
        return await self._server.request_client(context_api_key(context))

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
            client = await self._client_for(ctx.context)
            model = OpenAIModel(model_id, client, supports=info.supports, mistral=is_mistral(client))
            return await OpenAIModelHandler(model).generate(request, ctx)

        return Model(f'{self.name}/{model_id}', _generate, config_schema=OpenAIConfig, info=info)

    def _embedder_action(self, model_id: str) -> Action:
        async def _embed(request: EmbedRequest, ctx: EmbedContext) -> EmbedResponse:
            client = await self._client_for(ctx.context)
            return await embed_texts(client, model_id, request, mistral=is_mistral(client))

        return Embedder(
            f'{self.name}/{model_id}',
            _embed,
            info=EmbedderInfo(label=f'{self.name} - {model_id}', supports=EmbedderSupports(input=['text'])),
        )

    async def list_actions(self) -> list[Action]:
        """What the server lists, as models and embedders.

        Ids containing ``embed`` are listed as embedders, everything else as
        chat models; either can still be called as the other kind. A server
        without ``/models``, or one that rejects the call, lists nothing and
        logs why; that isn't cached, so the next listing tries again. Model
        Garden has no ``/models``, so it lists nothing and any id resolves.
        """
        if self._list_actions_cache is not None:
            return self._list_actions_cache
        try:
            model_ids = await self._server.model_ids()
        except (APIError, GenkitError) as e:
            logger.warning(
                'Could not list models; the Dev UI shows none for this plugin', plugin=self.name, error=str(e)
            )
            return []
        actions = [self._embedder_action(m) if 'embed' in m else self._model_action(m) for m in model_ids]
        self._list_actions_cache = actions
        return actions
