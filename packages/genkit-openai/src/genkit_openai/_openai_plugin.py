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


"""OpenAI OpenAI API Compatible Plugin for Genkit."""

import enum
import inspect
import os
import re
from collections.abc import Mapping
from typing import Any, Literal, TypeAlias, cast

from openai import APIError, AsyncOpenAI
from openai.types import Model as OpenAIModelEntry
from pydantic import BaseModel

from genkit import Embedding, FlowContext, GenkitError, ModelResponse
from genkit.embedder import (
    Embedder,
    EmbedderInfo,
    EmbedderSupports,
    EmbedRequest,
    EmbedResponse,
)
from genkit.model import (
    ConstrainedSupport,
    Model,
    ModelContext,
    ModelInfo,
    ModelRef,
    ModelRequest,
    ModelSupports,
)
from genkit.plugin_api import (
    Action,
    ActionKind,
    Plugin,
    loop_local_client,
)
from genkit_openai._models._audio import (
    SUPPORTED_STT_MODELS,
    SUPPORTED_TTS_MODELS,
    OpenAISttConfig,
    OpenAISTTModel,
    OpenAITtsConfig,
    OpenAITTSModel,
)
from genkit_openai._models._handler import OpenAIModelHandler
from genkit_openai._models._image import (
    SUPPORTED_IMAGE_MODELS,
    OpenAIDalleConfig,
    OpenAIGptImageConfig,
    OpenAIImageModel,
)
from genkit_openai._models._model import OpenAIModel
from genkit_openai._models._model_info import (
    GPT_REASONING_MODEL_SUPPORTS,
    MULTIMODAL_MODEL_SUPPORTS,
    O_SERIES_MODEL_SUPPORTS,
    SUPPORTED_EMBEDDING_MODELS,
    SUPPORTED_OPENAI_MODELS,
    KnownGpt,
    get_default_openai_model_info,
)
from genkit_openai._models._utils import reraise_openai_error
from genkit_openai._secrets import context_api_key
from genkit_openai._typing import OpenAIConfig

# Headers that tie a call to the plugin's OpenAI organization or project. A
# tenant key runs under the tenant's own account, so these are not copied.
_PLUGIN_ACCOUNT_HEADERS = frozenset({'openai-organization', 'openai-project'})

_CLIENT_OPTION_KEYS = frozenset(inspect.signature(AsyncOpenAI.__init__).parameters) - {'self'}


def _client_kwargs(
    *, api_key: str | None, base_url: str | None, client_options: Mapping[str, Any] | None
) -> dict[str, Any]:
    """Merge the typed settings into ``client_options`` for ``AsyncOpenAI``.

    Raises:
        TypeError: A ``client_options`` key ``AsyncOpenAI`` doesn't take, or
            ``api_key``/``base_url`` given both as an argument and in ``client_options``.
    """
    options = dict(client_options or {})
    for key in options:
        if key not in _CLIENT_OPTION_KEYS:
            raise TypeError(
                f'OpenAI got an unexpected client_options key {key!r}; '
                f'AsyncOpenAI accepts {", ".join(sorted(k for k in _CLIENT_OPTION_KEYS if not k.startswith("_")))}'
            )
    for name, value in (('api_key', api_key), ('base_url', base_url)):
        if value is None:
            continue
        if name in options:
            raise TypeError(f'OpenAI got {name} both as an argument and in client_options; pass it once')
        options[name] = value
    return options


def _missing_key_error() -> GenkitError:
    return GenkitError(
        status='FAILED_PRECONDITION',
        message=(
            'OpenAI needs an API key: set OPENAI_API_KEY or pass OpenAI(api_key=...), '
            "or send a per-request key as context={'secrets': {'api_key': ...}}."
        ),
    )


def open_ai_name(name: str) -> str:
    """Create an OpenAI action name.

    Args:
        name: Base name for the action.

    Returns:
        The fully qualified OpenAI action name.
    """
    return f'openai/{name}'


# Only this plugin's namespace. A vertexai/ or azure/ paste is a different
# name — remapping it here would send the request to the OpenAI API.
_STRIP_PREFIXES = ('openai/',)


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


class _ModelType(enum.Enum):
    """Classification of OpenAI model types based on name patterns."""

    EMBEDDER = 'embedder'
    IMAGE = 'image'
    TTS = 'tts'
    STT = 'stt'
    CHAT = 'chat'


def _is_gpt_image(name: str) -> bool:
    return 'gpt-image' in name or 'chatgpt-image' in name


def _classify_model(name: str) -> _ModelType:
    """Classify a model name into its type based on name patterns.

    Centralizes the name-matching logic used by both resolve() and
    list_actions() to avoid inconsistencies.

    Args:
        name: The model name (with or without 'openai/' prefix).

    Returns:
        The classified model type.
    """
    if 'embed' in name:
        return _ModelType.EMBEDDER
    if _is_gpt_image(name) or 'dall-e' in name:
        return _ModelType.IMAGE
    if 'tts' in name:
        return _ModelType.TTS
    if 'whisper' in name or 'transcribe' in name:
        return _ModelType.STT
    return _ModelType.CHAT


_DATED_SUFFIX = re.compile(r'-\d{4}-\d{2}-\d{2}$')


def _undated(name: str) -> str:
    """The alias of a dated snapshot id such as ``gpt-5-2025-08-07``, or the id unchanged."""
    return _DATED_SUFFIX.sub('', name)


_UNSUPPORTED_MODEL_PATTERN = re.compile(
    r'realtime|moderation|codex|-pro\b|-cyber\b|daybreak|rosalind|diarize'
    r'|^gpt-live-|^gpt-audio'
    r'|^(ft:)?(ada|babbage|curie|davinci)\b'
)
_LISTED_EXCEPTIONS = frozenset({'gpt-6-cyber'})


def _is_known_unsupported(name: str) -> bool:
    """Report whether discovery hides a model id because no action in this plugin can serve it."""
    if _undated(name) in _LISTED_EXCEPTIONS:
        return False
    return _UNSUPPORTED_MODEL_PATTERN.search(name) is not None


def _is_listed(name: str, model_type: _ModelType) -> bool:
    """Report whether discovery lists a model id."""
    if _is_known_unsupported(name):
        return False
    # DALL-E ids are shut down.
    return model_type != _ModelType.IMAGE or _is_gpt_image(name)


# Default ModelSupports for each multimodal model type, used as fallback when
# a model is not found in the registry.
_DEFAULT_SUPPORTS: dict[_ModelType, ModelSupports] = {
    _ModelType.IMAGE: ModelSupports(
        media=False,
        output=['media'],
        multiturn=False,
        system_role=False,
        tools=False,
    ),
    _ModelType.TTS: ModelSupports(
        media=False,
        output=['media'],
        multiturn=False,
        system_role=False,
        tools=False,
    ),
    _ModelType.STT: ModelSupports(
        media=True,
        output=['text', 'json'],
        multiturn=False,
        system_role=False,
        tools=False,
    ),
}

# Type alias for multimodal model classes.
_MultimodalModel: TypeAlias = OpenAIImageModel | OpenAITTSModel | OpenAISTTModel
_MultimodalModelConfig: TypeAlias = tuple[type[_MultimodalModel], dict[str, ModelInfo]]

# Maps multimodal model types to their class and registry.
_MULTIMODAL_CONFIG: dict[_ModelType, _MultimodalModelConfig] = {
    _ModelType.IMAGE: (OpenAIImageModel, SUPPORTED_IMAGE_MODELS),
    _ModelType.TTS: (OpenAITTSModel, SUPPORTED_TTS_MODELS),
    _ModelType.STT: (OpenAISTTModel, SUPPORTED_STT_MODELS),
}


def _get_multimodal_info(
    name: str,
    model_type: _ModelType,
    supported_models: dict[str, ModelInfo],
) -> tuple[ModelInfo, type[BaseModel]]:
    """Build the ModelInfo for a multimodal model.

    Uses registry metadata when available, falls back to default supports.

    Args:
        name: The raw model name (without the 'openai/' prefix).
        model_type: The classified model type for default supports fallback.
        supported_models: Registry of known models and their metadata.

    Returns:
        A tuple containing the ModelInfo and the config class this
        endpoint checks against.
    """
    schema = multimodal_config_class(name=name, model_type=model_type)
    if model_info := supported_models.get(name):
        return model_info, schema
    if model_info := supported_models.get(_undated(name)):
        return model_info.model_copy(update={'label': f'OpenAI - {name}'}), schema

    return ModelInfo(label=f'OpenAI - {name}', supports=_DEFAULT_SUPPORTS.get(model_type)), schema


def multimodal_config_class(*, name: str, model_type: _ModelType) -> type[BaseModel]:
    """The config class an image, TTS, or STT model checks generate() against."""
    if model_type == _ModelType.TTS:
        return OpenAITtsConfig
    if model_type == _ModelType.STT:
        return OpenAISttConfig
    if _is_gpt_image(name):
        return OpenAIGptImageConfig
    return OpenAIDalleConfig


_GPT_4_TURBO_SUPPORTS = ModelSupports(multiturn=True, media=True, tools=True, system_role=True, output=['json', 'text'])
_GPT_4_SUPPORTS = ModelSupports(multiturn=True, media=False, tools=True, system_role=True, output=['text'])
_GPT_TEXT_JSON_SUPPORTS = ModelSupports(
    multiturn=True, media=False, tools=True, system_role=True, output=['json', 'text']
)

# Shut-down chat ids keep their last supports for compatible hosts that still
# serve them, but init() does not list them.
_RETIRED_CHAT_MODEL_SUPPORTS: dict[str, ModelSupports] = {
    'gpt-4o-2024-05-13': MULTIMODAL_MODEL_SUPPORTS,
    'gpt-4.5-preview': MULTIMODAL_MODEL_SUPPORTS,
    'gpt-4-turbo': _GPT_4_TURBO_SUPPORTS,
    'gpt-4-turbo-2024-04-09': _GPT_4_TURBO_SUPPORTS,
    'gpt-4-turbo-preview': _GPT_4_TURBO_SUPPORTS,
    'gpt-4-0125-preview': _GPT_4_TURBO_SUPPORTS,
    'gpt-4-1106-preview': _GPT_4_TURBO_SUPPORTS,
    'gpt-4': _GPT_4_SUPPORTS,
    'gpt-4-0613': _GPT_4_SUPPORTS,
    'gpt-3.5-turbo': _GPT_TEXT_JSON_SUPPORTS,
    'gpt-3.5-turbo-0125': _GPT_TEXT_JSON_SUPPORTS,
    'gpt-3.5-turbo-1106': _GPT_TEXT_JSON_SUPPORTS,
    'o1': O_SERIES_MODEL_SUPPORTS,
    'o3-mini': ModelSupports(
        multiturn=True,
        media=False,
        tools=True,
        system_role=False,
        output=['json', 'text'],
        constrained=ConstrainedSupport.ALL,
    ),
    'o4-mini': O_SERIES_MODEL_SUPPORTS,
    'gpt-5-chat-latest': ModelSupports(multiturn=True, media=True, tools=False, system_role=True, output=['text']),
    'gpt-5.2-chat': GPT_REASONING_MODEL_SUPPORTS,
    'gpt-oss-120b': _GPT_TEXT_JSON_SUPPORTS,
    'gpt-oss-20b': _GPT_TEXT_JSON_SUPPORTS,
}


def _known_chat_model_info(name: str) -> ModelInfo | None:
    if info := SUPPORTED_OPENAI_MODELS.get(cast(KnownGpt, name)):
        return info
    if supports := _RETIRED_CHAT_MODEL_SUPPORTS.get(name):
        return ModelInfo(label=f'OpenAI - {name}', supports=supports)
    base = name.split(':')[1] if name.startswith('ft:') else _undated(name)
    if base in (name, '') or (info := _known_chat_model_info(base)) is None:
        return None
    return info.model_copy(update={'label': f'OpenAI - {name}'})


def _chat_model_info(name: str) -> ModelInfo:
    """The ModelInfo for a chat model id, or generic info for an id the plugin does not know.

    A dated snapshot takes its alias's info and a fine-tune ``ft:<base>:...``
    takes its base model's, stage included. Shut-down snapshots keep their
    retired supports instead.
    """
    return _known_chat_model_info(name) or get_default_openai_model_info(name)


def default_openai_metadata(name: str) -> dict[str, Any]:
    return {
        'model': {'label': f'OpenAI - {name}', 'supports': {'multiturn': True}},
    }


class OpenAI(Plugin):
    """A plugin for integrating OpenAI compatible models with the Genkit framework.

    This class registers OpenAI model handlers within a registry, allowing
    interaction with supported OpenAI models.
    """

    name = 'openai'

    @classmethod
    def gpt(cls, name: KnownGpt | str, *, config: OpenAIConfig | None = None) -> ModelRef[OpenAIConfig]:
        """Typed ref for an OpenAI chat model, e.g. ``OpenAI.gpt('gpt-5.5')``.

        Chat only: image, TTS, STT, and embedding ids validate different
        request shapes, so binding OpenAIConfig to them would let chat-only
        keys like frequency_penalty ride into the wrong endpoint.
        """
        # str(None) is 'None', and that classifies as chat, so a non-string
        # would mint a real-looking ref instead of failing here.
        if not isinstance(name, str):
            raise GenkitError(status='INVALID_ARGUMENT', message='OpenAI.gpt: model name must be a string.')
        local = _strip_ref_prefixes(name)
        if not local:
            raise GenkitError(status='INVALID_ARGUMENT', message='OpenAI.gpt: model name is required.')
        model_type = _classify_model(local)
        if model_type != _ModelType.CHAT:
            kind = {
                _ModelType.EMBEDDER: 'an embedder',
                _ModelType.IMAGE: 'an image model',
                _ModelType.TTS: 'a tts model',
                _ModelType.STT: 'an stt model',
            }[model_type]
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message=(
                    f"OpenAI.gpt: '{local}' is {kind}; it does not take "
                    f"OpenAIConfig. Pass it as a string ('openai/{local}')."
                ),
            )
        return ModelRef(name=f'openai/{local}', config_schema=OpenAIConfig, config=config)

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str | None = None,
        client_options: Mapping[str, Any] | None = None,
    ) -> None:
        """Initializes the OpenAI plugin.

        Args:
            api_key: OpenAI API key. Defaults to ``OPENAI_API_KEY``.
            base_url: OpenAI API base URL up to and including ``/v1``, e.g.
                ``https://openrouter.ai/api/v1``; the SDK appends
                ``/chat/completions``. Defaults to ``OPENAI_BASE_URL``, then the
                public API.
            client_options: Other ``AsyncOpenAI`` settings, such as
                ``organization``, ``project``, ``timeout``, ``max_retries``,
                ``default_headers``, or ``http_client``.

        Raises:
            TypeError: If ``client_options`` has a key ``AsyncOpenAI`` doesn't
                take, or repeats ``api_key`` or ``base_url``.
        """
        options = _client_kwargs(api_key=api_key, base_url=base_url, client_options=client_options)
        self._options = options
        self._plugin_api_key: str | None = options.get('api_key')
        plugin_headers: dict[str, str] = dict(options.get('default_headers') or {})
        self._tenant_headers = {k: v for k, v in plugin_headers.items() if k.lower() not in _PLUGIN_ACCOUNT_HEADERS}
        self._pins_authorization = any(k.lower() == 'authorization' for k in plugin_headers)
        self._list_actions_cache: list[Action] | None = None

    @loop_local_client
    def _runtime_client(self) -> AsyncOpenAI:
        return AsyncOpenAI(**self._options)

    @loop_local_client
    def _tenant_only_client(self) -> AsyncOpenAI:
        # Only used when the plugin has no key of its own. Its placeholder key is
        # never sent: every call through it swaps in the caller's key first.
        params: dict[str, Any] = {**self._options, 'api_key': 'unset'}
        return AsyncOpenAI(**params)

    def _has_plugin_key(self) -> bool:
        return bool(self._plugin_api_key or os.environ.get('OPENAI_API_KEY'))

    async def init(self) -> list[Action]:
        """Initialize plugin.

        Returns:
            Actions for built-in OpenAI models, embedders, image, TTS, and STT.
        """
        actions = []

        # Add known chat models.
        for name in SUPPORTED_OPENAI_MODELS:
            actions.append(self._create_model_action(name))

        # Add known embedders.
        for name in SUPPORTED_EMBEDDING_MODELS:
            actions.append(self._create_embedder_action(name))

        # Add multimodal models (Image, TTS, STT).
        for model_type, (model_class, supported_models) in _MULTIMODAL_CONFIG.items():
            for name in supported_models:
                actions.append(
                    self._create_multimodal_action(
                        name,
                        model_class,
                        supported_models,
                        model_type,
                    )
                )

        return actions

    def get_model_info(self, name: str) -> dict[str, Any] | None:
        """Retrieves metadata and supported features for the specified model.

        This method looks up the model's information from a predefined list
        of supported OpenAI-compatible models or provides default information.

        Returns:
            A dictionary containing the model's 'label' and 'supports'
            features, or None if no information can be found (though typically,
            a default is provided). The 'supports' key contains a dictionary representing
            the model's capabilities (e.g., tools, streaming).
        """
        info = _chat_model_info(name)
        supports = info.supports.model_dump(by_alias=True, exclude_none=True) if info.supports else {}
        return {
            'label': info.label,
            'supports': supports,
        }

    async def resolve(self, action_type: ActionKind, name: str) -> Action | None:
        """Resolve an action by creating and returning an Action object.

        Uses name-based pattern matching (mirroring JS implementation) to
        route to the correct model type: image, TTS, STT, embedder, or chat.

        Args:
            action_type: The kind of action to resolve.
            name: The id without the ``openai/`` prefix.

        Returns:
            Action object if found, None otherwise.
        """
        if action_type == ActionKind.EMBEDDER:
            if _classify_model(name) != _ModelType.EMBEDDER:
                return None
            return self._create_embedder_action(name)

        if action_type == ActionKind.MODEL:
            model_type = _classify_model(name)
            if model_type == _ModelType.EMBEDDER:
                return None  # Embedders should not be resolved as models.
            if model_type in _MULTIMODAL_CONFIG:
                model_class, supported_models = _MULTIMODAL_CONFIG[model_type]
                return self._create_multimodal_action(name, model_class, supported_models, model_type)
            return self._create_model_action(name)

        return None

    def _client_for_call(self, request: ModelRequest, ctx: ModelContext) -> AsyncOpenAI:
        """The plugin's client, or a copy carrying the caller's ``context.secrets`` key.

        A tenant who passed their own key is the one who should be billed. The
        copy shares the plugin client's connection pool, and concurrent tenants
        each get their own copy instead of swapping the key on a shared client.
        A plugin built without a key serves only callers who bring one.
        """
        return self._client_for_context(ctx.context)

    def _client_for_context(self, context: dict[str, Any]) -> AsyncOpenAI:
        key = context_api_key(context)
        if self._has_plugin_key():
            client = self._runtime_client()
            return self._with_tenant_key(client, key) if key else client
        if key is None:
            raise _missing_key_error()
        return self._with_tenant_key(self._tenant_only_client(), key)

    def _with_tenant_key(self, client: AsyncOpenAI, key: str) -> AsyncOpenAI:
        """A copy of ``client`` that authenticates as the tenant and nothing else.

        The tenant's key picks its own organization and project, so the
        plugin's ``organization``, ``project`` (including ``OPENAI_ORG_ID`` and
        ``OPENAI_PROJECT_ID``) and the matching ``default_headers`` are left
        off. Sent with a foreign key they fail with 401, or bill the plugin's
        project when the tenant is a member. ``base_url``, timeouts, retries
        and other headers carry over.
        """
        if self._pins_authorization:
            raise GenkitError(
                status='FAILED_PRECONDITION',
                message=(
                    "OpenAI(client_options={'default_headers': ...}) pins an Authorization header, which would replace "
                    'the context.secrets key. Drop that header to serve per-request keys.'
                ),
            )
        tenant = client.with_options(api_key=key, set_default_headers=self._tenant_headers)
        tenant.organization = None
        tenant.project = None
        return tenant

    def _embed_client(self, context: dict[str, Any]) -> AsyncOpenAI:
        """Same key order as models: ``context.secrets``, then the plugin key.

        ``Genkit.embed(context=...)`` carries the tenant key here.
        """
        return self._client_for_context(context)

    def _create_model_action(self, name: str) -> Action:
        """Create an Action object for an OpenAI model.

        Args:
            name: The model id as received (no plugin-prefix stripping).

        Returns:
            Action object for the model.
        """
        info = _chat_model_info(name)

        async def _generate(request: ModelRequest[OpenAIConfig], ctx: ModelContext) -> ModelResponse:
            openai_model = OpenAIModelHandler(
                OpenAIModel(name, self._client_for_call(request, ctx), supports=info.supports)
            )
            return await openai_model.generate(request, ctx)

        return Model(
            open_ai_name(name),
            _generate,
            config_schema=OpenAIConfig,
            info=info,
        )

    def _create_multimodal_action(
        self,
        name: str,
        model_class: type[_MultimodalModel],
        supported_models: dict[str, ModelInfo],
        model_type: _ModelType,
    ) -> Action:
        """Create an Action for a multimodal model (image, TTS, or STT).

        Args:
            name: The model id as received (no plugin-prefix stripping).
            model_class: The model class to instantiate.
            supported_models: Registry of known models and their metadata.
            model_type: The classified model type for default metadata fallback.

        Returns:
            Action object for the model.
        """
        info, config_schema = _get_multimodal_info(name, model_type, supported_models)

        async def _generate(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
            model_instance = model_class(name, self._client_for_call(request, ctx))
            return await model_instance.generate(request, ctx)

        return Model(
            open_ai_name(name),
            _generate,
            config_schema=config_schema,
            info=info,
        )

    def _create_embedder_action(self, name: str) -> Action:
        """Create an Action object for an OpenAI embedder.

        Args:
            name: The embedder id as received (no plugin-prefix stripping).

        Returns:
            Action object for the embedder.
        """
        embedder_info = SUPPORTED_EMBEDDING_MODELS.get(
            name,
            {
                'label': f'OpenAI Embedding - {name}',
                'dimensions': 1536,
                'supports': {'input': ['text']},
            },
        )

        async def embed_fn(request: EmbedRequest, ctx: FlowContext[Any]) -> EmbedResponse:
            """Embedder function that calls OpenAI embeddings API."""
            # Extract text from document content
            texts = []
            for doc in request.input:
                doc_text = ''.join(  # type: ignore[arg-type]
                    part.text for part in doc.content if part.text is not None and part.text
                )
                texts.append(doc_text)

            # Get optional parameters (omit when None; OpenAI create() uses Omit, not None)
            dimensions: int | None = None
            encoding_format: Literal['float'] | None = None
            dim_val = request.options.get('dimensions')
            if dim_val is not None:
                # bool is an int subclass, so True would otherwise pass as 1.
                if not isinstance(dim_val, int) or isinstance(dim_val, bool):
                    raise GenkitError(
                        status='INVALID_ARGUMENT',
                        message=f'dimensions must be an int, got {dim_val!r}',
                    )
                dimensions = dim_val
            # 'base64' is deliberately not forwarded: the SDK sends base64 either
            # way and only decodes the response when it was not asked explicitly.
            if request.options.get('encodingFormat') == 'float':
                encoding_format = 'float'

            # Call with only non-None optional params to satisfy strict typings
            client = self._embed_client(ctx.context)
            try:
                if dimensions is not None and encoding_format is not None:
                    response = await client.embeddings.create(
                        model=name,
                        input=texts,
                        dimensions=dimensions,
                        encoding_format=encoding_format,
                    )
                elif dimensions is not None:
                    response = await client.embeddings.create(
                        model=name,
                        input=texts,
                        dimensions=dimensions,
                    )
                elif encoding_format is not None:
                    response = await client.embeddings.create(
                        model=name,
                        input=texts,
                        encoding_format=encoding_format,
                    )
                else:
                    response = await client.embeddings.create(
                        model=name,
                        input=texts,
                    )
            except APIError as e:
                reraise_openai_error(e)

            # Convert OpenAI response to Genkit format
            embeddings = [Embedding(embedding=item.embedding) for item in response.data]
            return EmbedResponse(embeddings=embeddings)

        return Embedder(
            open_ai_name(name),
            embed_fn,
            info=EmbedderInfo(
                label=embedder_info['label'],
                supports=EmbedderSupports(input=embedder_info['supports']['input']),
                dimensions=embedder_info.get('dimensions'),
            ),
        )

    async def list_actions(self) -> list[Action]:
        """Generate a list of available actions or models.

        Uses pattern matching on model names (mirroring the JS implementation)
        to categorize models as embedders, image generators, TTS, STT, or chat.

        Returns:
            list[Action]: A list of Action objects. Empty for a
            plugin without its own key: listing models needs one, and the
            built-in catalog ``init`` registers is already in the Dev UI.
        """
        if not self._has_plugin_key():
            return []
        if self._list_actions_cache is not None:
            return self._list_actions_cache

        actions: list[Action] = []
        try:
            models_ = await self._runtime_client().models.list()
        except APIError as e:
            reraise_openai_error(e)
        models: list[OpenAIModelEntry] = models_.data
        for model in models:
            name = model.id
            model_type = _classify_model(name)
            if not _is_listed(name, model_type):
                continue
            if model_type == _ModelType.EMBEDDER:
                actions.append(self._create_embedder_action(name))
            elif model_type in _DEFAULT_SUPPORTS:
                model_class, supported_models = _MULTIMODAL_CONFIG[model_type]
                actions.append(self._create_multimodal_action(name, model_class, supported_models, model_type))
            else:
                actions.append(self._create_model_action(name))
        self._list_actions_cache = actions
        return actions


__all__ = ['OpenAI']
