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

"""Wire types for the Gemini Interactions API (steps schema).

Mirrors js/plugins/google-genai/src/common/interaction-types.ts: same type
names, same order. Keep the two in step: change JS first, then translate.

How the JS types map to Python:

- Each type is a TypedDict: the wire JSON itself. The plugin builds them as
  dict literals and sends them as they are. ``extra_items=object`` keeps
  fields not listed here, so new server fields pass through and are sent
  back unchanged.
- Responses are validated against these TypedDicts with pydantic
  (``parse_interaction``, ``parse_sse_event``). Nothing is filled in: a key
  is there only when the server sent it.
- Only ``type`` (``event_type`` for events) is required. JS doesn't check
  responses at runtime, so a field the server leaves out is ``undefined``
  there; here a required field would fail the whole response. The plugin's
  own code still sets every field the JS types require.
- Steps, content, text annotations, stream deltas and SSE events are open
  unions tagged by ``type`` (``event_type``): an unknown tag, or a known tag
  whose fields don't match its type, validates as ``UnknownStep``,
  ``UnknownContent``, ... These are ``dict`` subclasses holding the object
  as sent (not TypedDicts, which type checkers can't tell apart from the
  known types). Check ``isinstance(step, UnknownStep)`` first; after that,
  comparing ``type`` narrows to one type (pyright, ty and pyrefly).
- JS open string unions (``'a' | 'b' | (string & {})``) are ``str``, with
  the known values in a comment.
- JS inline object types get a name here (e.g. ``QueriesArguments`` and the
  SSE event types).

Python only:

- ``ModelOutputStep.error``: the spec's step error, read by the converters.
- ``ApiErrorBody``: the body of a failed HTTP call (JS checks it inline).

Not ported yet (translate from the JS file when the feature arrives):

- ``Content`` as a ``Step`` (input given as plain content blocks): the
  plugin sends and reads steps only.
- InteractionToolChoiceConfig, SpeechConfig, ImageConfig, VideoConfig,
  ModelGenerationConfig, the agent configs, ServiceTier,
  InteractionSafetySetting and CreateInteractionRequest: the create body is
  still built as a plain dict; they come with the request mapping for
  Gemini models.
- InteractionStreamResult: comes with the streaming client.
"""

from __future__ import annotations

from typing import Annotated, Literal, TypeAlias, TypeVar

from pydantic import (
    Discriminator,
    PlainValidator,
    Tag,
    TypeAdapter,
    ValidationError,
    ValidatorFunctionWrapHandler,
    WrapValidator,
)
from typing_extensions import NotRequired, TypedDict

# ---------------------------------------------------------------------------
# Open unions
#
# A union tagged by `type` (or `event_type`) that also takes tags it doesn't
# know:
#   Annotated[
#       Annotated[Known, Tag(KNOWN_TAG)]
#       | Annotated[UnknownX, keep_as(UnknownX, key), Tag(UNKNOWN_TAG)],
#       open_union_tag(key, KNOWN_TYPES),
#       keep_unknown(UnknownX, key),
#   ]
# A known tag is validated against its type; an unknown tag, or a known tag
# whose fields don't match, is kept unchanged as an `UnknownX` instance.
# ---------------------------------------------------------------------------

KNOWN_TAG = 'known'
UNKNOWN_TAG = 'unknown'


class UnknownObject(dict[str, object]):
    """Base of the ``Unknown*`` types: a wire object kept exactly as sent."""


_U = TypeVar('_U', bound=UnknownObject)


def tag_of(value: object, key: str) -> str | None:
    """The string under ``key`` when ``value`` is a dict (wire JSON), else None."""
    if isinstance(value, dict):
        # Read through items(): after isinstance on an `object`, type checkers
        # don't know the key type, so `value.get(key)` doesn't type-check.
        for name, item in value.items():
            if name == key:
                return item if isinstance(item, str) else None
    return None


def open_union_tag(key: str, known: frozenset[str]) -> Discriminator:
    """Pick the known or the unknown branch of an open union by the value's tag."""

    def tag(value: object) -> str:
        return KNOWN_TAG if tag_of(value, key) in known else UNKNOWN_TAG

    return Discriminator(tag)


def as_unknown(cls: type[_U], key: str, value: object) -> _U:
    """``value`` (an object with a string ``key``) as an ``Unknown*`` instance; raises ValueError otherwise."""
    if tag_of(value, key) is None or not isinstance(value, dict):
        raise ValueError(f"expected an object with a string '{key}'")
    return cls({str(name): item for name, item in value.items()})


def keep_as(cls: type[UnknownObject], key: str) -> PlainValidator:
    """Validator of the unknown branch: keep the object as sent, as a ``cls`` instance."""
    return PlainValidator(lambda value: as_unknown(cls, key, value))


def keep_unknown(cls: type[UnknownObject], key: str) -> WrapValidator:
    """Keep an object with a known tag whose fields don't match its type, as a ``cls`` instance."""

    def validate(value: object, handler: ValidatorFunctionWrapHandler) -> object:
        try:
            return handler(value)
        except ValidationError:
            if tag_of(value, key) is not None:
                return as_unknown(cls, key, value)
            raise

    return WrapValidator(validate)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


class InteractionFunctionTool(TypedDict, extra_items=object):
    """A function the model can call (https://ai.google.dev/gemini-api/docs/function-calling)."""

    type: Literal['function']
    # The name of the function to call.
    name: NotRequired[str]
    # Description and purpose of the function.
    description: NotRequired[str]
    # The function's parameters, as a JSON schema.
    parameters: NotRequired[dict[str, object]]


class InteractionGoogleSearchTool(TypedDict, extra_items=object):
    """Lets the model search Google (https://ai.google.dev/gemini-api/docs/grounding)."""

    type: Literal['google_search']
    # `web_search`, `image_search` or `enterprise_web_search`; unknown search
    # types are passed through for forward compatibility.
    search_types: NotRequired[list[str]]


class InteractionCodeExecutionTool(TypedDict, extra_items=object):
    """Lets the model run code (https://ai.google.dev/gemini-api/docs/code-execution)."""

    type: Literal['code_execution']


class InteractionUrlContextTool(TypedDict, extra_items=object):
    """Lets the model read and summarize web page content."""

    type: Literal['url_context']


class InteractionFileSearchTool(TypedDict, extra_items=object):
    """Lets the model search uploaded document corpora."""

    type: Literal['file_search']
    file_search_store_names: NotRequired[list[str]]
    metadata_filter: NotRequired[str]
    top_k: NotRequired[int]


class InteractionAllowedTools(TypedDict, extra_items=object):
    """The tools an MCP server tool may use."""

    mode: NotRequired[str]
    tools: NotRequired[list[str]]


class InteractionMcpServerTool(TypedDict, extra_items=object):
    """Lets the model connect to remote MCP servers."""

    type: Literal['mcp_server']
    name: NotRequired[str]
    url: NotRequired[str]
    headers: NotRequired[dict[str, str]]
    allowed_tools: NotRequired[list[InteractionAllowedTools]]


class InteractionGoogleMapsTool(TypedDict, extra_items=object):
    """Lets the model search Google Maps."""

    type: Literal['google_maps']
    enable_widget: NotRequired[bool]
    latitude: NotRequired[float]
    longitude: NotRequired[float]


class InteractionComputerUseTool(TypedDict, extra_items=object):
    """Lets the model use a computer."""

    type: Literal['computer_use']
    disabled_safety_policies: NotRequired[list[str]]
    enable_prompt_injection_detection: NotRequired[bool]
    environment: NotRequired[str]
    excluded_predefined_functions: NotRequired[list[str]]


class InteractionRetrievalTool(TypedDict, extra_items=object):
    """Lets the model retrieve from external search or RAG stores."""

    type: Literal['retrieval']
    exa_ai_search_config: NotRequired[dict[str, object]]
    parallel_ai_search_config: NotRequired[dict[str, object]]
    rag_store_config: NotRequired[dict[str, object]]
    retrieval_types: NotRequired[list[str]]
    vertex_ai_search_config: NotRequired[dict[str, object]]


class InteractionDynamicTool(TypedDict, extra_items=object):
    """Any other tool, sent as given."""

    type: str


InteractionTool: TypeAlias = (
    InteractionFunctionTool
    | InteractionGoogleSearchTool
    | InteractionCodeExecutionTool
    | InteractionUrlContextTool
    | InteractionFileSearchTool
    | InteractionMcpServerTool
    | InteractionGoogleMapsTool
    | InteractionComputerUseTool
    | InteractionRetrievalTool
    | InteractionDynamicTool
)
"""A tool that can be used by the model."""

# ---------------------------------------------------------------------------
# Text annotations
# ---------------------------------------------------------------------------


class UrlCitation(TypedDict, extra_items=object):
    """A URL citation annotation."""

    type: Literal['url_citation']
    # Start of the annotated segment (in bytes).
    start_index: NotRequired[int]
    # End of the annotated segment, exclusive.
    end_index: NotRequired[int]
    url: NotRequired[str]
    title: NotRequired[str]


class FileCitation(TypedDict, extra_items=object):
    """A file citation annotation."""

    type: Literal['file_citation']
    start_index: NotRequired[int]
    end_index: NotRequired[int]
    document_uri: NotRequired[str]
    file_name: NotRequired[str]
    # Source attributed for a portion of the text.
    source: NotRequired[str]
    page_number: NotRequired[int]
    # Media ID in case of image citations.
    media_id: NotRequired[str]
    # User-provided metadata about the retrieved context.
    custom_metadata: NotRequired[dict[str, object]]


class ReviewSnippet(TypedDict, extra_items=object):
    """One review snippet of a place citation (inline in JS)."""

    review_id: NotRequired[str]
    title: NotRequired[str]
    url: NotRequired[str]


class PlaceCitation(TypedDict, extra_items=object):
    """A place (Google Maps) citation annotation."""

    type: Literal['place_citation']
    start_index: NotRequired[int]
    end_index: NotRequired[int]
    name: NotRequired[str]
    # In `places/{place_id}` format.
    place_id: NotRequired[str]
    url: NotRequired[str]
    review_snippets: NotRequired[list[ReviewSnippet]]


class SpeechAnnotation(TypedDict, extra_items=object):
    """Assigns a TTS input turn to a speaker and sets its delivery style."""

    type: Literal['speech_metadata']
    start_index: NotRequired[int]
    end_index: NotRequired[int]
    # The speaker to associate with this turn.
    speaker: NotRequired[str]
    # Style instruction for the speech synthesis.
    style: NotRequired[str]


class WordInfo(TypedDict, extra_items=object):
    """Word-level speech recognition annotation for transcription output."""

    type: Literal['word_info']
    start_index: NotRequired[int]
    end_index: NotRequired[int]
    text: NotRequired[str]
    # Offsets in time relative to the start of the audio.
    start_offset: NotRequired[str]
    end_offset: NotRequired[str]
    # Speaker label for this word (e.g. "spk_1").
    speaker: NotRequired[str]


class UnknownTextAnnotation(UnknownObject):
    """A text annotation type this plugin doesn't know yet (or doesn't match its type), kept as sent."""


TEXT_ANNOTATION_TYPES = frozenset({'url_citation', 'file_citation', 'place_citation', 'speech_metadata', 'word_info'})

KnownTextAnnotation: TypeAlias = Annotated[
    UrlCitation | FileCitation | PlaceCitation | SpeechAnnotation | WordInfo,
    Discriminator('type'),
]
TextAnnotation: TypeAlias = Annotated[
    Annotated[KnownTextAnnotation, Tag(KNOWN_TAG)]
    | Annotated[UnknownTextAnnotation, keep_as(UnknownTextAnnotation, 'type'), Tag(UNKNOWN_TAG)],
    open_union_tag('type', TEXT_ANNOTATION_TYPES),
    keep_unknown(UnknownTextAnnotation, 'type'),
]
"""An annotation on text content, tagged by ``type``."""

# ---------------------------------------------------------------------------
# Content
# ---------------------------------------------------------------------------


class TextContent(TypedDict, extra_items=object):
    """A text content block."""

    type: Literal['text']
    text: NotRequired[str]
    # Citations on model output, or speech metadata on TTS input.
    annotations: NotRequired[list[TextAnnotation]]


class ImageContent(TypedDict, extra_items=object):
    """An image content block."""

    type: Literal['image']
    # Base64 image bytes.
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]
    # MediaResolution: `low`, `medium`, `high` or `ultra_high`.
    resolution: NotRequired[str]


class AudioContent(TypedDict, extra_items=object):
    """An audio content block."""

    type: Literal['audio']
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]
    # Sent on streamed audio (copied onto the content when the stream is
    # assembled), e.g. single-voice TTS, which has no `mime_type`.
    sample_rate: NotRequired[int]
    channels: NotRequired[int]


class DocumentContent(TypedDict, extra_items=object):
    """A document content block."""

    type: Literal['document']
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]


class VideoContent(TypedDict, extra_items=object):
    """A video content block."""

    type: Literal['video']
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]
    # MediaResolution: `low`, `medium`, `high` or `ultra_high`.
    resolution: NotRequired[str]
    # VideoProcessing: `static` (frames at 1 FPS, the default) or `agentic`
    # (the model explores the timeline and loads only what it needs).
    processing: NotRequired[str]


class ThoughtContent(TypedDict, extra_items=object):
    """A thought content block."""

    type: Literal['thought']
    # Signature to match the backend source to be part of the generation.
    signature: NotRequired[str]
    # JS: text and image only. Any content is accepted here, so another type
    # doesn't turn the whole step into an unknown one.
    summary: NotRequired[list[Content]]


class FunctionCallContent(TypedDict, extra_items=object):
    """A function tool call content block."""

    type: Literal['function_call']
    name: NotRequired[str]
    arguments: NotRequired[dict[str, object]]
    # A unique ID for this specific tool call.
    id: NotRequired[str]


class FunctionResultContent(TypedDict, extra_items=object):
    """A function tool result content block."""

    type: Literal['function_result']
    name: NotRequired[str]
    is_error: NotRequired[bool]
    result: NotRequired[dict[str, object] | str]
    # The ID of the function call this answers.
    call_id: NotRequired[str]


CONTENT_TYPES = frozenset({
    'text',
    'image',
    'audio',
    'document',
    'video',
    'thought',
    'function_call',
    'function_result',
})


class UnknownContent(UnknownObject):
    """A content type this plugin doesn't know yet (or doesn't match its type), kept as sent."""


KnownContent: TypeAlias = Annotated[
    TextContent
    | ImageContent
    | AudioContent
    | DocumentContent
    | VideoContent
    | ThoughtContent
    | FunctionCallContent
    | FunctionResultContent,
    Discriminator('type'),
]
Content: TypeAlias = Annotated[
    Annotated[KnownContent, Tag(KNOWN_TAG)]
    | Annotated[UnknownContent, keep_as(UnknownContent, 'type'), Tag(UNKNOWN_TAG)],
    open_union_tag('type', CONTENT_TYPES),
    keep_unknown(UnknownContent, 'type'),
]
"""The content of a step, tagged by ``type``."""

# ---------------------------------------------------------------------------
# Steps
#
# Tool call/result steps: only `type` is required here (see the module
# docstring). The platform omits fields the spec marks as required at times
# (e.g. a `google_search_call` without `arguments`), so content fields are
# sent back only when present.
# ---------------------------------------------------------------------------


class Status(TypedDict, extra_items=object):
    """A step error (google.rpc.Status). Python only: read by the converters."""

    # A google.rpc code.
    code: NotRequired[int]
    message: NotRequired[str]


class ModelOutputStep(TypedDict, extra_items=object):
    """Output from the model.

    ``content`` is optional: the server can end a response with an empty
    ``{type: 'model_output'}`` step (seen live on vertexAI with Google Maps).
    """

    type: Literal['model_output']
    content: NotRequired[list[Content]]
    error: NotRequired[Status]


class UserInputStep(TypedDict, extra_items=object):
    """Input from the user."""

    type: Literal['user_input']
    content: NotRequired[list[Content]]


class QueriesArguments(TypedDict, extra_items=object):
    """The arguments of a Google Search or Google Maps call (inline in JS)."""

    queries: NotRequired[list[str]]


class UrlsArguments(TypedDict, extra_items=object):
    """The arguments of a URL context call (inline in JS)."""

    urls: NotRequired[list[str]]


class CodeArguments(TypedDict, extra_items=object):
    """The arguments of a code execution call (inline in JS)."""

    code: NotRequired[str]
    # `python` or `bash`.
    language: NotRequired[str]


class GoogleSearchCallStep(TypedDict, extra_items=object):
    """A Google Search call the server ran."""

    type: Literal['google_search_call']
    id: NotRequired[str]
    arguments: NotRequired[QueriesArguments]
    # `web_search`, `image_search` or `enterprise_web_search`.
    search_type: NotRequired[str]
    signature: NotRequired[str]


class GoogleSearchResultStep(TypedDict, extra_items=object):
    """The result of a Google Search call."""

    type: Literal['google_search_result']
    call_id: NotRequired[str]
    # The search results, e.g. `[{search_suggestions}]`.
    result: NotRequired[list[dict[str, object]] | dict[str, object]]
    is_error: NotRequired[bool]
    signature: NotRequired[str]


class CodeExecutionCallStep(TypedDict, extra_items=object):
    """A code execution call the server ran."""

    type: Literal['code_execution_call']
    id: NotRequired[str]
    arguments: NotRequired[CodeArguments]
    signature: NotRequired[str]


class CodeExecutionResultStep(TypedDict, extra_items=object):
    """The result of a code execution call."""

    type: Literal['code_execution_result']
    call_id: NotRequired[str]
    result: NotRequired[dict[str, object] | str]
    signature: NotRequired[str]


class FunctionCallStep(TypedDict, extra_items=object):
    """A function call: for the client to run, or already run by the server (e.g. Antigravity)."""

    type: Literal['function_call']
    name: NotRequired[str]
    arguments: NotRequired[dict[str, object]]
    id: NotRequired[str]
    signature: NotRequired[str]


class FunctionResultStep(TypedDict, extra_items=object):
    """The result of a function call."""

    type: Literal['function_result']
    name: NotRequired[str]
    call_id: NotRequired[str]
    is_error: NotRequired[bool]
    # JS: text and image content only in the list. Any content is accepted.
    result: NotRequired[dict[str, object] | str | list[Content]]
    signature: NotRequired[str]


class ThoughtStep(TypedDict, extra_items=object):
    """A thought, with its signature."""

    type: Literal['thought']
    signature: NotRequired[str]
    # JS: text and image only. Any content is accepted.
    summary: NotRequired[list[Content]]


class ProcessingCallStep(TypedDict, extra_items=object):
    """Agentic video understanding: the model asked to load part of a video. Needs no reply."""

    type: Literal['processing_call']
    id: NotRequired[str]
    signature: NotRequired[str]


class ProcessingResultStep(TypedDict, extra_items=object):
    """Agentic video understanding: the result of a ``processing_call``."""

    type: Literal['processing_result']
    call_id: NotRequired[str]
    signature: NotRequired[str]


# Server-side built-in tool steps. The server runs these tools itself; the
# steps (with their signatures) must be sent back in the history so the model
# keeps the tool results in later turns.


class UrlContextCallStep(TypedDict, extra_items=object):
    """A URL context call the server ran."""

    type: Literal['url_context_call']
    id: NotRequired[str]
    arguments: NotRequired[UrlsArguments]
    signature: NotRequired[str]


class UrlContextResultStep(TypedDict, extra_items=object):
    """The result of a URL context call."""

    type: Literal['url_context_result']
    call_id: NotRequired[str]
    # One entry per URL: `{url, status}` (`success`, `error`, ...).
    result: NotRequired[list[dict[str, object]]]
    is_error: NotRequired[bool]
    signature: NotRequired[str]


class FileSearchCallStep(TypedDict, extra_items=object):
    """A file search call the server ran."""

    type: Literal['file_search_call']
    id: NotRequired[str]
    signature: NotRequired[str]


class FileSearchResultStep(TypedDict, extra_items=object):
    """The result of a file search call."""

    type: Literal['file_search_result']
    call_id: NotRequired[str]
    # Not in the spec's step schema, but the streamed delta carries it (and
    # the stream assembly copies it into the step), so it's kept.
    result: NotRequired[list[dict[str, object]]]
    signature: NotRequired[str]


class GoogleMapsCallStep(TypedDict, extra_items=object):
    """A Google Maps call the server ran."""

    type: Literal['google_maps_call']
    id: NotRequired[str]
    arguments: NotRequired[QueriesArguments]
    signature: NotRequired[str]


class GoogleMapsResultStep(TypedDict, extra_items=object):
    """The result of a Google Maps call."""

    type: Literal['google_maps_result']
    call_id: NotRequired[str]
    # e.g. `[{places: [{place_id, name, url}]}]`.
    result: NotRequired[list[dict[str, object]]]
    signature: NotRequired[str]


class McpServerToolCallStep(TypedDict, extra_items=object):
    """An MCP server tool call the server ran (no signature in the spec)."""

    type: Literal['mcp_server_tool_call']
    id: NotRequired[str]
    name: NotRequired[str]
    server_name: NotRequired[str]
    arguments: NotRequired[dict[str, object]]


class McpServerToolResultStep(TypedDict, extra_items=object):
    """The result of an MCP server tool call."""

    type: Literal['mcp_server_tool_result']
    call_id: NotRequired[str]
    name: NotRequired[str]
    server_name: NotRequired[str]
    # JS: text and image content only in the list. Any content is accepted.
    result: NotRequired[dict[str, object] | str | list[Content]]


BuiltInToolStep: TypeAlias = (
    UrlContextCallStep
    | UrlContextResultStep
    | FileSearchCallStep
    | FileSearchResultStep
    | GoogleMapsCallStep
    | GoogleMapsResultStep
    | McpServerToolCallStep
    | McpServerToolResultStep
)


class UnknownStep(UnknownObject):
    """A step type this plugin doesn't know yet (or doesn't match its type), kept as sent."""


STEP_TYPES = frozenset({
    'model_output',
    'user_input',
    'google_search_call',
    'google_search_result',
    'code_execution_call',
    'code_execution_result',
    'function_call',
    'function_result',
    'thought',
    'processing_call',
    'processing_result',
    'url_context_call',
    'url_context_result',
    'file_search_call',
    'file_search_result',
    'google_maps_call',
    'google_maps_result',
    'mcp_server_tool_call',
    'mcp_server_tool_result',
})

KnownStep: TypeAlias = Annotated[
    ModelOutputStep
    | UserInputStep
    | GoogleSearchCallStep
    | GoogleSearchResultStep
    | CodeExecutionCallStep
    | CodeExecutionResultStep
    | FunctionCallStep
    | FunctionResultStep
    | ThoughtStep
    | ProcessingCallStep
    | ProcessingResultStep
    | BuiltInToolStep,
    Discriminator('type'),
]
Step: TypeAlias = Annotated[
    Annotated[KnownStep, Tag(KNOWN_TAG)] | Annotated[UnknownStep, keep_as(UnknownStep, 'type'), Tag(UNKNOWN_TAG)],
    open_union_tag('type', STEP_TYPES),
    keep_unknown(UnknownStep, 'type'),
]
"""A step of the interaction timeline, tagged by ``type``."""

# ---------------------------------------------------------------------------
# Usage
# ---------------------------------------------------------------------------


class ModalityTokens(TypedDict, extra_items=object):
    """The token count for a single modality."""

    # ResponseModality: `text`, `image`, `audio`, ...
    modality: NotRequired[str]
    tokens: NotRequired[int]


class GroundingToolCount(TypedDict, extra_items=object):
    """The number of times a grounding tool was used."""

    # The grounding tool, e.g. `google_search` or `google_maps`.
    type: NotRequired[str]
    count: NotRequired[int]


class Usage(TypedDict, extra_items=object):
    """Statistics on the interaction request's token usage."""

    # Tokens in the prompt (context).
    total_input_tokens: NotRequired[int]
    input_tokens_by_modality: NotRequired[list[ModalityTokens]]
    # Tokens in the cached part of the prompt.
    total_cached_tokens: NotRequired[int]
    cached_tokens_by_modality: NotRequired[list[ModalityTokens]]
    # Tokens across all the generated responses.
    total_output_tokens: NotRequired[int]
    output_tokens_by_modality: NotRequired[list[ModalityTokens]]
    # Tokens in tool-use prompts.
    total_tool_use_tokens: NotRequired[int]
    tool_use_tokens_by_modality: NotRequired[list[ModalityTokens]]
    # Thought tokens of thinking models.
    total_thought_tokens: NotRequired[int]
    # Prompt + responses + other internal tokens.
    total_tokens: NotRequired[int]
    grounding_tool_count: NotRequired[list[GroundingToolCount]]


# ---------------------------------------------------------------------------
# Stream deltas (the `delta` of a `step.delta` event). Not used until the
# streaming client arrives; kept in step with JS.
# ---------------------------------------------------------------------------


class TextDelta(TypedDict, extra_items=object):
    type: Literal['text']
    text: NotRequired[str]


class ImageDelta(TypedDict, extra_items=object):
    type: Literal['image']
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]
    resolution: NotRequired[str]


class AudioDelta(TypedDict, extra_items=object):
    type: Literal['audio']
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]
    sample_rate: NotRequired[int]
    channels: NotRequired[int]


class DocumentDelta(TypedDict, extra_items=object):
    type: Literal['document']
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]


class VideoDelta(TypedDict, extra_items=object):
    type: Literal['video']
    data: NotRequired[str]
    uri: NotRequired[str]
    mime_type: NotRequired[str]
    resolution: NotRequired[str]


class ThoughtSummaryDelta(TypedDict, extra_items=object):
    type: Literal['thought_summary']
    content: NotRequired[Content]


class ThoughtSignatureDelta(TypedDict, extra_items=object):
    type: Literal['thought_signature']
    signature: NotRequired[str]


class ArgumentsDelta(TypedDict, extra_items=object):
    type: Literal['arguments_delta']
    # A piece of the function call's JSON arguments.
    arguments: NotRequired[str]


class CodeExecutionCallDelta(TypedDict, extra_items=object):
    type: Literal['code_execution_call']
    arguments: NotRequired[CodeArguments]
    signature: NotRequired[str]


class UrlContextCallDelta(TypedDict, extra_items=object):
    type: Literal['url_context_call']
    arguments: NotRequired[UrlsArguments]
    signature: NotRequired[str]


class GoogleSearchCallDelta(TypedDict, extra_items=object):
    type: Literal['google_search_call']
    arguments: NotRequired[QueriesArguments]
    signature: NotRequired[str]


class McpServerToolCallDelta(TypedDict, extra_items=object):
    type: Literal['mcp_server_tool_call']
    name: NotRequired[str]
    server_name: NotRequired[str]
    arguments: NotRequired[dict[str, object]]


class FileSearchCallDelta(TypedDict, extra_items=object):
    type: Literal['file_search_call']
    signature: NotRequired[str]


class GoogleMapsCallDelta(TypedDict, extra_items=object):
    type: Literal['google_maps_call']
    arguments: NotRequired[QueriesArguments]
    signature: NotRequired[str]


class FunctionResultDelta(TypedDict, extra_items=object):
    type: Literal['function_result']
    name: NotRequired[str]
    call_id: NotRequired[str]
    is_error: NotRequired[bool]
    result: NotRequired[dict[str, object] | str | list[Content]]


class CodeExecutionResultDelta(TypedDict, extra_items=object):
    type: Literal['code_execution_result']
    result: NotRequired[str]
    is_error: NotRequired[bool]
    signature: NotRequired[str]


class UrlContextResultDelta(TypedDict, extra_items=object):
    type: Literal['url_context_result']
    result: NotRequired[list[dict[str, object]]]
    is_error: NotRequired[bool]
    signature: NotRequired[str]


class GoogleSearchResultDelta(TypedDict, extra_items=object):
    type: Literal['google_search_result']
    result: NotRequired[list[dict[str, object]]]
    is_error: NotRequired[bool]
    signature: NotRequired[str]


class McpServerToolResultDelta(TypedDict, extra_items=object):
    type: Literal['mcp_server_tool_result']
    name: NotRequired[str]
    server_name: NotRequired[str]
    result: NotRequired[dict[str, object] | str | list[Content]]


class FileSearchResultDelta(TypedDict, extra_items=object):
    type: Literal['file_search_result']
    result: NotRequired[list[dict[str, object]]]
    signature: NotRequired[str]


class GoogleMapsResultDelta(TypedDict, extra_items=object):
    type: Literal['google_maps_result']
    result: NotRequired[list[dict[str, object]]]
    signature: NotRequired[str]


class ProcessingCallDelta(TypedDict, extra_items=object):
    type: Literal['processing_call']
    signature: NotRequired[str]


class ProcessingResultDelta(TypedDict, extra_items=object):
    type: Literal['processing_result']
    signature: NotRequired[str]


class TextAnnotationDelta(TypedDict, extra_items=object):
    type: Literal['text_annotation_delta']
    annotations: NotRequired[list[TextAnnotation]]


class UnknownStepDelta(UnknownObject):
    """A delta type this plugin doesn't know yet (or doesn't match its type), kept as sent."""


STEP_DELTA_TYPES = frozenset({
    'text',
    'image',
    'audio',
    'document',
    'video',
    'thought_summary',
    'thought_signature',
    'arguments_delta',
    'code_execution_call',
    'url_context_call',
    'google_search_call',
    'mcp_server_tool_call',
    'file_search_call',
    'google_maps_call',
    'function_result',
    'code_execution_result',
    'url_context_result',
    'google_search_result',
    'mcp_server_tool_result',
    'file_search_result',
    'google_maps_result',
    'processing_call',
    'processing_result',
    'text_annotation_delta',
})

KnownStepDelta: TypeAlias = Annotated[
    TextDelta
    | ImageDelta
    | AudioDelta
    | DocumentDelta
    | VideoDelta
    | ThoughtSummaryDelta
    | ThoughtSignatureDelta
    | ArgumentsDelta
    | CodeExecutionCallDelta
    | UrlContextCallDelta
    | GoogleSearchCallDelta
    | McpServerToolCallDelta
    | FileSearchCallDelta
    | GoogleMapsCallDelta
    | FunctionResultDelta
    | CodeExecutionResultDelta
    | UrlContextResultDelta
    | GoogleSearchResultDelta
    | McpServerToolResultDelta
    | FileSearchResultDelta
    | GoogleMapsResultDelta
    | ProcessingCallDelta
    | ProcessingResultDelta
    | TextAnnotationDelta,
    Discriminator('type'),
]
StepDeltaData: TypeAlias = Annotated[
    Annotated[KnownStepDelta, Tag(KNOWN_TAG)]
    | Annotated[UnknownStepDelta, keep_as(UnknownStepDelta, 'type'), Tag(UNKNOWN_TAG)],
    open_union_tag('type', STEP_DELTA_TYPES),
    keep_unknown(UnknownStepDelta, 'type'),
]
"""The ``delta`` of a ``step.delta`` event, tagged by ``type``."""

# ---------------------------------------------------------------------------
# The interaction and its errors
# ---------------------------------------------------------------------------


class InteractionError(TypedDict, extra_items=object):
    """An error recorded on an interaction or sent in an SSE ``error`` event."""

    # Error code, e.g. `safety` or `malformed_function_call`. JS types it as
    # a string; an HTTP error body can carry a number.
    code: NotRequired[str | int]
    # Human-readable error message.
    message: NotRequired[str]


class ApiErrorBody(TypedDict, extra_items=object):
    """The body of a failed API call: ``{"error": {"code": ..., "message": ...}}``. Python only."""

    error: NotRequired[InteractionError]


class GeminiInteraction(TypedDict, extra_items=object):
    """An interaction, as the API returns it."""

    # The model or agent used for the interaction.
    model: NotRequired[str]
    agent: NotRequired[str]
    # The environment ID for the sandbox, if used.
    environment_id: NotRequired[str]
    # The ID to continue from in later interactions (`v1_...`).
    id: NotRequired[str]
    previous_interaction_id: NotRequired[str]
    # `in_progress`, `queued`, `requires_action`, `completed`, `incomplete`,
    # `failed` or `cancelled` (JS lists these; Python also handles
    # `budget_exceeded`).
    status: NotRequired[str]
    # ISO 8601 times.
    created: NotRequired[str]
    updated: NotRequired[str]
    role: NotRequired[str]
    # Steps comprising the interaction timeline.
    steps: NotRequired[list[Step]]
    usage: NotRequired[Usage]
    # Set when `status` is `failed` (e.g. a safety block or an unrecoverable
    # tool-call error).
    errors: NotRequired[list[InteractionError]]


# ---------------------------------------------------------------------------
# SSE events (inline object types in JS's InteractionSseEvent)
# ---------------------------------------------------------------------------


class InteractionCreatedEvent(TypedDict, extra_items=object):
    event_type: Literal['interaction.created']
    interaction: NotRequired[GeminiInteraction]
    event_id: NotRequired[str]


class InteractionCompletedEvent(TypedDict, extra_items=object):
    event_type: Literal['interaction.completed']
    interaction: NotRequired[GeminiInteraction]
    event_id: NotRequired[str]


class InteractionStatusUpdateEvent(TypedDict, extra_items=object):
    event_type: Literal['interaction.status_update']
    # Only present when the interaction is stored (`store: true`).
    interaction_id: NotRequired[str]
    status: NotRequired[str]
    event_id: NotRequired[str]


class InteractionErrorEvent(TypedDict, extra_items=object):
    event_type: Literal['error']
    error: NotRequired[InteractionError]
    event_id: NotRequired[str]


class StepStartEvent(TypedDict, extra_items=object):
    event_type: Literal['step.start']
    index: NotRequired[int]
    step: NotRequired[Step]
    event_id: NotRequired[str]


class StepDeltaEvent(TypedDict, extra_items=object):
    event_type: Literal['step.delta']
    index: NotRequired[int]
    delta: NotRequired[StepDeltaData]
    event_id: NotRequired[str]


class StepStopEvent(TypedDict, extra_items=object):
    event_type: Literal['step.stop']
    index: NotRequired[int]
    # Usage so far in the interaction (sent by vertexAI).
    usage: NotRequired[Usage]
    # Usage of this step.
    step_usage: NotRequired[Usage]
    event_id: NotRequired[str]


class UnknownSseEvent(UnknownObject):
    """An event type this plugin doesn't know yet (or doesn't match its type), kept as sent."""


SSE_EVENT_TYPES = frozenset({
    'interaction.created',
    'interaction.completed',
    'interaction.status_update',
    'error',
    'step.start',
    'step.delta',
    'step.stop',
})

KnownSseEvent: TypeAlias = Annotated[
    InteractionCreatedEvent
    | InteractionCompletedEvent
    | InteractionStatusUpdateEvent
    | InteractionErrorEvent
    | StepStartEvent
    | StepDeltaEvent
    | StepStopEvent,
    Discriminator('event_type'),
]
InteractionSseEvent: TypeAlias = Annotated[
    Annotated[KnownSseEvent, Tag(KNOWN_TAG)]
    | Annotated[UnknownSseEvent, keep_as(UnknownSseEvent, 'event_type'), Tag(UNKNOWN_TAG)],
    open_union_tag('event_type', SSE_EVENT_TYPES),
    keep_unknown(UnknownSseEvent, 'event_type'),
]
"""One server-sent event of a streamed interaction, tagged by ``event_type``."""

# ---------------------------------------------------------------------------
# Parsing and checks
# ---------------------------------------------------------------------------

INTERACTION_ADAPTER: TypeAdapter[GeminiInteraction] = TypeAdapter(GeminiInteraction)
STEP_ADAPTER: TypeAdapter[Step] = TypeAdapter(Step)
CONTENT_ADAPTER: TypeAdapter[Content] = TypeAdapter(Content)
SSE_EVENT_ADAPTER: TypeAdapter[InteractionSseEvent] = TypeAdapter(InteractionSseEvent)
API_ERROR_BODY_ADAPTER: TypeAdapter[ApiErrorBody] = TypeAdapter(ApiErrorBody)


def parse_interaction(body: object) -> GeminiInteraction:
    """Validate a response body as an interaction (raises ``ValidationError`` when it isn't one).

    Unknown steps and content, and known ones whose fields don't match their
    type, are kept as sent (``UnknownStep`` / ``UnknownContent``).
    """
    return INTERACTION_ADAPTER.validate_python(body)


def parse_sse_event(data: object) -> InteractionSseEvent:
    """Validate the JSON of one SSE ``data:`` line as an event."""
    return SSE_EVENT_ADAPTER.validate_python(data)
