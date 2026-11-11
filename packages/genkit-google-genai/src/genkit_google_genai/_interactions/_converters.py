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

"""Converters between Genkit message parts and Interactions API wire shapes."""

from __future__ import annotations

import copy
import json
import re
from collections.abc import Mapping
from typing import Any, Literal, TypeVar, cast
from urllib.parse import urlparse

from genkit_google_genai._interactions._errors import interaction_error_status, is_interaction_content_block_code
from genkit_google_genai._interactions._types import (
    STEP_ADAPTER,
    AudioContent,
    CodeArguments,
    CodeExecutionCallStep,
    CodeExecutionResultStep,
    Content,
    DocumentContent,
    FileSearchCallStep,
    FileSearchResultStep,
    FunctionCallStep,
    FunctionResultStep,
    GeminiInteraction,
    GoogleMapsCallStep,
    GoogleMapsResultStep,
    GoogleSearchCallStep,
    GoogleSearchResultStep,
    ImageContent,
    InteractionFunctionTool,
    McpServerToolCallStep,
    McpServerToolResultStep,
    ModalityTokens,
    ModelOutputStep,
    ProcessingCallStep,
    ProcessingResultStep,
    QueriesArguments,
    Status,
    Step,
    TextContent,
    ThoughtStep,
    UnknownContent,
    UnknownStep,
    UrlContextCallStep,
    UrlContextResultStep,
    UrlsArguments,
    Usage,
    UserInputStep,
    VideoContent,
)
from pydantic import BaseModel, TypeAdapter, ValidationError
from typing_extensions import NotRequired, TypedDict

from genkit import Failure, FinishReason, GenkitError, Media, Message, ModelResponse, Operation, Part, get_logger
from genkit.model import (
    GenerationUsage,
    ToolDefinition,
    ToolRequest,
    ToolResponse,
)
from genkit.plugin_api import StatusName, provider_error

logger = get_logger(__name__)

InteractionRole = Literal['user', 'model']

# Keys we round-trip through Genkit custom parts and metadata for wire concepts
# Genkit has no first-class part for. Both directions read them, so a typo in
# one place would silently break pairing — keep them defined once.
THOUGHT_SIGNATURE = 'thoughtSignature'
CALL_ID = 'callId'
GOOGLE_SEARCH_CALL = 'googleSearchCall'
GOOGLE_SEARCH_RESULT = 'googleSearchResult'
EXECUTABLE_CODE = 'executableCode'
CODE_EXECUTION_RESULT = 'codeExecutionResult'
PROCESSING_CALL = 'processingCall'
PROCESSING_RESULT = 'processingResult'
URL_CONTEXT_CALL = 'urlContextCall'
URL_CONTEXT_RESULT = 'urlContextResult'
FILE_SEARCH_CALL = 'fileSearchCall'
FILE_SEARCH_RESULT = 'fileSearchResult'
GOOGLE_MAPS_CALL = 'googleMapsCall'
GOOGLE_MAPS_RESULT = 'googleMapsResult'
MCP_SERVER_TOOL_CALL = 'mcpServerToolCall'
MCP_SERVER_TOOL_RESULT = 'mcpServerToolResult'
UNKNOWN_STEP = 'unknownStep'
UNKNOWN_CONTENT = 'unknownContent'
SERVER_FUNCTION_CALL = 'serverFunctionCall'
SERVER_FUNCTION_RESULT = 'serverFunctionResult'
THOUGHT_CUSTOM = 'thought'
# Wire function_result.content is only text/image blocks, not a JSON array.
CONTENT_BLOCK_TYPES = frozenset({'text', 'image'})
# The language when a code execution step names none.
CODE_LANGUAGE = 'python'


# Custom parts that stand in for server-side tool steps, read back when the
# history is resent (camelCase keys, like the JS plugin's zod schemas). Only
# the fields the plugin reads; ids default to '' when missing.
class GoogleSearchCallPart(TypedDict):
    id: NotRequired[str]
    arguments: NotRequired[QueriesArguments]
    searchType: NotRequired[str]


# `result` is an array of items (e.g. `[{search_suggestions}]`); an object is
# still accepted for parts saved before that was fixed in JS.
class GoogleSearchResultPart(TypedDict):
    callId: NotRequired[str]
    result: NotRequired[list[dict[str, object]] | dict[str, object]]
    isError: NotRequired[bool]


class ExecutableCodePart(TypedDict):
    code: NotRequired[str]
    language: NotRequired[str]


class CodeExecutionResultPart(TypedDict):
    output: NotRequired[str]
    outcome: NotRequired[str]


class CallPart(TypedDict):
    id: NotRequired[str]


class ResultPart(TypedDict):
    callId: NotRequired[str]


class UrlContextCallPart(TypedDict):
    id: NotRequired[str]
    arguments: NotRequired[UrlsArguments]


class UrlContextResultPart(TypedDict):
    callId: NotRequired[str]
    result: NotRequired[list[dict[str, object]]]
    isError: NotRequired[bool]


class ListResultPart(TypedDict):
    callId: NotRequired[str]
    result: NotRequired[list[dict[str, object]]]


class GoogleMapsCallPart(TypedDict):
    id: NotRequired[str]
    arguments: NotRequired[QueriesArguments]


class McpServerToolCallPart(TypedDict):
    id: NotRequired[str]
    name: NotRequired[str]
    serverName: NotRequired[str]
    arguments: NotRequired[dict[str, object]]


class McpServerToolResultPart(TypedDict):
    callId: NotRequired[str]
    name: NotRequired[str]
    serverName: NotRequired[str]
    result: NotRequired[dict[str, object] | str | list[Content]]


GOOGLE_SEARCH_CALL_PART: TypeAdapter[GoogleSearchCallPart] = TypeAdapter(GoogleSearchCallPart)
GOOGLE_SEARCH_RESULT_PART: TypeAdapter[GoogleSearchResultPart] = TypeAdapter(GoogleSearchResultPart)
EXECUTABLE_CODE_PART: TypeAdapter[ExecutableCodePart] = TypeAdapter(ExecutableCodePart)
CODE_EXECUTION_RESULT_PART: TypeAdapter[CodeExecutionResultPart] = TypeAdapter(CodeExecutionResultPart)
CALL_PART: TypeAdapter[CallPart] = TypeAdapter(CallPart)
RESULT_PART: TypeAdapter[ResultPart] = TypeAdapter(ResultPart)
URL_CONTEXT_CALL_PART: TypeAdapter[UrlContextCallPart] = TypeAdapter(UrlContextCallPart)
URL_CONTEXT_RESULT_PART: TypeAdapter[UrlContextResultPart] = TypeAdapter(UrlContextResultPart)
LIST_RESULT_PART: TypeAdapter[ListResultPart] = TypeAdapter(ListResultPart)
GOOGLE_MAPS_CALL_PART: TypeAdapter[GoogleMapsCallPart] = TypeAdapter(GoogleMapsCallPart)
MCP_SERVER_TOOL_CALL_PART: TypeAdapter[McpServerToolCallPart] = TypeAdapter(McpServerToolCallPart)
MCP_SERVER_TOOL_RESULT_PART: TypeAdapter[McpServerToolResultPart] = TypeAdapter(McpServerToolResultPart)
CONTENT_LIST_ADAPTER: TypeAdapter[list[Content]] = TypeAdapter(list[Content])
JSON_OBJECT_ADAPTER: TypeAdapter[dict[str, object]] = TypeAdapter(dict[str, object])

_T = TypeVar('_T')


def read_as(adapter: TypeAdapter[_T], value: object) -> _T | None:
    """``value`` validated as the adapter's type, or None when it doesn't match."""
    try:
        return adapter.validate_python(value)
    except ValidationError:
        return None


FAILED_MESSAGE = 'Interaction failed'
# Step errors report a google.rpc code; its position here is that code.
GRPC_STATUS_NAMES: tuple[StatusName, ...] = (
    'OK',
    'CANCELLED',
    'UNKNOWN',
    'INVALID_ARGUMENT',
    'DEADLINE_EXCEEDED',
    'NOT_FOUND',
    'ALREADY_EXISTS',
    'PERMISSION_DENIED',
    'RESOURCE_EXHAUSTED',
    'FAILED_PRECONDITION',
    'ABORTED',
    'OUT_OF_RANGE',
    'UNIMPLEMENTED',
    'INTERNAL',
    'UNAVAILABLE',
    'DATA_LOSS',
    'UNAUTHENTICATED',
)
# Chat generate can pause for a tool; a background job cannot.
BACKGROUND_INTERRUPT_UNSUPPORTED = 'Background models do not support interrupts'

# Content types guessed from a URL's file extension when a media part has
# none (same list as the JS plugin).
KNOWN_MIME_TYPES: dict[str, str] = {
    'jpg': 'image/jpeg',
    'jpeg': 'image/jpeg',
    'png': 'image/png',
    'mp4': 'video/mp4',
    'pdf': 'application/pdf',
}
YOUTUBE_HOSTS = frozenset({'youtube.com', 'youtu.be'})

# Turns a Genkit role into the side of the transcript it belongs on. Tool
# results are part of the user's side; system prompts aren't turns at all.
INTERACTION_ROLES: dict[str, InteractionRole] = {
    'user': 'user',
    'model': 'model',
    'tool': 'user',
}

# Terminal statuses that can still carry partial output worth surfacing.
PARTIAL_TERMINAL_STATUSES: dict[str, tuple[FinishReason, str]] = {
    'incomplete': (FinishReason.LENGTH, 'Interaction incomplete (truncated output)'),
    'budget_exceeded': (FinishReason.ABORTED, 'Interaction exceeded its budget'),
}


def step_errors(interaction: GeminiInteraction) -> list[Status]:
    """The errors on the interaction's ``model_output`` steps."""
    errors: list[Status] = []
    for step in interaction.get('steps') or []:
        if isinstance(step, UnknownStep) or step['type'] != 'model_output':
            continue
        error = step.get('error')
        if error is not None:
            errors.append(error)
    return errors


def interaction_error_message(interaction: GeminiInteraction) -> str | None:
    """Read the failure message a failed Interaction reports.

    Prefer the top-level errors list when the job died before any output
    step, then fall back to a step error so callers see the server's reason
    instead of a generic 'Interaction failed'.
    """
    for err in interaction.get('errors') or []:
        message = err.get('message')
        if message:
            return message
    for error in step_errors(interaction):
        message = error.get('message')
        if message:
            return message
    return None


def error_from_provider_code(code: object, cause: BaseException) -> GenkitError | None:
    """Turn an error code the Interactions API reported into a GenkitError.

    Top-level errors carry a string code: an Interactions code like
    ``service_unavailable`` or ``malformed_function_call`` (see
    ``INTERACTION_ERROR_CODE_TO_STATUS``), a canonical name like
    ``RESOURCE_EXHAUSTED``, or a number; step errors carry a gRPC int.
    Numbers 1-16 are gRPC codes and 400-599 are HTTP codes. Anything else,
    and any code that only maps to OK or UNKNOWN, returns None so the caller
    can look at the next error.
    """
    text = str(code).strip() if code is not None and not isinstance(code, bool) else ''
    if text.isdigit():
        number = int(text)
        if 400 <= number <= 599:
            error = provider_error(cause, http_status=number)
        elif number < len(GRPC_STATUS_NAMES):
            error = provider_error(cause, status=GRPC_STATUS_NAMES[number])
        else:
            return None
    elif (status := interaction_error_status(text)) is not None:
        error = provider_error(cause, status=status)
    elif text.upper() in GRPC_STATUS_NAMES:
        error = provider_error(cause, status=cast(StatusName, text.upper()))
    else:
        return None
    if error.status in ('OK', 'UNKNOWN'):
        return None
    return error


def first_error(interaction: GeminiInteraction) -> tuple[str | int | None, str | None]:
    """The code and message of the interaction's first top-level error, if any."""
    for err in interaction.get('errors') or []:
        return err.get('code'), err.get('message')
    return None, None


def blocked_response(interaction: GeminiInteraction) -> ModelResponse | None:
    """The response for an interaction that failed on a content-policy block, or None.

    Safety, recitation and the other content-policy codes return a response
    with finish_reason 'blocked' and no content, like a blocked
    generateContent response, instead of raising.
    """
    code, message = first_error(interaction)
    if not is_interaction_content_block_code(code):
        return None
    return model_response(
        interaction,
        content=[],
        finish_reason=FinishReason.BLOCKED,
        finish_message=message or interaction_error_message(interaction) or FAILED_MESSAGE,
    )


def interaction_error(interaction: GeminiInteraction, cause: BaseException) -> GenkitError:
    """The error for the first failure on an Interaction that has a usable code, else UNKNOWN."""
    for err in interaction.get('errors') or []:
        if error := error_from_provider_code(err.get('code'), cause):
            return error
    for step_error in step_errors(interaction):
        if error := error_from_provider_code(step_error.get('code'), cause):
            return error
    return provider_error(cause, status='UNKNOWN')


def clean_schema(schema: dict[str, Any]) -> dict[str, Any]:
    """Strip schema keys the Interactions API rejects."""
    out = copy.deepcopy(schema)
    for key in list(out):
        if key in ('$schema', 'additionalProperties'):
            del out[key]
            continue
        value = out[key]
        if isinstance(value, dict):
            out[key] = clean_schema(value)
        elif isinstance(value, list):
            if key == 'type':
                out[key] = next((item for item in value if item != 'null'), value[0] if value else value)
            else:
                out[key] = [clean_schema(item) if isinstance(item, dict) else item for item in value]
    return out


def ensure_tool_ids(messages: list[Message]) -> list[Message]:
    """Assign stable tool call IDs so wire payloads stay pairable."""
    request_ids: list[str] = []
    next_id_counter = 0

    new_messages = [message.model_copy(deep=True) for message in messages]

    for message in new_messages:
        for part in message.content:
            if part.tool_request is not None:
                if not part.tool_request.ref:
                    new_id = f'genkit-auto-id-{next_id_counter}'
                    next_id_counter += 1
                    part.tool_request.ref = new_id
                request_ids.append(part.tool_request.ref)

    # A response that already named its call isn't in the pairing pool.
    claimed = {
        part.tool_response.ref
        for message in new_messages
        for part in message.content
        if part.tool_response is not None and part.tool_response.ref
    }
    available = [ref for ref in request_ids if ref not in claimed]

    # Responses without refs reuse leftover request IDs in order; unmatched
    # ones get orphan IDs so the wire never sends an empty call_id.
    for message in new_messages:
        for part in message.content:
            if part.tool_response is not None and not part.tool_response.ref:
                if available:
                    part.tool_response.ref = available.pop(0)
                else:
                    part.tool_response.ref = f'genkit-orphan-id-{next_id_counter}'
                    next_id_counter += 1

    return new_messages


def to_interaction_tool(tool: ToolDefinition) -> InteractionFunctionTool:
    """Convert a Genkit tool definition to an Interactions function tool."""
    func: InteractionFunctionTool = {
        'type': 'function',
        'name': tool.name,
    }
    if tool.description is not None:
        func['description'] = tool.description
    if tool.input_schema is not None:
        if isinstance(tool.input_schema, dict):
            func['parameters'] = clean_schema(tool.input_schema)
        else:
            func['parameters'] = clean_schema(dict(tool.input_schema))
    return func


def to_interaction_content(part: Part) -> Content | None:
    """Convert a Genkit part to an Interactions content block."""
    if part.text is not None:
        text: TextContent = {'type': 'text', 'text': part.text}
        return text
    if part.media is not None:
        return to_interaction_media(part)
    logger.warning('Unsupported part type for Interaction input: %s', part.model_dump(by_alias=True))
    return None


def extract_mime_type(url: str) -> str:
    """The content type a URL names: a data URL's header, or a known file extension; else ''."""
    if not url.startswith('data:'):
        extension = url.rsplit('.', 1)[-1]
        return KNOWN_MIME_TYPES.get(extension, '')
    header, separator, _ = url[len('data:') :].partition(',')
    if not separator:
        return ''
    return header.removesuffix(';base64').strip()


def is_youtube_url(url: str) -> bool:
    """True for a youtube.com or youtu.be link."""
    host = urlparse(url).hostname or ''
    return host.removeprefix('www.').removeprefix('m.') in YOUTUBE_HOSTS


def video_block(part: Part, *, mime_type: str | None = None) -> VideoContent:
    """A video block, with ``metadata['processing']`` (`static` / `agentic`) when the part sets it."""
    video: VideoContent = {'type': 'video'}
    if mime_type:
        video['mime_type'] = mime_type
    processing = (part.metadata or {}).get('processing')
    if isinstance(processing, str):
        video['processing'] = processing
    return video


def to_interaction_media(part: Part) -> Content:
    """Convert a media part to an Interactions image/audio/video/document block.

    Image, audio and video types get their own block; every other type is
    sent as a ``document`` (PDF, CSV, Word, Excel, ...): the server decides
    which types it accepts and returns a clear error for the rest; its list
    has grown beyond the PDF and CSV in the published spec.
    """
    if part.media is None:
        raise GenkitError(status='INVALID_ARGUMENT', message='Media part missing media')
    url = part.media.url
    content_type = part.media.content_type or extract_mime_type(url)

    if not content_type:
        if url.startswith('data:'):
            raise GenkitError(status='INVALID_ARGUMENT', message='Media part missing contentType')
        # A URI whose type can't be worked out (e.g. a YouTube link or a Files
        # API URI). The server works out the type itself, so it's sent without
        # a mime_type, as the generateContent path does. YouTube links are
        # video; anything else is sent as a document (checked live: the
        # server still treats a YouTube link sent as a document as video).
        if is_youtube_url(url):
            youtube = video_block(part)
            youtube['uri'] = url
            return youtube
        unknown: DocumentContent = {'type': 'document', 'uri': url}
        return unknown

    block: ImageContent | AudioContent | VideoContent | DocumentContent
    if content_type.startswith('image/'):
        block = {'type': 'image', 'mime_type': content_type}
    elif content_type.startswith('audio/'):
        block = {'type': 'audio', 'mime_type': content_type}
    elif content_type.startswith('video/'):
        block = video_block(part, mime_type=content_type)
    else:
        block = {'type': 'document', 'mime_type': content_type}

    # Inline data URLs travel as base64; anything else is a reference the API fetches.
    if url.startswith('data:'):
        if ',' not in url:
            raise GenkitError(
                status='INVALID_ARGUMENT',
                message='Malformed data URL for media part: missing payload separator',
            )
        block['data'] = url.split(',', 1)[1]
    else:
        block['uri'] = url
    return block


def to_interaction_role(role: str) -> InteractionRole:
    """Map a Genkit message role to the Interactions API role."""
    if role == 'system':
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message='System role should be handled as system_instruction, not part of turns.',
        )
    return INTERACTION_ROLES.get(role, 'user')


def split_system_instruction(messages: list[Message]) -> tuple[str | None, list[Message]]:
    """Lift system turns out of the transcript into the interaction's instruction.

    An interaction carries one system instruction that governs the whole thing
    rather than a turn in the tape, so however many system messages a prompt
    has, they fold into a single block of text ahead of the conversation.
    """
    instructions: list[str] = []
    turns: list[Message] = []
    for message in messages:
        if message.role != 'system':
            turns.append(message)
            continue
        if any(part.text is None for part in message.content):
            logger.warning('Dropping non-text content from a system message; system instructions are text only.')
        if message.text:
            instructions.append(message.text)
    return '\n\n'.join(instructions) or None, turns


def signature_of(metadata: Mapping[str, object] | None) -> str | None:
    """The thought signature stored on a part's metadata, if any."""
    signature = (metadata or {}).get(THOUGHT_SIGNATURE)
    return signature if isinstance(signature, str) and signature else None


def str_field(payload: Mapping[str, object], *keys: str) -> str:
    """The first non-empty string under one of ``keys`` (camelCase or snake_case), else ''."""
    for key in keys:
        value = payload.get(key)
        if isinstance(value, str) and value:
            return value
    return ''


def without_none(value: object) -> object:
    """A copy with ``None``-valued keys dropped at every dict level (JSON ``null`` means unset)."""
    if isinstance(value, dict):
        return {key: without_none(item) for key, item in value.items() if item is not None}
    if isinstance(value, list):
        return [without_none(item) for item in value]
    return value


def to_function_call_step(request: ToolRequest, signature: str | None = None) -> FunctionCallStep:
    """Compile a Genkit tool request into a function_call step.

    Arguments on the wire are a JSON object. A scalar like ``'Paris'`` would
    silently become ``{}`` and the model would see a different call than the
    one the caller made.
    """
    dumped = plain(request.input)
    args = {} if dumped is None else read_as(JSON_OBJECT_ADAPTER, dumped)
    if args is None:
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                'Tool request input must be a JSON object (got '
                f'{type(dumped).__name__}). Wrap scalars as '
                "{'city': 'Paris'} rather than passing a bare string."
            ),
        )
    step: FunctionCallStep = {
        'type': 'function_call',
        'name': request.name,
        'arguments': args,
        'id': request.ref or '',
    }
    if signature:
        step['signature'] = signature
    return step


def content_block_list(value: object) -> list[Content] | None:
    """``value`` as content blocks when it's a non-empty list of text/image blocks, else None."""
    if not isinstance(value, list) or not value:
        return None
    blocks = read_as(CONTENT_LIST_ADAPTER, value)
    if blocks is None or any(block['type'] not in CONTENT_BLOCK_TYPES for block in blocks):
        return None
    return blocks


def metadata_flag(metadata: dict[str, Any] | None, *keys: str) -> object:
    """Read the first present metadata key (camelCase or snake_case)."""
    if not metadata:
        return None
    for key in keys:
        if key in metadata:
            return metadata[key]
    return None


def to_function_result_step(
    response: ToolResponse,
    metadata: dict[str, Any] | None = None,
) -> FunctionResultStep:
    """Compile a Genkit tool response into a function_result step.

    Ordinary JSON arrays (``[1, 2]``) are not multimodal content — they
    have to be boxed. Only a list of text/image blocks goes on the wire as
    content. A failed result keeps ``is_error`` so the model sees the miss.
    """
    output = plain(response.output)
    result: dict[str, object] | str | list[Content]
    if isinstance(output, str):
        result = output
    elif (as_object := read_as(JSON_OBJECT_ADAPTER, output)) is not None:
        result = as_object
    elif (blocks := content_block_list(output)) is not None:
        result = blocks
    elif output is None:
        result = {}
    else:
        result = {'result': output}
    step: FunctionResultStep = {
        'type': 'function_result',
        'name': response.name,
        'result': result,
        'call_id': response.ref or '',
    }
    if metadata_flag(metadata, 'isError', 'is_error'):
        step['is_error'] = True
    return step


def thought_from_custom(custom: Mapping[str, object] | None) -> Step | None:
    """Replay a thought step that was stashed on the part, images and all.

    Rebuilding from ``reasoning`` would drop image summaries and the
    signature the next turn needs to continue the same thought.
    """
    thought = read_as(STEP_ADAPTER, (custom or {}).get(THOUGHT_CUSTOM))
    if thought is None or isinstance(thought, UnknownStep) or thought['type'] != 'thought':
        return None
    return copy.deepcopy(thought)


def to_thought_step(part: Part) -> Step:
    """Compile a Genkit reasoning part into a thought step."""
    original = thought_from_custom(part.custom)
    if original is not None:
        return original
    thought: ThoughtStep = {
        'type': 'thought',
        'summary': [{'type': 'text', 'text': part.reasoning or ''}],
    }
    if signature := signature_of(part.metadata):
        thought['signature'] = signature
    return thought


def to_server_function_call_step(payload: Mapping[str, object], signature: str | None) -> FunctionCallStep:
    """Replay a server-executed function_call that was stashed as custom."""
    step: FunctionCallStep = {
        'type': 'function_call',
        'name': str_field(payload, 'name'),
        'arguments': read_as(JSON_OBJECT_ADAPTER, payload.get('arguments')) or {},
        'id': str_field(payload, 'id'),
    }
    if signature:
        step['signature'] = signature
    return step


def function_result_value(value: object) -> dict[str, object] | str | list[Content] | None:
    """A stored function result as one of the wire shapes (object, string or content blocks), else None."""
    if isinstance(value, str):
        return value
    as_object = read_as(JSON_OBJECT_ADAPTER, value)
    if as_object is not None:
        return as_object
    return read_as(CONTENT_LIST_ADAPTER, value)


def to_server_function_result_step(payload: Mapping[str, object]) -> FunctionResultStep:
    """Replay a function_result that was stashed as custom.

    ``is_error`` is sent back as stored, ``False`` included (like the other
    tool results); it's left out only when the server didn't send it.
    """
    step: FunctionResultStep = {
        'type': 'function_result',
        'name': str_field(payload, 'name'),
        'call_id': str_field(payload, 'call_id', 'callId'),
    }
    result = function_result_value(payload.get('result'))
    if result is not None:
        step['result'] = result
    for key in ('is_error', 'isError'):
        is_error = payload.get(key)
        if isinstance(is_error, bool):
            step['is_error'] = is_error
            break
    return step


def to_server_tool_step(custom: Mapping[str, object], metadata: Mapping[str, object]) -> Step | None:
    """Compile the custom parts that stand in for Google-side tool activity.

    Search and code execution run on Google's side, so Genkit has no first-class
    part for them and we round-trip them through custom parts instead. Fields
    the part doesn't have are left out, as in JS.
    """
    signature = signature_of(metadata)
    if (search := read_as(GOOGLE_SEARCH_CALL_PART, custom.get(GOOGLE_SEARCH_CALL))) is not None:
        search_call: GoogleSearchCallStep = {'type': 'google_search_call', 'id': search.get('id', '')}
        if 'arguments' in search:
            search_call['arguments'] = search['arguments']
        if search_type := search.get('searchType'):
            search_call['search_type'] = search_type
        if signature:
            search_call['signature'] = signature
        return search_call
    if (found := read_as(GOOGLE_SEARCH_RESULT_PART, custom.get(GOOGLE_SEARCH_RESULT))) is not None:
        search_result: GoogleSearchResultStep = {'type': 'google_search_result', 'call_id': found.get('callId', '')}
        if 'result' in found:
            search_result['result'] = found['result']
        if 'isError' in found:
            search_result['is_error'] = found['isError']
        if signature:
            search_result['signature'] = signature
        return search_result
    if (code := read_as(EXECUTABLE_CODE_PART, custom.get(EXECUTABLE_CODE))) is not None:
        # The Interactions API takes 'python' or 'bash'; generateContent parts
        # say 'PYTHON', which it rejects (400, seen live).
        arguments: CodeArguments = {'language': code.get('language', '').lower() or CODE_LANGUAGE}
        if 'code' in code:
            arguments = {'code': code['code'], **arguments}
        code_call: CodeExecutionCallStep = {
            'type': 'code_execution_call',
            'id': str_field(metadata, CALL_ID),
            'arguments': arguments,
        }
        if signature:
            code_call['signature'] = signature
        return code_call
    if (output := read_as(CODE_EXECUTION_RESULT_PART, custom.get(CODE_EXECUTION_RESULT))) is not None:
        code_result: CodeExecutionResultStep = {
            'type': 'code_execution_result',
            'call_id': str_field(metadata, CALL_ID),
        }
        if 'output' in output:
            code_result['result'] = output['output']
        if signature:
            code_result['signature'] = signature
        return code_result
    return to_processing_step(custom, signature) or to_built_in_tool_step(custom, signature)


def to_processing_step(custom: Mapping[str, object], signature: str | None) -> Step | None:
    """Rebuild an agentic video ``processing_call`` / ``processing_result`` step, as it came back."""
    if (call := read_as(CALL_PART, custom.get(PROCESSING_CALL))) is not None:
        processing_call: ProcessingCallStep = {'type': 'processing_call', 'id': call.get('id', '')}
        if signature:
            processing_call['signature'] = signature
        return processing_call
    if (result := read_as(RESULT_PART, custom.get(PROCESSING_RESULT))) is not None:
        processing_result: ProcessingResultStep = {'type': 'processing_result', 'call_id': result.get('callId', '')}
        if signature:
            processing_result['signature'] = signature
        return processing_result
    return None


def to_built_in_tool_step(custom: Mapping[str, object], signature: str | None) -> Step | None:
    """Rebuild a server-side built-in tool step (url context, file search, Google Maps, MCP server).

    The reverse of ``from_built_in_tool_step``: the step goes back in the
    history with its signature, and fields the part doesn't have are left
    out (JS toBuiltInToolStep).
    """
    if (url_call := read_as(URL_CONTEXT_CALL_PART, custom.get(URL_CONTEXT_CALL))) is not None:
        url_context_call: UrlContextCallStep = {'type': 'url_context_call', 'id': url_call.get('id', '')}
        if 'arguments' in url_call:
            url_context_call['arguments'] = url_call['arguments']
        if signature:
            url_context_call['signature'] = signature
        return url_context_call
    if (url_result := read_as(URL_CONTEXT_RESULT_PART, custom.get(URL_CONTEXT_RESULT))) is not None:
        url_context_result: UrlContextResultStep = {
            'type': 'url_context_result',
            'call_id': url_result.get('callId', ''),
        }
        if 'result' in url_result:
            url_context_result['result'] = url_result['result']
        if 'isError' in url_result:
            url_context_result['is_error'] = url_result['isError']
        if signature:
            url_context_result['signature'] = signature
        return url_context_result
    if (file_call := read_as(CALL_PART, custom.get(FILE_SEARCH_CALL))) is not None:
        file_search_call: FileSearchCallStep = {'type': 'file_search_call', 'id': file_call.get('id', '')}
        if signature:
            file_search_call['signature'] = signature
        return file_search_call
    if (file_result := read_as(LIST_RESULT_PART, custom.get(FILE_SEARCH_RESULT))) is not None:
        file_search_result: FileSearchResultStep = {
            'type': 'file_search_result',
            'call_id': file_result.get('callId', ''),
        }
        if 'result' in file_result:
            file_search_result['result'] = file_result['result']
        if signature:
            file_search_result['signature'] = signature
        return file_search_result
    if (maps_call := read_as(GOOGLE_MAPS_CALL_PART, custom.get(GOOGLE_MAPS_CALL))) is not None:
        google_maps_call: GoogleMapsCallStep = {'type': 'google_maps_call', 'id': maps_call.get('id', '')}
        if 'arguments' in maps_call:
            google_maps_call['arguments'] = maps_call['arguments']
        if signature:
            google_maps_call['signature'] = signature
        return google_maps_call
    if (maps_result := read_as(LIST_RESULT_PART, custom.get(GOOGLE_MAPS_RESULT))) is not None:
        google_maps_result: GoogleMapsResultStep = {
            'type': 'google_maps_result',
            'call_id': maps_result.get('callId', ''),
        }
        if 'result' in maps_result:
            google_maps_result['result'] = maps_result['result']
        if signature:
            google_maps_result['signature'] = signature
        return google_maps_result
    return to_mcp_server_tool_step(custom)


def to_mcp_server_tool_step(custom: Mapping[str, object]) -> Step | None:
    """Rebuild an MCP server tool call or result step (MCP steps have no signature in the spec)."""
    if (mcp_call := read_as(MCP_SERVER_TOOL_CALL_PART, custom.get(MCP_SERVER_TOOL_CALL))) is not None:
        call_step: McpServerToolCallStep = {'type': 'mcp_server_tool_call', 'id': mcp_call.get('id', '')}
        if name := mcp_call.get('name'):
            call_step['name'] = name
        if server_name := mcp_call.get('serverName'):
            call_step['server_name'] = server_name
        if 'arguments' in mcp_call:
            call_step['arguments'] = mcp_call['arguments']
        return call_step
    if (mcp_result := read_as(MCP_SERVER_TOOL_RESULT_PART, custom.get(MCP_SERVER_TOOL_RESULT))) is not None:
        result_step: McpServerToolResultStep = {
            'type': 'mcp_server_tool_result',
            'call_id': mcp_result.get('callId', ''),
        }
        if name := mcp_result.get('name'):
            result_step['name'] = name
        if server_name := mcp_result.get('serverName'):
            result_step['server_name'] = server_name
        if 'result' in mcp_result:
            result_step['result'] = mcp_result['result']
        return result_step
    return None


def unknown_step_from_custom(custom: Mapping[str, object]) -> Step | None:
    """Resend a step stored as ``unknownStep`` as it came back.

    ``None`` values are dropped first: parts saved by older plugin versions
    can hold fields the SDK filled in with ``None`` that the server never sent.
    """
    return read_as(STEP_ADAPTER, without_none(custom.get(UNKNOWN_STEP)))


def to_standalone_step(part: Part) -> Step | None:
    """Return the step a part compiles to on its own, or None if it is inline content."""
    signature = signature_of(part.metadata)
    if part.tool_request:
        return to_function_call_step(part.tool_request, signature)
    if part.tool_response:
        return to_function_result_step(part.tool_response, part.metadata)
    if part.reasoning is not None:
        return to_thought_step(part)
    if part.custom is not None:
        custom = part.custom
        original_thought = thought_from_custom(custom)
        if original_thought is not None:
            return original_thought
        unknown = unknown_step_from_custom(custom)
        if unknown is not None:
            return unknown
        server_call = read_as(JSON_OBJECT_ADAPTER, custom.get(SERVER_FUNCTION_CALL))
        if server_call is not None:
            return to_server_function_call_step(server_call, signature)
        server_result = read_as(JSON_OBJECT_ADAPTER, custom.get(SERVER_FUNCTION_RESULT))
        if server_result is not None:
            return to_server_function_result_step(server_result)
        return to_server_tool_step(custom, part.metadata or {})
    return None


def to_content_step(role: InteractionRole, content: list[Content]) -> Step:
    """Wrap a turn's inline content in the step for whoever produced it."""
    if role == 'model':
        model_step: ModelOutputStep = {'type': 'model_output', 'content': content}
        return model_step
    user_step: UserInputStep = {'type': 'user_input', 'content': content}
    return user_step


def ensure_code_execution_ids(messages: list[Message]) -> list[Message]:
    """Give code execution calls and results without a call ID matching IDs.

    History saved from generateContent has no ``callId`` on its
    ``executableCode`` / ``codeExecutionResult`` parts, and the Interactions
    API rejects an empty id. A call without one gets
    ``genkit-code-execution-id-N`` and the next result without one gets the
    same ID; a result with no call waiting gets
    ``genkit-code-execution-orphan-id-N`` (like ``ensure_tool_ids`` does for
    tool requests). Parts that have an ID keep it. Returns copies.
    """
    pending: list[str] = []
    next_id = 0
    result: list[Message] = []
    for message in messages:
        content: list[Part] = []
        for part in message.content:
            custom = part.custom or {}
            is_call = EXECUTABLE_CODE in custom
            if (part.metadata or {}).get(CALL_ID) or not (is_call or CODE_EXECUTION_RESULT in custom):
                content.append(part)
                continue
            if is_call:
                call_id = f'genkit-code-execution-id-{next_id}'
                next_id += 1
                pending.append(call_id)
            elif pending:
                call_id = pending.pop(0)
            else:
                call_id = f'genkit-code-execution-orphan-id-{next_id}'
                next_id += 1
            content.append(with_metadata(part, CALL_ID, call_id))
        result.append(message.model_copy(update={'content': content}))
    return result


def to_interaction_steps(messages: list[Message]) -> list[Step]:
    """Convert Genkit messages to Interactions API steps.

    Text and media collapse into a content step, but tool calls, thoughts,
    and Google-side tool activity are steps of their own. Flush any buffered
    inline content before those so mixed parts stay in message order — a
    thought sitting between two sentences must not jump the first sentence
    to the end of the turn.
    """
    steps: list[Step] = []
    for message in ensure_code_execution_ids(messages):
        role = to_interaction_role(message.role)
        inline: list[Content] = []

        for part in message.content:
            step = to_standalone_step(part)
            if step is not None:
                if inline:
                    steps.append(to_content_step(role, list(inline)))
                    inline.clear()
                steps.append(step)
                continue
            content = to_interaction_content(part)
            if content is not None:
                inline.append(content)
        if inline:
            steps.append(to_content_step(role, inline))
    return steps


def with_metadata(part: Part, key: str, value: object) -> Part:
    """Return a copy of the part carrying one more metadata entry."""
    if not value:
        return part
    return part.model_copy(update={'metadata': {**(part.metadata or {}), key: value}})


def plain(value: object) -> object:
    """Unwrap SDK models into plain data so parts stay JSON-serializable."""
    if isinstance(value, BaseModel):
        return value.model_dump(mode='python')
    if isinstance(value, dict):
        return {key: plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [plain(item) for item in value]
    return value


def server_tool_part(
    key: str, payload: Mapping[str, object], *, signature: str | None, call_id: str | None = None
) -> Part:
    """Wrap Google-side tool activity in the custom part Genkit round-trips."""
    part = Part.from_custom({key: dict(payload)}, metadata={CALL_ID: call_id} if call_id else None)
    return with_metadata(part, THOUGHT_SIGNATURE, signature)


def from_google_search_call(step: GoogleSearchCallStep) -> Part:
    """Convert a google_search_call step to a Genkit custom part (fields left out when missing, as in JS)."""
    payload: dict[str, object] = {'id': step.get('id', '')}
    if 'arguments' in step:
        payload['arguments'] = step['arguments']
    if search_type := step.get('search_type'):
        payload['searchType'] = search_type
    return server_tool_part(GOOGLE_SEARCH_CALL, payload, signature=step.get('signature'))


def from_google_search_result(step: GoogleSearchResultStep) -> Part:
    """Convert a google_search_result step to a Genkit custom part (fields left out when missing, as in JS)."""
    payload: dict[str, object] = {CALL_ID: step.get('call_id', '')}
    if 'result' in step:
        payload['result'] = step['result']
    if 'is_error' in step:
        payload['isError'] = step['is_error']
    return server_tool_part(GOOGLE_SEARCH_RESULT, payload, signature=step.get('signature'))


def from_code_execution_call(step: CodeExecutionCallStep) -> Part:
    """Convert a code_execution_call step to a Genkit custom part."""
    arguments = step.get('arguments') or {}
    return server_tool_part(
        EXECUTABLE_CODE,
        {'code': arguments.get('code'), 'language': arguments.get('language') or CODE_LANGUAGE},
        signature=step.get('signature'),
        call_id=step.get('id'),
    )


def from_code_execution_result(step: CodeExecutionResultStep) -> Part:
    """Convert a code_execution_result step to a Genkit custom part."""
    result = step.get('result')
    return server_tool_part(
        CODE_EXECUTION_RESULT,
        {
            'output': result if isinstance(result, str) else json.dumps(result),
            'outcome': 'OUTCOME_OK',
        },
        signature=step.get('signature'),
        call_id=step.get('call_id'),
    )


def from_function_call_step(step: FunctionCallStep) -> Part:
    """Convert a pending client function_call step to a Genkit tool request part."""
    part = Part(
        tool_request=ToolRequest(
            name=step.get('name', ''),
            input=step.get('arguments', {}),
            ref=step.get('id'),
        )
    )
    return with_metadata(part, THOUGHT_SIGNATURE, step.get('signature'))


def from_server_function_call_step(step: FunctionCallStep) -> Part:
    """Stash an already-resolved or sandbox function_call as custom so Genkit's tool loop skips it."""
    payload: dict[str, object] = {
        'id': step.get('id'),
        'name': step.get('name', ''),
        'arguments': step.get('arguments', {}),
    }
    return with_metadata(
        Part.from_custom({SERVER_FUNCTION_CALL: payload}),
        THOUGHT_SIGNATURE,
        step.get('signature'),
    )


def from_function_result_step(step: FunctionResultStep) -> Part:
    """Stash a function_result as custom — never a ToolResponsePart.

    The API echoes the client's own tool result back on the model tape.
    Putting that on a ToolResponsePart would send it again next turn.
    """
    payload: dict[str, object] = {
        'name': step.get('name', ''),
        'result': step.get('result'),
        'call_id': step.get('call_id'),
    }
    # Kept only when the server sent it, so it's resent the same way.
    if 'is_error' in step:
        payload['is_error'] = step['is_error']
    return Part.from_custom({SERVER_FUNCTION_RESULT: payload})


def from_processing_step(step: ProcessingCallStep | ProcessingResultStep) -> Part:
    """Convert an agentic video processing step to a custom part with its signature (JS)."""
    if step['type'] == 'processing_call':
        part = Part.from_custom({PROCESSING_CALL: {'id': step.get('id', '')}})
    else:
        part = Part.from_custom({PROCESSING_RESULT: {CALL_ID: step.get('call_id', '')}})
    return with_metadata(part, THOUGHT_SIGNATURE, step.get('signature'))


def add_mcp_names(payload: dict[str, object], step: McpServerToolCallStep | McpServerToolResultStep) -> None:
    """Copy an MCP step's tool and server names onto its custom part, when set."""
    if name := step.get('name'):
        payload['name'] = name
    if server_name := step.get('server_name'):
        payload['serverName'] = server_name


def from_built_in_tool_step(
    step: UrlContextCallStep
    | UrlContextResultStep
    | FileSearchCallStep
    | FileSearchResultStep
    | GoogleMapsCallStep
    | GoogleMapsResultStep
    | McpServerToolCallStep
    | McpServerToolResultStep,
) -> Part:
    """Convert a server-side built-in tool step to a custom part (JS fromBuiltInToolStep).

    The signature goes in ``metadata.thoughtSignature``; fields the server
    left out are left out. ``to_built_in_tool_step`` turns the part back into
    the step when the history is resent. Never a ``toolRequest``: the server
    ran the tool, so there's nothing for Genkit to run.
    """
    payload: dict[str, object] = {}
    # MCP steps have no signature in the spec.
    if step['type'] == 'mcp_server_tool_call':
        payload['id'] = step.get('id', '')
        add_mcp_names(payload, step)
        if 'arguments' in step:
            payload['arguments'] = step['arguments']
        return Part.from_custom({MCP_SERVER_TOOL_CALL: payload})
    if step['type'] == 'mcp_server_tool_result':
        payload[CALL_ID] = step.get('call_id', '')
        add_mcp_names(payload, step)
        if 'result' in step:
            payload['result'] = step['result']
        return Part.from_custom({MCP_SERVER_TOOL_RESULT: payload})

    signature = step.get('signature')
    if step['type'] == 'url_context_call':
        key = URL_CONTEXT_CALL
        payload['id'] = step.get('id', '')
        if 'arguments' in step:
            payload['arguments'] = step['arguments']
    elif step['type'] == 'url_context_result':
        key = URL_CONTEXT_RESULT
        payload[CALL_ID] = step.get('call_id', '')
        if 'result' in step:
            payload['result'] = step['result']
        if 'is_error' in step:
            payload['isError'] = step['is_error']
    elif step['type'] == 'file_search_call':
        key = FILE_SEARCH_CALL
        payload['id'] = step.get('id', '')
    elif step['type'] == 'file_search_result':
        key = FILE_SEARCH_RESULT
        payload[CALL_ID] = step.get('call_id', '')
        if 'result' in step:
            payload['result'] = step['result']
    elif step['type'] == 'google_maps_call':
        key = GOOGLE_MAPS_CALL
        payload['id'] = step.get('id', '')
        if 'arguments' in step:
            payload['arguments'] = step['arguments']
    else:
        key = GOOGLE_MAPS_RESULT
        payload[CALL_ID] = step.get('call_id', '')
        if 'result' in step:
            payload['result'] = step['result']
    return with_metadata(Part.from_custom({key: payload}), THOUGHT_SIGNATURE, signature)


def interaction_media_mime_type(content: ImageContent | AudioContent | DocumentContent | VideoContent) -> str | None:
    """The content type of an Interactions media block.

    Audio that comes with ``data`` but no ``mime_type`` is raw 16-bit PCM: the
    server streams single-voice TTS that way (``{type: 'audio', data,
    sample_rate, channels}``, seen live Nov 2026), while it labels the same
    audio ``audio/l16; rate=24000; channels=1`` in other cases (e.g. two
    speakers). The label is built from the ``sample_rate`` and ``channels`` it
    sends (24000 and 1 when missing), so the audio can be played and joined.
    A given ``mime_type`` always wins.
    """
    mime_type = content.get('mime_type')
    data = content.get('data')
    if mime_type or content['type'] != 'audio' or not data:
        return mime_type
    sample_rate = content.get('sample_rate') or 24000
    channels = content.get('channels') or 1
    return f'audio/l16; rate={sample_rate}; channels={channels}'


def from_media_content(content: ImageContent | AudioContent | DocumentContent | VideoContent) -> Part:
    """Convert wire media content to a Genkit media part."""
    mime_type = interaction_media_mime_type(content)
    url = content.get('uri')
    data = content.get('data')
    if data and mime_type:
        url = f'data:{mime_type};base64,{data}'
    return Part(media=Media(url=url or '', content_type=mime_type))


def from_text_content(content: TextContent) -> Part:
    """Convert wire text content to a Genkit text part."""
    # Empty annotations still show up in metadata so round-trips stay stable.
    return Part.from_text(content.get('text', ''), metadata={'annotations': content.get('annotations')})


def from_visual_content(content: ImageContent | VideoContent) -> Part:
    """Convert wire image or video content, keeping the resolution it came back at."""
    return with_metadata(from_media_content(content), 'resolution', content.get('resolution'))


def from_thought_step(step: ThoughtStep) -> Part:
    """Convert a thought step to a Genkit reasoning part."""
    reasoning = '\n'.join(
        item.get('text', '') if not isinstance(item, UnknownContent) and item['type'] == 'text' else '[Image]'
        for item in step.get('summary') or []
    )
    # The signature can come in a step of its own (summary and signature are
    # sometimes separate steps), so it's set only when this step has one.
    part = Part(reasoning=reasoning, custom={THOUGHT_CUSTOM: dict(step)})
    return with_metadata(part, THOUGHT_SIGNATURE, step.get('signature'))


def from_interaction_content(content: Content) -> Part:
    """Convert an Interactions content block back to a Genkit part.

    Thought and function call/result blocks (content, not steps) and unknown
    types come back as a custom ``unknownContent`` part.
    """
    if not isinstance(content, UnknownContent):
        if content['type'] == 'text':
            return from_text_content(content)
        if content['type'] == 'image' or content['type'] == 'video':
            return from_visual_content(content)
        if content['type'] == 'audio' or content['type'] == 'document':
            return from_media_content(content)
    return Part.from_custom({UNKNOWN_CONTENT: dict(content)})


def get_pending_function_call_ids(steps: list[Step], status: str | None) -> set[str]:
    """Return IDs of function_call steps awaiting client execution on a paused turn.

    Sandbox filesystem tools (such as ``write_to_file``) also appear as
    ``function_call`` steps on Antigravity, paired with a matching
    ``function_result`` in the same interaction. Only unmatched calls on a
    ``requires_action`` turn become ``ToolRequest`` parts for Genkit's loop.
    """
    if status != 'requires_action':
        return set()
    resolved: set[str] = set()
    calls: list[str] = []
    for step in steps:
        if isinstance(step, UnknownStep):
            continue
        if step['type'] == 'function_result' and (call_id := step.get('call_id')):
            resolved.add(call_id)
        elif step['type'] == 'function_call' and (call := step.get('id')):
            calls.append(call)
    return {call for call in calls if call not in resolved}


def is_pending_call(step: Step, pending_ids: set[str]) -> bool:
    """True for a function_call step waiting for the client (see ``get_pending_function_call_ids``)."""
    if isinstance(step, UnknownStep) or step['type'] != 'function_call':
        return False
    call_id = step.get('id')
    return bool(call_id) and call_id in pending_ids


def from_interaction_step(step: Step, *, is_pending: bool = False) -> list[Part]:
    """Convert an Interactions step to Genkit parts."""
    if isinstance(step, UnknownStep):
        return [Part.from_custom({UNKNOWN_STEP: dict(step)})]
    if step['type'] == 'model_output':
        return [from_interaction_content(content) for content in step.get('content') or []]
    if step['type'] == 'user_input':
        # The API echoes our prompt back; including it would duplicate the input.
        return []
    if step['type'] == 'google_search_call':
        return [from_google_search_call(step)]
    if step['type'] == 'google_search_result':
        return [from_google_search_result(step)]
    if step['type'] == 'code_execution_call':
        return [from_code_execution_call(step)]
    if step['type'] == 'code_execution_result':
        return [from_code_execution_result(step)]
    if step['type'] == 'thought':
        return [from_thought_step(step)]
    if step['type'] == 'function_call':
        return [from_function_call_step(step)] if is_pending else [from_server_function_call_step(step)]
    if step['type'] == 'function_result':
        return [from_function_result_step(step)]
    if step['type'] == 'processing_call' or step['type'] == 'processing_result':
        return [from_processing_step(step)]
    return [from_built_in_tool_step(step)]


def interaction_message_metadata(interaction: GeminiInteraction) -> dict[str, Any] | None:
    """Build message.metadata fields that identify the source Interaction."""
    metadata: dict[str, Any] = {}
    if interaction_id := interaction.get('id'):
        metadata['interactionId'] = interaction_id
    if environment_id := interaction.get('environment_id'):
        metadata['environmentId'] = environment_id
    if status := interaction.get('status'):
        # Preserve the wire status when finish_reason is a coarser Genkit enum.
        metadata['interactionStatus'] = status
    return metadata or None


def snake_to_camel(name: str) -> str:
    """``google_search`` -> ``googleSearch``."""
    return re.sub(r'_([a-z0-9])', lambda match: match.group(1).upper(), name)


def usage_from_interaction(usage: Usage) -> GenerationUsage:
    """Map Interactions Usage onto Genkit GenerationUsage.

    The per-modality breakdowns are token counts, so they go in ``custom``
    as e.g. ``inputTextTokens`` / ``outputImageTokens``: Genkit's
    ``input_characters``, ``input_images``, ... count characters and media
    parts (``with_basic_usage_stats`` adds those where the request is known).
    """
    response_usage = GenerationUsage(
        input_tokens=usage.get('total_input_tokens'),
        output_tokens=usage.get('total_output_tokens'),
        total_tokens=usage.get('total_tokens'),
        cached_content_tokens=usage.get('total_cached_tokens'),
        thoughts_tokens=usage.get('total_thought_tokens'),
    )
    # Genkit has no standard fields for these.
    custom: dict[str, float] = {}

    def add_modality_tokens(direction: str, breakdown: list[ModalityTokens] | None) -> None:
        for item in breakdown or []:
            modality = item.get('modality')
            tokens = item.get('tokens')
            if modality and isinstance(tokens, int):
                key = f'{direction}{modality[:1].upper()}{modality[1:]}Tokens'
                custom[key] = custom.get(key, 0) + tokens

    add_modality_tokens('input', usage.get('input_tokens_by_modality'))
    add_modality_tokens('output', usage.get('output_tokens_by_modality'))
    # Tokens loaded by tools, e.g. video frames read during agentic video
    # understanding.
    if tool_use_tokens := usage.get('total_tool_use_tokens'):
        custom['toolUseTokens'] = tool_use_tokens
    # How many times each grounding tool was used, e.g. `googleSearchCount`.
    for tool_count in usage.get('grounding_tool_count') or []:
        tool = tool_count.get('type')
        count = tool_count.get('count')
        if tool and isinstance(count, int):
            key = f'{snake_to_camel(tool)}Count'
            custom[key] = custom.get(key, 0) + count
    if custom:
        response_usage.custom = custom
    return response_usage


def with_basic_usage_stats(response: ModelResponse, request_messages: list[Message]) -> ModelResponse:
    """Add character and media-part counts to an Interactions response's usage.

    ``input_characters``, ``input_images``, ... are counted from the request
    messages and the response message, the same way as on generateContent
    models (``GenerationUsage.from_messages``). Telemetry reads these fields.
    The token counts from the API are kept and win over the basic counts.

    Call it on the final response, with the request's messages. Background
    models (Deep Research) don't have the request when checked, so they don't
    get these counts.
    """
    output = response.message or Message(role='model', content=[])
    usage = GenerationUsage.from_messages(request_messages, output)
    if response.usage is not None:
        for field, value in response.usage:
            if value is not None:
                setattr(usage, field, value)
    response.usage = usage
    return response


def parts_from_steps(steps: list[Step], *, status: str | None = None) -> list[Part]:
    """Flatten interaction steps into Genkit parts, dropping empty ones."""
    pending_ids = get_pending_function_call_ids(steps, status)
    return [
        part
        for step in steps
        for part in from_interaction_step(step, is_pending=is_pending_call(step, pending_ids))
        if part.model_dump(exclude_none=True)
    ]


def model_response(
    interaction: GeminiInteraction,
    *,
    content: list[Part],
    finish_reason: FinishReason,
    finish_message: str | None = None,
) -> ModelResponse:
    """Build a ModelResponse that carries the raw interaction (as the server sent it) alongside its parts.

    Only ``raw``: Genkit deprecates the response-level ``custom`` in favor of
    ``raw``.
    """
    response = ModelResponse.model_construct(
        finish_reason=finish_reason,
        finish_message=finish_message,
        message=Message(role='model', content=content, metadata=interaction_message_metadata(interaction)),
        raw=dict(interaction),
    )
    usage = interaction.get('usage')
    if usage is not None:
        response.usage = usage_from_interaction(usage)
    return response


def cancelled_response(interaction: GeminiInteraction) -> ModelResponse:
    """Build the ModelResponse for a cancelled Interaction."""
    return model_response(
        interaction,
        content=[Part.from_text('Operation cancelled.')],
        finish_reason=FinishReason.ABORTED,
        finish_message='Operation cancelled',
    )


def steps_response(
    interaction: GeminiInteraction,
    *,
    finish_reason: FinishReason,
    finish_message: str | None = None,
) -> ModelResponse | None:
    """Build a ModelResponse from interaction steps, or None when there are none."""
    steps = list(interaction.get('steps') or [])
    if not steps:
        return None
    return model_response(
        interaction,
        content=parts_from_steps(steps, status=interaction.get('status')),
        finish_reason=finish_reason,
        finish_message=finish_message,
    )


def completed_response(interaction: GeminiInteraction) -> ModelResponse | None:
    """Build the ModelResponse for a completed Interaction, if it has steps."""
    return steps_response(interaction, finish_reason=FinishReason.STOP)


def from_interaction_sync(interaction: GeminiInteraction) -> ModelResponse:
    """Convert a finished or paused interaction to a synchronous model response.

    ``requires_action`` is a deliberate pause for client function calling, so
    unmatched ``function_call`` steps return as ``ToolRequest`` parts while
    already-resolved sandbox calls stay in ``custom['serverFunctionCall']``.
    Truncated or over-budget turns keep their real finish reason so a
    generate that ran out of tokens does not look like a clean stop.
    In-flight statuses raise FAILED_PRECONDITION — this helper is not a poll
    loop. A failed interaction raises a GenkitError with the status its
    error code maps to, or UNKNOWN when there's no usable code, except a
    content-policy block, which returns a response with finish_reason
    'blocked'.
    """
    status = interaction.get('status')
    if status == 'failed':
        if (blocked := blocked_response(interaction)) is not None:
            return blocked
        failure = ValueError(interaction_error_message(interaction) or FAILED_MESSAGE)
        raise interaction_error(interaction, failure) from failure
    if status == 'cancelled':
        return cancelled_response(interaction)
    if status in ('in_progress', 'queued'):
        raise GenkitError(
            status='FAILED_PRECONDITION',
            message=f'Interaction is still running (status={status!r})',
        )
    if status is not None and (partial := PARTIAL_TERMINAL_STATUSES.get(status)):
        finish_reason, message = partial
        response = steps_response(interaction, finish_reason=finish_reason, finish_message=message)
        if response is None:
            raise ValueError(message)
        return response
    if status in ('completed', 'requires_action') or status is None:
        return completed_response(interaction) or model_response(
            interaction, content=[], finish_reason=FinishReason.STOP
        )
    raise ValueError(f'Unknown interaction status: {status!r}')


def from_interaction(interaction: GeminiInteraction) -> Operation:
    """Convert an interaction poll result to a Genkit operation."""
    op = Operation.model_construct(id=interaction.get('id') or '')

    status = interaction.get('status')
    if status in ('in_progress', 'queued'):
        # Keep polling — still running or waiting on the server.
        op.done = False
    elif status == 'cancelled':
        op.done = True
        op.output = cancelled_response(interaction)
    elif status == 'completed':
        op.done = True
        op.output = completed_response(interaction) or model_response(
            interaction, content=[], finish_reason=FinishReason.STOP
        )
    elif status is not None and (partial := PARTIAL_TERMINAL_STATUSES.get(status)):
        # Halted for length or budget. Prefer whatever landed over a bare error.
        finish_reason, message = partial
        op.done = True
        op.output = steps_response(interaction, finish_reason=finish_reason, finish_message=message)
        if op.output is None:
            op.error = Failure(message=message)
    elif status == 'failed':
        # Always exit the poll loop on failure; leaving done unset hangs forever.
        op.done = True
        if (blocked := blocked_response(interaction)) is not None:
            op.output = blocked
        else:
            op.error = Failure(message=interaction_error_message(interaction) or FAILED_MESSAGE)
    elif status == 'requires_action':
        # Chat generate can pause for a tool. A background job cannot — there
        # is no interrupt/resume on generate_operation — so end the poll
        # instead of hanging `while not op.done` or inventing a resume handle.
        op.done = True
        op.error = Failure(message=BACKGROUND_INTERRUPT_UNSUPPORTED)
    else:
        # A status we do not map still has to exit `while not op.done`.
        op.done = True
        op.error = Failure(message=f'Unknown interaction status: {status!r}')
    return op
