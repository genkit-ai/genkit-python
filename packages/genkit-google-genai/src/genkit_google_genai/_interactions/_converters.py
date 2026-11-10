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
from typing import Any, Literal, cast
from urllib.parse import urlparse

from genkit_google_genai._interactions._errors import interaction_error_status, is_interaction_content_block_code

# Private SDK module: the SDK exports this type only here. It goes away with
# the SDK types when the plugin gets its own Interactions types.
from google.genai._gaos.types import UnrecognizedStr
from google.genai.interactions import (
    AudioContent,
    AudioContentParam,
    CodeExecutionCallStep,
    CodeExecutionCallStepParam,
    CodeExecutionResultStep,
    CodeExecutionResultStepParam,
    Content,
    ContentParam,
    DocumentContent,
    DocumentContentParam,
    FunctionCallStep,
    FunctionCallStepParam,
    FunctionParam,
    FunctionResultStep,
    FunctionResultStepParam,
    FunctionResultStepResultUnionParam,
    GoogleSearchCallStep,
    GoogleSearchCallStepParam,
    GoogleSearchResultStep,
    GoogleSearchResultStepParam,
    ImageContent,
    ImageContentParam,
    Interaction,
    ModalityTokens,
    ModelOutputStep,
    ModelOutputStepParam,
    Step,
    StepParam,
    TextContent,
    TextContentParam,
    ThoughtStep,
    ThoughtStepParam,
    UnknownContent,
    Usage,
    UserInputStep,
    UserInputStepParam,
    VideoContent,
    VideoContentParam,
)
from pydantic import BaseModel

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
UNKNOWN_STEP = 'unknownStep'
SERVER_FUNCTION_CALL = 'serverFunctionCall'
SERVER_FUNCTION_RESULT = 'serverFunctionResult'
THOUGHT_CUSTOM = 'thought'
# Wire function_result.content is only text/image blocks, not a JSON array.
CONTENT_BLOCK_TYPES = frozenset({'text', 'image'})
# The language when a code execution step names none.
CODE_LANGUAGE: Literal['python'] = 'python'

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


def interaction_error_message(interaction: Interaction) -> str | None:
    """Read the failure message a failed Interaction reports.

    Prefer the top-level errors list when the job died before any output
    step, then fall back to a step error so callers see the server's reason
    instead of a generic 'Interaction failed'.
    """
    for err in interaction.errors or []:
        message = getattr(err, 'message', None)
        if message:
            return str(message)
        if isinstance(err, dict) and err.get('message'):
            return str(err['message'])
    for step in interaction.steps or []:
        if isinstance(step, ModelOutputStep) and step.error is not None and step.error.message:
            return step.error.message
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


def first_error(interaction: Interaction) -> tuple[object, str | None]:
    """The code and message of the interaction's first top-level error, if any."""
    for err in interaction.errors or []:
        return err.code, err.message
    return None, None


def blocked_response(interaction: Interaction) -> ModelResponse | None:
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


def interaction_error(interaction: Interaction, cause: BaseException) -> GenkitError:
    """The error for the first failure on an Interaction that has a usable code, else UNKNOWN."""
    for err in interaction.errors or []:
        code = err.get('code') if isinstance(err, dict) else getattr(err, 'code', None)
        if error := error_from_provider_code(code, cause):
            return error
    for step in interaction.steps or []:
        if isinstance(step, ModelOutputStep) and step.error is not None:
            if error := error_from_provider_code(step.error.code, cause):
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


def to_interaction_tool(tool: ToolDefinition) -> FunctionParam:
    """Convert a Genkit tool definition to an Interactions function tool."""
    func: FunctionParam = {
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


def to_interaction_content(part: Part) -> ContentParam | None:
    """Convert a Genkit part to an Interactions content block."""
    if part.text is not None:
        text: TextContentParam = {'type': 'text', 'text': part.text}
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


def video_block(part: Part, *, mime_type: str | None = None) -> VideoContentParam:
    """A video block, with ``metadata['processing']`` (`static` / `agentic`) when the part sets it."""
    video: VideoContentParam = {'type': 'video'}
    if mime_type:
        video['mime_type'] = UnrecognizedStr(mime_type)
    processing = (part.metadata or {}).get('processing')
    if isinstance(processing, str):
        video['processing'] = UnrecognizedStr(processing)
    return video


def to_interaction_media(part: Part) -> ContentParam:
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
        unknown: DocumentContentParam = {'type': 'document', 'uri': url}
        return unknown

    # The SDK types each mime_type as a few known values or its own
    # `UnrecognizedStr`; any other type (e.g. Word) goes through that.
    mime_type = UnrecognizedStr(content_type)
    block: ImageContentParam | AudioContentParam | VideoContentParam | DocumentContentParam
    if content_type.startswith('image/'):
        block = {'type': 'image', 'mime_type': mime_type}
    elif content_type.startswith('audio/'):
        block = {'type': 'audio', 'mime_type': mime_type}
    elif content_type.startswith('video/'):
        block = video_block(part, mime_type=content_type)
    else:
        block = {'type': 'document', 'mime_type': mime_type}

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


def with_signature(step: StepParam, metadata: dict[str, Any]) -> StepParam:
    """Carry a thought signature from part metadata back onto its step."""
    signature = metadata.get(THOUGHT_SIGNATURE)
    if signature is not None:
        cast(dict[str, Any], step)['signature'] = signature
    return step


def to_function_call_step(request: ToolRequest) -> FunctionCallStepParam:
    """Compile a Genkit tool request into a function_call step.

    Arguments on the wire are a JSON object. A scalar like ``'Paris'`` would
    silently become ``{}`` and the model would see a different call than the
    one the caller made.
    """
    dumped = plain(request.input)
    if dumped is None:
        args: dict[str, Any] = {}
    elif not isinstance(dumped, dict):
        raise GenkitError(
            status='INVALID_ARGUMENT',
            message=(
                'Tool request input must be a JSON object (got '
                f'{type(dumped).__name__}). Wrap scalars as '
                "{'city': 'Paris'} rather than passing a bare string."
            ),
        )
    else:
        args = cast(dict[str, Any], dumped)
    return {
        'type': 'function_call',
        'name': request.name,
        'arguments': args,
        'id': request.ref or '',
    }


def is_content_block_list(value: object) -> bool:
    """True when value is a non-empty list of text/image content blocks."""
    if not isinstance(value, list) or not value:
        return False
    for item in value:
        if not isinstance(item, dict):
            return False
        block = cast(dict[str, object], item)
        if block.get('type') not in CONTENT_BLOCK_TYPES:
            return False
    return True


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
) -> FunctionResultStepParam:
    """Compile a Genkit tool response into a function_result step.

    Ordinary JSON arrays (``[1, 2]``) are not multimodal content — they
    have to be boxed. Only a list of text/image blocks goes on the wire as
    content. A failed result keeps ``is_error`` so the model sees the miss.
    """
    output = plain(response.output)
    if isinstance(output, (str, dict)):
        result = cast(FunctionResultStepResultUnionParam, output)
    elif is_content_block_list(output):
        result = cast(FunctionResultStepResultUnionParam, output)
    elif output is None:
        result = cast(FunctionResultStepResultUnionParam, {})
    else:
        result = cast(FunctionResultStepResultUnionParam, {'result': output})
    step: FunctionResultStepParam = {
        'type': 'function_result',
        'name': response.name,
        'result': result,
        'call_id': response.ref or '',
    }
    if metadata_flag(metadata, 'isError', 'is_error'):
        step['is_error'] = True
    return step


def thought_from_custom(custom: dict[str, Any] | None) -> StepParam | None:
    """Replay a thought step that was stashed on the part, images and all.

    Rebuilding from ``reasoning`` would drop image summaries and the
    signature the next turn needs to continue the same thought.
    """
    if not custom:
        return None
    thought = custom.get(THOUGHT_CUSTOM)
    if isinstance(thought, dict) and thought.get('type') == 'thought':
        return cast(StepParam, copy.deepcopy(thought))
    return None


def to_thought_step(part: Part) -> StepParam:
    """Compile a Genkit reasoning part into a thought step."""
    original = thought_from_custom(part.custom)
    if original is not None:
        return original
    thought: ThoughtStepParam = {
        'type': 'thought',
        'summary': [{'type': 'text', 'text': part.reasoning or ''}],
    }
    return with_signature(thought, part.metadata or {})


def to_server_function_call_step(payload: dict[str, Any]) -> FunctionCallStepParam:
    """Replay a server-executed function_call that was stashed as custom."""
    raw_args = payload.get('arguments')
    return {
        'type': 'function_call',
        'name': payload.get('name') or '',
        'arguments': cast(dict[str, Any], raw_args) if isinstance(raw_args, dict) else {},
        'id': payload.get('id') or '',
    }


def to_server_function_result_step(payload: dict[str, Any]) -> FunctionResultStepParam:
    """Replay a function_result that was stashed as custom, including is_error."""
    step: FunctionResultStepParam = {
        'type': 'function_result',
        'name': payload.get('name') or '',
        'result': cast(FunctionResultStepResultUnionParam, payload.get('result')),
        'call_id': payload.get('call_id') or payload.get('callId') or '',
    }
    if payload.get('is_error') or payload.get('isError'):
        step['is_error'] = True
    return step


def to_server_tool_step(custom: dict[str, Any], metadata: dict[str, Any]) -> StepParam | None:
    """Compile the custom parts that stand in for Google-side tool activity.

    Search and code execution run on Google's side, so Genkit has no first-class
    part for them and we round-trip them through custom parts instead.
    """
    if GOOGLE_SEARCH_CALL in custom:
        call = custom[GOOGLE_SEARCH_CALL]
        search_call: GoogleSearchCallStepParam = {
            'type': 'google_search_call',
            'id': call.get('id', ''),
            'arguments': call.get('arguments'),
        }
        return search_call
    if GOOGLE_SEARCH_RESULT in custom:
        result = custom[GOOGLE_SEARCH_RESULT]
        search_result: GoogleSearchResultStepParam = {
            'type': 'google_search_result',
            'call_id': result.get(CALL_ID, ''),
            'result': result.get('result'),
        }
        return search_result
    if EXECUTABLE_CODE in custom:
        code = custom[EXECUTABLE_CODE]
        stored_language = code.get('language')
        # The Interactions API takes 'python' or 'bash'; generateContent parts
        # say 'PYTHON', which it rejects (400, seen live). Cast: the SDK types
        # `language` as 'python' only; our own Interactions types remove it.
        language = cast(
            Literal['python'],
            (stored_language.lower() if isinstance(stored_language, str) else '') or CODE_LANGUAGE,
        )
        code_call: CodeExecutionCallStepParam = {
            'type': 'code_execution_call',
            'id': metadata.get(CALL_ID, ''),
            'arguments': {'code': code.get('code'), 'language': language},
        }
        return code_call
    if CODE_EXECUTION_RESULT in custom:
        code_result: CodeExecutionResultStepParam = {
            'type': 'code_execution_result',
            'call_id': metadata.get(CALL_ID, ''),
            'result': custom[CODE_EXECUTION_RESULT].get('output'),
        }
        return code_result
    return None


def to_standalone_step(part: Part) -> StepParam | None:
    """Return the step a part compiles to on its own, or None if it is inline content."""
    if part.tool_request:
        return with_signature(to_function_call_step(part.tool_request), part.metadata or {})
    if part.tool_response:
        return to_function_result_step(part.tool_response, part.metadata)
    if part.reasoning is not None:
        return to_thought_step(part)
    if part.custom is not None:
        custom = part.custom or {}
        original_thought = thought_from_custom(custom)
        if original_thought is not None:
            return original_thought
        unknown = custom.get(UNKNOWN_STEP)
        if isinstance(unknown, dict):
            return cast(StepParam, copy.deepcopy(unknown))
        metadata = part.metadata or {}
        server_call = custom.get(SERVER_FUNCTION_CALL)
        if isinstance(server_call, dict):
            return with_signature(to_server_function_call_step(server_call), metadata)
        server_result = custom.get(SERVER_FUNCTION_RESULT)
        if isinstance(server_result, dict):
            return to_server_function_result_step(server_result)
        step = to_server_tool_step(custom, metadata)
        return with_signature(step, metadata) if step is not None else None
    return None


def to_content_step(role: InteractionRole, content: list[ContentParam]) -> StepParam:
    """Wrap a turn's inline content in the step for whoever produced it."""
    if role == 'model':
        model_step: ModelOutputStepParam = {'type': 'model_output', 'content': content}
        return model_step
    user_step: UserInputStepParam = {'type': 'user_input', 'content': content}
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


def to_interaction_steps(messages: list[Message]) -> list[StepParam]:
    """Convert Genkit messages to Interactions API steps.

    Text and media collapse into a content step, but tool calls, thoughts,
    and Google-side tool activity are steps of their own. Flush any buffered
    inline content before those so mixed parts stay in message order — a
    thought sitting between two sentences must not jump the first sentence
    to the end of the turn.
    """
    steps: list[StepParam] = []
    for message in ensure_code_execution_ids(messages):
        role = to_interaction_role(message.role)
        inline: list[ContentParam] = []

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


def server_tool_part(key: str, payload: dict[str, Any], *, signature: str | None, call_id: str | None = None) -> Part:
    """Wrap Google-side tool activity in the custom part Genkit round-trips."""
    part = Part.from_custom({key: payload}, metadata={CALL_ID: call_id} if call_id else None)
    return with_metadata(part, THOUGHT_SIGNATURE, signature)


def from_google_search_call(step: GoogleSearchCallStep) -> Part:
    """Convert a google_search_call step to a Genkit custom part."""
    return server_tool_part(
        GOOGLE_SEARCH_CALL,
        {'id': step.id, 'arguments': plain(step.arguments)},
        signature=step.signature,
    )


def from_google_search_result(step: GoogleSearchResultStep) -> Part:
    """Convert a google_search_result step to a Genkit custom part."""
    return server_tool_part(
        GOOGLE_SEARCH_RESULT,
        {CALL_ID: step.call_id, 'result': plain(step.result or [])},
        signature=step.signature,
    )


def from_code_execution_call(step: CodeExecutionCallStep) -> Part:
    """Convert a code_execution_call step to a Genkit custom part."""
    arguments = step.arguments
    return server_tool_part(
        EXECUTABLE_CODE,
        {
            'code': arguments.code if arguments else None,
            'language': (arguments.language if arguments else None) or CODE_LANGUAGE,
        },
        signature=step.signature,
        call_id=step.id,
    )


def from_code_execution_result(step: CodeExecutionResultStep) -> Part:
    """Convert a code_execution_result step to a Genkit custom part."""
    result = step.result
    return server_tool_part(
        CODE_EXECUTION_RESULT,
        {
            'output': result if isinstance(result, str) else json.dumps(result),
            'outcome': 'OUTCOME_OK',
        },
        signature=step.signature,
        call_id=step.call_id,
    )


def from_function_call_step(step: FunctionCallStep) -> Part:
    """Convert a pending client function_call step to a Genkit tool request part."""
    part = Part(
        tool_request=ToolRequest(
            name=step.name or '',
            input=plain(step.arguments) if step.arguments is not None else {},
            ref=step.id,
        )
    )
    return with_metadata(part, THOUGHT_SIGNATURE, getattr(step, 'signature', None))


def from_server_function_call_step(step: FunctionCallStep) -> Part:
    """Stash an already-resolved or sandbox function_call as custom so Genkit's tool loop skips it."""
    payload: dict[str, Any] = {
        'id': step.id,
        'name': step.name or '',
        'arguments': plain(step.arguments) if step.arguments is not None else {},
    }
    return with_metadata(
        Part.from_custom({SERVER_FUNCTION_CALL: payload}),
        THOUGHT_SIGNATURE,
        getattr(step, 'signature', None),
    )


def from_function_result_step(step: FunctionResultStep) -> Part:
    """Stash a function_result as custom — never a ToolResponsePart.

    The API echoes the client's own tool result back on the model tape.
    Putting that on a ToolResponsePart would send it again next turn.
    """
    return Part.from_custom({
        SERVER_FUNCTION_RESULT: {
            'name': step.name or '',
            'result': plain(step.result),
            'call_id': step.call_id,
            'is_error': step.is_error,
        }
    })


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
    if content.mime_type or not isinstance(content, AudioContent) or not content.data:
        return content.mime_type
    return f'audio/l16; rate={content.sample_rate or 24000}; channels={content.channels or 1}'


def from_media_content(content: ImageContent | AudioContent | DocumentContent | VideoContent) -> Part:
    """Convert wire media content to a Genkit media part."""
    mime_type = interaction_media_mime_type(content)
    url = content.uri
    if content.data and mime_type:
        url = f'data:{mime_type};base64,{content.data}'
    return Part(media=Media(url=url or '', content_type=mime_type))


def from_text_content(content: TextContent) -> Part:
    """Convert wire text content to a Genkit text part."""
    # Empty annotations still show up in metadata so round-trips stay stable.
    return Part.from_text(content.text or '', metadata={'annotations': plain(content.annotations)})


def from_visual_content(content: ImageContent | VideoContent) -> Part:
    """Convert wire image or video content, keeping the resolution it came back at."""
    return with_metadata(from_media_content(content), 'resolution', content.resolution)


def from_thought_step(step: ThoughtStep) -> Part:
    """Convert a thought step to a Genkit reasoning part."""
    summary = step.summary or []
    reasoning = '\n'.join(item.text or '' if isinstance(item, TextContent) else '[Image]' for item in summary)
    return Part(
        reasoning=reasoning,
        metadata={THOUGHT_SIGNATURE: step.signature},
        custom={THOUGHT_CUSTOM: step.model_dump(mode='python')},
    )


def from_interaction_content(content: Content) -> Part:
    """Convert an Interactions content block back to a Genkit part."""
    if isinstance(content, TextContent):
        return from_text_content(content)
    if isinstance(content, (ImageContent, VideoContent)):
        return from_visual_content(content)
    if isinstance(content, (AudioContent, DocumentContent)):
        return from_media_content(content)
    if isinstance(content, UnknownContent):
        return Part.from_custom({'unknownContent': content.model_dump(mode='python')})
    return Part.from_custom({'unknownContent': content})


def get_pending_function_call_ids(steps: list[Step], status: str | None) -> set[str]:
    """Return IDs of function_call steps awaiting client execution on a paused turn.

    Sandbox filesystem tools (such as ``write_to_file``) also appear as
    ``function_call`` steps on Antigravity, paired with a matching
    ``function_result`` in the same interaction. Only unmatched calls on a
    ``requires_action`` turn become ``ToolRequest`` parts for Genkit's loop.
    """
    if status != 'requires_action':
        return set()
    resolved = {step.call_id for step in steps if isinstance(step, FunctionResultStep) and step.call_id}
    return {step.id for step in steps if isinstance(step, FunctionCallStep) and step.id and step.id not in resolved}


def from_interaction_step(step: Step, *, is_pending: bool = False) -> list[Part]:
    """Convert an Interactions step to Genkit parts."""
    if isinstance(step, ModelOutputStep):
        return [from_interaction_content(content) for content in (step.content or [])]
    if isinstance(step, UserInputStep):
        # The API echoes our prompt back; including it would duplicate the input.
        return []
    if isinstance(step, GoogleSearchCallStep):
        return [from_google_search_call(step)]
    if isinstance(step, GoogleSearchResultStep):
        return [from_google_search_result(step)]
    if isinstance(step, CodeExecutionCallStep):
        return [from_code_execution_call(step)]
    if isinstance(step, CodeExecutionResultStep):
        return [from_code_execution_result(step)]
    if isinstance(step, ThoughtStep):
        return [from_thought_step(step)]
    if isinstance(step, FunctionCallStep):
        return [from_function_call_step(step)] if is_pending else [from_server_function_call_step(step)]
    if isinstance(step, FunctionResultStep):
        return [from_function_result_step(step)]
    if isinstance(step, BaseModel):
        return [Part.from_custom({UNKNOWN_STEP: step.model_dump(mode='python')})]
    return [Part.from_custom({UNKNOWN_STEP: step})]


def interaction_message_metadata(interaction: Interaction) -> dict[str, Any] | None:
    """Build message.metadata fields that identify the source Interaction."""
    metadata: dict[str, Any] = {}
    if interaction.id:
        metadata['interactionId'] = interaction.id
    if interaction.environment_id:
        metadata['environmentId'] = interaction.environment_id
    if interaction.status:
        # Preserve the wire status when finish_reason is a coarser Genkit enum.
        metadata['interactionStatus'] = interaction.status
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
        input_tokens=usage.total_input_tokens,
        output_tokens=usage.total_output_tokens,
        total_tokens=usage.total_tokens,
        cached_content_tokens=usage.total_cached_tokens,
        thoughts_tokens=usage.total_thought_tokens,
    )
    # Genkit has no standard fields for these.
    custom: dict[str, float] = {}

    def add_modality_tokens(direction: str, breakdown: list[ModalityTokens] | None) -> None:
        for item in breakdown or []:
            if item.modality and isinstance(item.tokens, int):
                key = f'{direction}{item.modality[:1].upper()}{item.modality[1:]}Tokens'
                custom[key] = custom.get(key, 0) + item.tokens

    add_modality_tokens('input', usage.input_tokens_by_modality)
    add_modality_tokens('output', usage.output_tokens_by_modality)
    # Tokens loaded by tools, e.g. video frames read during agentic video
    # understanding.
    if usage.total_tool_use_tokens:
        custom['toolUseTokens'] = usage.total_tool_use_tokens
    # How many times each grounding tool was used, e.g. `googleSearchCount`.
    for tool_count in usage.grounding_tool_count or []:
        if tool_count.type and isinstance(tool_count.count, int):
            key = f'{snake_to_camel(tool_count.type)}Count'
            custom[key] = custom.get(key, 0) + tool_count.count
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
        for part in from_interaction_step(
            step,
            is_pending=isinstance(step, FunctionCallStep) and bool(step.id) and step.id in pending_ids,
        )
        if part.model_dump(exclude_none=True)
    ]


def model_response(
    interaction: Interaction,
    *,
    content: list[Part],
    finish_reason: FinishReason,
    finish_message: str | None = None,
) -> ModelResponse:
    """Build a ModelResponse that carries the raw interaction alongside its parts."""
    dumped = interaction.model_dump(mode='python')
    response = ModelResponse.model_construct(
        finish_reason=finish_reason,
        finish_message=finish_message,
        message=Message(role='model', content=content, metadata=interaction_message_metadata(interaction)),
        custom=dumped,
        raw=dumped,
    )
    if interaction.usage is not None:
        response.usage = usage_from_interaction(interaction.usage)
    return response


def cancelled_response(interaction: Interaction) -> ModelResponse:
    """Build the ModelResponse for a cancelled Interaction."""
    return model_response(
        interaction,
        content=[Part.from_text('Operation cancelled.')],
        finish_reason=FinishReason.ABORTED,
        finish_message='Operation cancelled',
    )


def steps_response(
    interaction: Interaction,
    *,
    finish_reason: FinishReason,
    finish_message: str | None = None,
) -> ModelResponse | None:
    """Build a ModelResponse from interaction steps, or None when there are none."""
    steps = list(interaction.steps or [])
    if not steps:
        return None
    return model_response(
        interaction,
        content=parts_from_steps(steps, status=interaction.status),
        finish_reason=finish_reason,
        finish_message=finish_message,
    )


def completed_response(interaction: Interaction) -> ModelResponse | None:
    """Build the ModelResponse for a completed Interaction, if it has steps."""
    return steps_response(interaction, finish_reason=FinishReason.STOP)


def from_interaction_sync(interaction: Interaction) -> ModelResponse:
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
    status = interaction.status
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
    if partial := PARTIAL_TERMINAL_STATUSES.get(cast(str, status)):
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


def from_interaction(interaction: Interaction) -> Operation:
    """Convert an interaction poll result to a Genkit operation."""
    op = Operation.model_construct(id=interaction.id or '')

    status = interaction.status
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
    elif partial := PARTIAL_TERMINAL_STATUSES.get(cast(str, status)):
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
