# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# SPDX-License-Identifier: Apache-2.0

"""Translation of Genkit prompts and their rendered requests into MCP shapes."""

import json
import math
import re
from typing import Any

from mcp.types import (
    AudioContent,
    BlobResourceContents,
    ContentBlock,
    EmbeddedResource,
    ImageContent,
    Prompt,
    PromptArgument,
    PromptMessage,
    ResourceLink,
    TextContent,
)
from pydantic import AnyUrl

from genkit import Message, Part, Role
from genkit.model import ModelRequest
from genkit.plugin_api import Action

_DATA_URL = re.compile(r'^data:(?P<mime>[^;,]+)(?:;[^;,]*)*;base64,(?P<data>.*)$', re.DOTALL)

_INTEGER = re.compile(r'-?(?:0|[1-9][0-9]*)')

_MAX_REF_DEPTH = 32
"""How many ``$ref`` hops are followed before a schema is taken as it stands."""

SCALAR_ARGUMENT = 'input'
"""The one argument a prompt whose input is not an object takes."""

JSON_ARGUMENT = 'inputJson'
"""The argument that carries input fields a prompt's schema does not name, as a JSON object."""

_PREFIXES: dict[str, str] = {Role.SYSTEM: 'System instructions:\n', Role.TOOL: 'Tool output:\n'}
"""How a turn MCP has no role for is marked when it is sent as a user turn."""


class PromptArgumentError(ValueError):
    """An MCP argument that cannot be read as what the prompt's schema expects."""


def prompt_to_mcp(action: Action) -> Prompt:
    """Describe one Genkit prompt action as an MCP prompt.

    Each named input field becomes one argument. A prompt whose input is not an
    object takes it as ``input``, and one that accepts fields its schema does not
    name also takes them together as ``inputJson``.

    Args:
        action: A registered ``prompt`` action.

    Returns:
        The MCP prompt description.
    """
    schema = dict(action.input_schema or {})
    return Prompt(name=action.name, description=action.description, arguments=_arguments(schema))


def prompt_input(schema: dict[str, Any], arguments: dict[str, str]) -> object:
    """Read MCP's string arguments as the input the prompt's schema describes.

    Args:
        schema: The prompt's input JSON schema.
        arguments: What the client sent, every value a string.

    Returns:
        The prompt input, or ``None`` when no arguments were sent.

    Raises:
        PromptArgumentError: An argument does not parse as its field's type.
    """
    if not arguments:
        return None
    root = schema
    schema = _resolve(schema, root)
    if not _is_object(schema):
        if set(arguments) != {SCALAR_ARGUMENT}:
            raise PromptArgumentError(f'this prompt takes a single {SCALAR_ARGUMENT!r} argument')
        return _convert(schema, arguments[SCALAR_ARGUMENT], SCALAR_ARGUMENT)

    properties = _as_object(schema.get('properties'))
    takes_json = _takes_json_input(schema, properties)
    value: dict[str, Any] = {}
    if takes_json and JSON_ARGUMENT in arguments:
        fields = _convert({**schema, 'type': 'object'}, arguments[JSON_ARGUMENT], JSON_ARGUMENT)
        value.update(_as_object(fields))
    for name, text in arguments.items():
        if takes_json and name == JSON_ARGUMENT:
            continue
        value[name] = _convert(_field_schema(schema, properties, name, root), text, name)
    return value


def request_to_messages(request: ModelRequest, prompt_name: str) -> list[PromptMessage]:
    """Convert a rendered prompt into MCP prompt messages.

    Context documents come first, as user turns. MCP has only user and
    assistant turns, so system and tool turns are sent as user turns with their
    text marked.

    Args:
        request: What running the prompt action rendered.
        prompt_name: Name of the prompt, used to mint a URI for binary content
            that has no URI of its own.

    Returns:
        One MCP message per content part.
    """
    messages: list[PromptMessage] = []
    index = 0
    for number, doc in enumerate(request.docs or [], start=1):
        header = f'Context document {number}:'
        if doc.metadata:
            header += f'\nMetadata: {json.dumps(doc.metadata)}'
        messages.append(PromptMessage(role='user', content=TextContent(type='text', text=header)))
        for part in doc.content:
            messages.append(PromptMessage(role='user', content=_part_to_block(part, prompt_name, index)))
            index += 1
    for message in request.messages:
        for part in message.content:
            messages.append(_part_to_message(message, part, prompt_name, index))
            index += 1
    return messages


def _arguments(schema: dict[str, Any]) -> list[PromptArgument]:
    """List the arguments a prompt with this input schema takes, sorted by name."""
    schema = _resolve(schema, schema)
    if not _is_object(schema):
        description = schema.get('description')
        return [
            PromptArgument(
                name=SCALAR_ARGUMENT,
                description=description if isinstance(description, str) else None,
                required=True,
            )
        ]
    properties = _as_object(schema.get('properties'))
    raw_required = schema.get('required')
    required = {name for name in raw_required if isinstance(name, str)} if isinstance(raw_required, list) else set()
    arguments: list[PromptArgument] = []
    for name, field in properties.items():
        description = _as_object(field).get('description')
        arguments.append(
            PromptArgument(
                name=name,
                description=description if isinstance(description, str) else None,
                required=name in required,
            )
        )
    if _takes_json_input(schema, properties):
        arguments.append(
            PromptArgument(name=JSON_ARGUMENT, description='JSON object of prompt input fields', required=False)
        )
    return sorted(arguments, key=lambda argument: argument.name)


def _is_object(schema: dict[str, Any]) -> bool:
    """Whether the input is an object, which a schema with no type or union is taken to be."""
    kind = schema.get('type')
    if kind is None:
        return not schema.get('anyOf') and not schema.get('oneOf')
    return kind == 'object'


def _resolve(schema: dict[str, Any], root: dict[str, Any]) -> dict[str, Any]:
    """Reduce a schema to the one type an argument is read as, where it has one.

    A ``$ref`` into the root's ``$defs`` is followed, and a ``null`` alternative
    is dropped, since an optional field's argument is only sent with a value. A
    schema left with several types is returned as it is, and its argument is
    read as text.
    """
    for _ in range(_MAX_REF_DEPTH):
        ref = schema.get('$ref')
        if not isinstance(ref, str) or not ref.startswith('#/$defs/'):
            break
        target = _as_object(_as_object(root.get('$defs')).get(ref.removeprefix('#/$defs/')))
        schema = {**{key: value for key, value in schema.items() if key != '$ref'}, **target}
    kind = schema.get('type')
    if isinstance(kind, list):
        types = [item for item in kind if item != 'null']
        return {**schema, 'type': types[0]} if len(types) == 1 else schema
    for union in ('anyOf', 'oneOf'):
        branches = schema.get(union)
        if not isinstance(branches, list):
            continue
        typed = [_resolve(_as_object(branch), root) for branch in branches]
        present = [branch for branch in typed if branch.get('type') != 'null']
        if len(present) == 1:
            rest = {key: value for key, value in schema.items() if key != union}
            return {**rest, **present[0]}
    return schema


def _takes_json_input(schema: dict[str, Any], properties: dict[str, Any]) -> bool:
    """Whether the prompt accepts fields it does not name, so ``inputJson`` is offered.

    A prompt with no named fields is taken to accept any, unless its schema
    says otherwise.
    """
    additional = schema.get('additionalProperties')
    if additional is False or JSON_ARGUMENT in properties:
        return False
    return not properties or additional is not None


def _field_schema(
    schema: dict[str, Any], properties: dict[str, Any], name: str, root: dict[str, Any]
) -> dict[str, Any]:
    """The schema of one input field, falling back to what unnamed fields must match."""
    field = properties[name] if name in properties else schema.get('additionalProperties')
    return _resolve(_as_object(field), root)


def _convert(schema: dict[str, Any], text: str, name: str) -> object:
    """Parse one string argument as the type its schema names; text stays text."""
    kind = schema.get('type')
    if kind == 'integer':
        return _integer(text, name)
    if kind == 'number':
        return _number(text, name)
    if kind == 'boolean':
        if text.lower() in ('true', 'false'):
            return text.lower() == 'true'
        raise PromptArgumentError(f'argument {name!r} must be true or false')
    if kind in ('object', 'array'):
        try:
            value = json.loads(text)
        except json.JSONDecodeError as error:
            raise PromptArgumentError(f'argument {name!r} must be JSON: {error.msg}') from None
        expected = dict if kind == 'object' else list
        if not isinstance(value, expected):
            raise PromptArgumentError(f'argument {name!r} must be a JSON {kind}')
        return value
    return text


def _integer(text: str, name: str) -> int:
    """Parse a whole number, allowing the JSON spelling of one such as ``3.0``."""
    if _INTEGER.fullmatch(text):
        return int(text)
    value = _number(text, name)
    if value != int(value):
        raise PromptArgumentError(f'argument {name!r} must be a whole number')
    return int(value)


def _number(text: str, name: str) -> float:
    """Parse a finite number."""
    try:
        value = float(text)
    except ValueError:
        raise PromptArgumentError(f'argument {name!r} must be a number') from None
    if not math.isfinite(value):
        raise PromptArgumentError(f'argument {name!r} must be a finite number')
    return value


def _part_to_message(message: Message, part: Part, prompt_name: str, index: int) -> PromptMessage:
    """Convert one part of a rendered turn into an MCP message."""
    content = _part_to_block(part, prompt_name, index)
    prefix = _PREFIXES.get(message.role)
    if prefix is not None and isinstance(content, TextContent):
        content = TextContent(type='text', text=prefix + content.text)
    return PromptMessage(role='assistant' if message.role == Role.MODEL else 'user', content=content)


def _part_to_block(part: Part, prompt_name: str, index: int) -> ContentBlock:
    """Convert one Genkit content part into an MCP content block.

    A part MCP has no block for is sent as its JSON. A part's metadata is not
    sent, since MCP content has nowhere to carry it.
    """
    if part.text is not None:
        return TextContent(type='text', text=part.text)
    if part.media is not None:
        return _media_block(part.media.url, part.media.content_type, prompt_name, index)
    if part.resource is not None:
        return _link(str(part.resource.uri), None)
    if part.data is not None or 'data' in part.model_fields_set:
        return TextContent(type='text', text=json.dumps(part.data))
    return TextContent(type='text', text=json.dumps(part.model_dump(by_alias=True, exclude_none=True)))


def _media_block(url: str, content_type: str | None, prompt_name: str, index: int) -> ContentBlock:
    """Embed base64 inline media, and link to media that lives elsewhere."""
    match = _DATA_URL.match(url)
    if match is None:
        return _link(url, content_type)
    mime_type = content_type or match.group('mime')
    data = match.group('data')
    if mime_type.startswith('image/'):
        return ImageContent(type='image', mimeType=mime_type, data=data)
    if mime_type.startswith('audio/'):
        return AudioContent(type='audio', mimeType=mime_type, data=data)
    return EmbeddedResource(
        type='resource',
        resource=BlobResourceContents(
            uri=AnyUrl(f'genkit://prompt/{prompt_name}/content/{index}'),
            mimeType=mime_type,
            blob=data,
        ),
    )


def _link(url: str, mime_type: str | None) -> ContentBlock:
    """Link to content by URL, or send the URL as text when it is not absolute."""
    try:
        uri = AnyUrl(url)
    except ValueError:
        return TextContent(type='text', text=url)
    return ResourceLink(type='resource_link', uri=uri, name=url, mimeType=mime_type)


def _as_object(value: object) -> dict[str, Any]:
    """Read a schema value that should be a nested object, or nothing."""
    if not isinstance(value, dict):
        return {}
    return {str(key): item for key, item in value.items()}
