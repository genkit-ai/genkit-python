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

"""Unwrapping replies to an output schema that was sent wrapped in an object.

OpenAI only accepts a ``json_schema`` response format whose root is
``type: "object"``. A schema with any other root (a list, enum or scalar) is
sent as the ``value`` property of an object, and the reply, which arrives as
``{"value": ...}``, is turned back into the caller's value here.
"""

import json
import re
from typing import Any, cast

# The property that holds the caller's value inside the wrapper object.
OUTPUT_VALUE_FIELD = 'value'

# The wrapper's opening, as the tokens it is made of. Whitespace may sit
# before and between them.
_PREFIX_TOKENS = ('{', f'"{OUTPUT_VALUE_FIELD}"', ':')
_TRAILING = ' \t\r\n}'

# Keywords a bare ``$ref`` may sit beside: at the root, and in a definition.
_ROOT_REF_KEYS = frozenset({'$ref', '$schema', '$defs', 'definitions'})
_REF_KEYS = frozenset({'$ref'})
_DEF_REF = re.compile(r'#/(\$defs|definitions)/([^/~]+)')


def has_object_root(schema: dict[str, Any]) -> bool:
    """Whether the schema's root is explicitly an object.

    Either directly, or through a bare root ``$ref`` to a top-level definition
    (as Pydantic emits for a recursive model), which OpenAI accepts unwrapped.
    Anything else, including ``properties`` without a ``type`` (which also
    allows non-object values), is wrapped.

    OpenAI rejects a ``$ref`` with sibling keywords (``description``,
    ``title``, ...) and does not resolve escaped names (``~0``, ``~1``) at the
    root, while non-strict requests accept both once wrapped, so those roots
    are wrapped.
    """
    seen: set[str] = set()
    node: object = schema
    allowed = _ROOT_REF_KEYS
    while isinstance(node, dict):
        node = cast(dict[str, Any], node)
        if 'type' in node:
            return node['type'] == 'object'
        ref = node.get('$ref')
        if not isinstance(ref, str) or ref in seen or not set(node) <= allowed:
            return False
        seen.add(ref)
        match = _DEF_REF.fullmatch(ref)
        if match is None:
            return False
        defs = schema.get(match[1])
        node = cast(dict[str, Any], defs).get(match[2]) if isinstance(defs, dict) else None
        allowed = _REF_KEYS
    return False


def wrap_schema_root(schema: dict[str, Any]) -> dict[str, Any]:
    """``schema`` as the one required ``value`` property of an object schema.

    ``$schema``, ``$defs`` and ``definitions`` stay at the root, so references
    to the definitions still resolve.

    Any other local reference (``#``, ``#/items/...``) pointed into the
    caller's schema, which is no longer the root. Strict mode accepts
    references only to top-level definitions, so each referenced subschema
    moves into ``$defs``, and both its old place and every reference to it
    point there instead.
    """
    hoister = _RefHoister(schema)
    wrapped: dict[str, Any] = {
        'type': 'object',
        'required': [OUTPUT_VALUE_FIELD],
        'additionalProperties': False,
    }
    inner: dict[str, Any] = {}
    for key, value in schema.items():
        if key == '$schema':
            wrapped[key] = value
        elif key == 'definitions':
            wrapped[key] = hoister.walk(value, '#/definitions', names=True)
        elif key == '$defs':
            if isinstance(value, dict):
                for name, definition in cast(dict[str, Any], value).items():
                    hoister.defs[name] = hoister.walk(definition, f'#/$defs/{_escape(name)}')
            else:
                wrapped[key] = value
        else:
            inner[key] = value
    wrapped['properties'] = {OUTPUT_VALUE_FIELD: hoister.walk(inner, '#')}
    if hoister.defs:
        wrapped['$defs'] = hoister.defs
    return wrapped


# Keywords that hold data rather than subschemas: a ``$ref`` inside them is
# not a reference.
_DATA_KEYWORDS = frozenset({'const', 'enum', 'default', 'examples'})
# Keywords whose values map names to subschemas. Keys there are names, so a
# property called ``default`` or ``$ref`` is a schema like any other.
_NAME_MAPS = frozenset({'properties', 'patternProperties', '$defs', 'definitions', 'dependentSchemas'})
_DEFS_PREFIXES = ('#/$defs/', '#/definitions/')


class _RefHoister:
    """Moves the subschemas that local references point at into ``$defs`` as it copies a schema."""

    def __init__(self, schema: dict[str, Any]) -> None:
        self.defs: dict[str, Any] = {}
        defs = schema.get('$defs')
        taken = set(cast(dict[str, Any], defs)) if isinstance(defs, dict) else set()
        refs: set[str] = set()
        _collect_refs(schema, refs, names=False)
        # Each referenced JSON pointer, and the name it moves to in defs.
        self.targets: dict[str, str] = {}
        for ref in sorted(r for r in refs if _hoistable(r) and _resolves(schema, r)):
            self.targets[ref] = _unique_def_name(ref, taken)

    def walk(self, node: object, pointer: str, *, names: bool = False) -> object:
        """Copy ``node``, found at ``pointer`` in the caller's schema.

        A node that a reference points at goes into defs, and a reference to it
        is left in its place. ``names`` says ``node`` maps names to subschemas.
        """
        if (name := self.targets.get(pointer)) is not None:
            self.defs[name] = self._copy(node, pointer, names=names)
            return {'$ref': f'#/$defs/{name}'}
        return self._copy(node, pointer, names=names)

    def _copy(self, node: object, pointer: str, *, names: bool) -> object:
        if isinstance(node, list):
            return [self.walk(v, f'{pointer}/{i}') for i, v in enumerate(cast(list[Any], node))]
        if not isinstance(node, dict):
            return node
        out: dict[str, Any] = {}
        for key, value in cast(dict[str, Any], node).items():
            at = f'{pointer}/{_escape(key)}'
            if names:
                out[key] = self.walk(value, at)
            elif key == '$ref' and isinstance(value, str):
                name = self.targets.get(value)
                out[key] = f'#/$defs/{name}' if name is not None else value
            elif key in _DATA_KEYWORDS:
                out[key] = value
            else:
                out[key] = self.walk(value, at, names=key in _NAME_MAPS)
        return out


def _collect_refs(node: object, found: set[str], *, names: bool) -> None:
    """Add every ``$ref`` in ``node`` to ``found``."""
    if isinstance(node, list):
        for value in cast(list[Any], node):
            _collect_refs(value, found, names=False)
    elif isinstance(node, dict):
        for key, value in cast(dict[str, Any], node).items():
            if names:
                _collect_refs(value, found, names=False)
            elif key == '$ref' and isinstance(value, str):
                found.add(value)
            elif key not in _DATA_KEYWORDS:
                _collect_refs(value, found, names=key in _NAME_MAPS)


def _defs_path(ref: str) -> str | None:
    """What follows ``#/$defs/`` or ``#/definitions/`` in ``ref``, or None."""
    for prefix in _DEFS_PREFIXES:
        if ref.startswith(prefix):
            return ref[len(prefix) :]
    return None


def _hoistable(ref: str) -> bool:
    """Whether ``ref`` is a local reference strict mode can't follow once the schema is wrapped.

    A reference to a whole top-level definition is fine; one into the middle
    of a definition or anywhere else in the schema is not.
    """
    if ref != '#' and not ref.startswith('#/'):
        return False
    if (rest := _defs_path(ref)) is not None:
        return '/' in rest
    return ref not in ('#/$defs', '#/definitions')


def _resolves(schema: dict[str, Any], ref: str) -> bool:
    """Whether the JSON pointer ``ref`` names a schema inside ``schema``."""
    node: object = schema
    for segment in ref.removeprefix('#').split('/')[1:]:
        segment = segment.replace('~1', '/').replace('~0', '~')
        if isinstance(node, dict):
            node = cast(dict[str, Any], node)
            if segment not in node:
                return False
            node = node[segment]
        elif isinstance(node, list):
            items = cast(list[Any], node)
            if not segment.isdigit() or int(segment) >= len(items):
                return False
            node = items[int(segment)]
        else:
            return False
    return isinstance(node, dict)


def _escape(key: str) -> str:
    return key.replace('~', '~0').replace('/', '~1')


def _unique_def_name(ref: str, taken: set[str]) -> str:
    """Name the definition a referenced pointer moves to, avoiding the names already taken.

    A pointer into a definition is named after its path there
    (``Person_properties_name``), and any other after its path under the
    wrapper (``value_items``).
    """
    path = _defs_path(ref)
    if path is None:
        path = OUTPUT_VALUE_FIELD + ref.removeprefix('#')
    base = re.sub(r'[^\w-]', '_', path)
    name, i = base, 2
    while name in taken:
        name, i = f'{base}_{i}', i + 1
    taken.add(name)
    return name


def unwrap_output_text(text: str) -> str:
    """JSON text of the wrapped value in ``text``, or ``text`` unchanged.

    The value's text is kept as sent, so it reads the same as when streamed.
    Text that is not an object whose only key is ``value`` is returned as is:
    a host that ignored the schema may reply with the bare value, or with an
    object that has a ``value`` property of its own.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if not isinstance(parsed, dict) or OUTPUT_VALUE_FIELD not in parsed:
        return text
    if len(cast(dict[str, Any], parsed)) > 1:
        # Keys beside the value: a host that ignored the wrapper and replied
        # with an object of its own, which is left as sent.
        return text
    # The shape is known, so stripping the opening and closing brace leaves
    # exactly the value.
    unwrapper = OutputStreamUnwrapper()
    return unwrapper.push(text) + unwrapper.flush()


def _match_prefix(text: str) -> int | None:
    """Match the start of ``text`` against the wrapper's opening.

    Returns the opening's length when it is all there, 0 when ``text`` could
    still become it, and None when it cannot.
    """
    i = 0
    for token in _PREFIX_TOKENS:
        while i < len(text) and text[i].isspace():
            i += 1
        rest = text[i : i + len(token)]
        if rest != token:
            return 0 if token.startswith(rest) and i + len(rest) == len(text) else None
        i += len(token)
    return i


class OutputStreamUnwrapper:
    """Strips the ``{"value":`` opening and the closing ``}`` from streamed text.

    Chunks then read as the caller's value (``[{...},`` for a list), which is
    what the output format's chunk parser expects.

    Text is held back for as long as it could still be the start of the
    wrapper, so the result does not depend on how the stream is split.
    Trailing whitespace and braces are then held until more text arrives,
    because the last ``}`` belongs to the wrapper only if nothing follows it;
    ``flush`` drops it. A reply that is not a wrapper passes through unchanged.
    """

    def __init__(self) -> None:
        """Start before any text has arrived."""
        self._started = False
        self._wrapped = False
        self._value_started = False
        self._passing = False
        self._head = ''
        self._held = ''

    def push(self, delta: str) -> str:
        """Take a streamed delta and return the text to emit for it, possibly empty."""
        if self._passing:
            return delta
        if self._started:
            return self._release(delta)
        self._head += delta
        match = _match_prefix(self._head)
        if match == 0:
            return ''
        self._started = True
        out, self._head = self._head, ''
        if match is not None:
            self._wrapped = True
            out = out[match:]
        return self._release(out)

    def _release(self, text: str) -> str:
        text = self._held + text
        # Whitespace after the colon belongs to the wrapper too, however much
        # of it arrives after the opening matched.
        if self._wrapped and not self._value_started:
            text = text.lstrip()
            self._value_started = bool(text)
        kept = text.rstrip(_TRAILING)
        self._held = text[len(kept) :]
        return kept

    def pass_through(self) -> str:
        """Stop unwrapping and return the text held back so far.

        For a reply that turns out not to be the output: text before a tool
        call is a preamble.
        """
        out = self._head + self._held
        self._started = self._passing = True
        self._head = self._held = ''
        return out

    def flush(self) -> str:
        """Return what is left at the end of the stream, minus the wrapper's closing brace."""
        if self._passing:
            return ''
        if not self._started:
            out, self._head = self._head, ''
            return out
        text, self._held = self._held, ''
        if not self._wrapped:
            return text
        brace = text.rfind('}')
        if brace >= 0:
            text = text[:brace]
        return text.rstrip()
