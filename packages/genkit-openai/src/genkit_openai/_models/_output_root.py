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


def unwrap_output_text(text: str) -> str:
    """JSON text of the wrapped value in ``text``, or ``text`` unchanged.

    The value's text is kept as sent, so it reads the same as when streamed.
    Text that is not exactly ``{"value": ...}`` is returned as is: a host that
    ignored the schema may reply with the bare value.
    """
    try:
        parsed = json.loads(text)
    except ValueError:
        return text
    if not isinstance(parsed, dict) or set(cast(dict[str, Any], parsed)) != {OUTPUT_VALUE_FIELD}:
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
