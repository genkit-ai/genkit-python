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

"""Tool call content attributes encode Pydantic values like core's to_json_attr."""

import json

from genkit_otel._genai_instrumentation import _encode_json_value
from pydantic import BaseModel, ConfigDict, Field


class Reservation(BaseModel):
    model_config = ConfigDict(populate_by_name=True)

    party_size: int = Field(alias='partySize')
    notes: str | None = None


def test_pydantic_value_uses_wire_aliases_and_drops_none() -> None:
    """A model with an alias and a None field encodes as {"partySize": 4}, the shape core's genkit:input uses."""
    booking = Reservation.model_validate({'partySize': 4})

    assert json.loads(_encode_json_value(booking)) == {'partySize': 4}


def test_nested_pydantic_values_are_encoded() -> None:
    """Models inside lists and dicts encode the same way."""
    value = {
        'bookings': [
            Reservation.model_validate({'partySize': 2}),
            Reservation.model_validate({'partySize': 6, 'notes': 'window'}),
        ]
    }

    assert json.loads(_encode_json_value(value)) == {
        'bookings': [{'partySize': 2}, {'partySize': 6, 'notes': 'window'}],
    }


def test_plain_string_passes_through_unquoted() -> None:
    """A str is emitted as-is, not JSON-quoted."""
    assert _encode_json_value('table for two') == 'table for two'
