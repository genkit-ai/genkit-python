#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Nested models and other non-JSON leaves still serialize to valid JSON attributes."""

import json
from datetime import UTC, datetime

from pydantic import BaseModel, ConfigDict, Field

from genkit._core._telemetry._instrumentation import to_json_attr


class Reservation(BaseModel):
    """Booking payload whose alias and optional field must survive encoding."""

    model_config = ConfigDict(populate_by_name=True)

    party_size: int = Field(alias='partySize')
    notes: str | None = None


def test_nested_model_in_dict_is_json() -> None:
    """A dict holding models serializes through the alias and drops None."""
    encoded = to_json_attr({'bookings': [Reservation(partySize=2)]})
    assert json.loads(encoded) == {'bookings': [{'partySize': 2}]}


def test_list_of_models_is_json() -> None:
    """A list of models is JSON, not a str() of the list."""
    encoded = to_json_attr([Reservation(partySize=3, notes='window')])
    assert json.loads(encoded) == [{'partySize': 3, 'notes': 'window'}]


def test_top_level_model_unchanged() -> None:
    """A bare model keeps the existing compact model_dump_json output."""
    assert to_json_attr(Reservation(partySize=4)) == '{"partySize":4}'


def test_datetime_leaf_is_iso() -> None:
    """Leaves the json module rejects, such as datetimes, encode as JSON scalars."""
    encoded = to_json_attr({'at': datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)})
    assert json.loads(encoded) == {'at': '2026-01-02T03:04:05Z'}


def test_unencodable_leaf_still_returns_string() -> None:
    """Anything neither json nor pydantic can encode stays a readable string."""
    encoded = to_json_attr({'weird': object()})
    assert 'weird' in encoded
