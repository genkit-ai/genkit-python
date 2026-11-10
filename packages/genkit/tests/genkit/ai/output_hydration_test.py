#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""output_schema hydrates every type pydantic validates, not just BaseModel.

``ai.generate(output_schema=T)`` types ``.output`` as ``T``; these tests pin
that runtime returns a ``T`` too, for responses, stream chunks, and prompts.
"""

import dataclasses
import datetime
import enum
import ipaddress
import pathlib
import sys
import typing
import uuid
from decimal import Decimal
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field, RootModel, ValidationError, field_validator
from pydantic.dataclasses import dataclass as pydantic_dataclass
from typing_extensions import TypeAliasType, TypedDict

from genkit import Genkit, GenkitError, Message, ModelResponse, ModelResponseChunk, Part
from genkit._core._error import ErrorReason
from genkit._core._partial import anchor_type, construct_partial, hydrate_partial, unwrap_annotated
from genkit._core._schema import InvalidOutputSchemaError, output_adapter, to_json_schema
from genkit._core._typing import Role
from genkit.testing import define_scripted_model


class InvoiceItem(BaseModel):
    sku: str
    quantity: int


class InvoiceList(RootModel[list[InvoiceItem]]):
    pass


class ScoreRoot(RootModel[int]):
    pass


class LabelRoot(RootModel[str]):
    pass


class EvenScoreRoot(RootModel[int]):
    @field_validator('root')
    @classmethod
    def _must_be_even(cls, v: int) -> int:
        if v % 2 != 0:
            raise ValueError('score must be even')
        return v


@dataclasses.dataclass
class Vendor:
    vendor_name: str
    risk_score: int


@pydantic_dataclass
class PydanticVendor:
    vendor_name: str
    risk_score: int


class VendorDict(TypedDict):
    vendor_name: str
    risk_score: int


class Triage(enum.Enum):
    APPROVE = 'APPROVE'
    ESCALATE = 'ESCALATE'


class TriageStr(str, enum.Enum):  # noqa: UP042  # enum.StrEnum is 3.11+
    APPROVE = 'APPROVE'
    ESCALATE = 'ESCALATE'


class Plan(str, enum.Enum):  # noqa: UP042  # enum.StrEnum is 3.11+
    PRO = 'PRO'
    PRO_PLUS = 'PRO_PLUS'


class TriageRoot(RootModel[Triage]):
    pass


class Pickup(BaseModel):
    pickup_time: str


class Delivery(BaseModel):
    address: str


@dataclasses.dataclass
class Contact:
    vendor: Vendor
    emails: list[str]


class PurchaseOrder(BaseModel):
    lines: list[VendorDict]
    contact: Contact | None = None


VendorListAlias = TypeAliasType('VendorListAlias', list[Vendor])


def _ai_with_reply(text: str, chunks: list[str] | None = None) -> Genkit:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text(text)]))]
    if chunks is not None:
        pm.chunks = [[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text(c)]) for c in chunks]]
    return ai


VENDOR_JSON = '{"vendor_name": "Acme", "risk_score": 12}'
BETA_VENDOR_JSON = '{"vendor_name": "Beta", "risk_score": 5}'
UID = '550e8400-e29b-41d4-a716-446655440000'

CASES: list[tuple[str, Any, str, Any]] = [
    ('list_of_models', list[InvoiceItem], '[{"sku": "CH-1", "quantity": 2}]', [InvoiceItem(sku='CH-1', quantity=2)]),
    ('root_model', InvoiceList, '[{"sku": "CH-1", "quantity": 2}]', InvoiceList([InvoiceItem(sku='CH-1', quantity=2)])),
    ('scalar_root_int', ScoreRoot, '42', ScoreRoot(42)),
    ('scalar_root_str', LabelRoot, '"approved"', LabelRoot('approved')),
    ('scalar_root_enum', TriageRoot, '"ESCALATE"', TriageRoot(Triage.ESCALATE)),
    ('inline_root_int', RootModel[int], '7', RootModel[int](7)),
    ('inline_root_str', RootModel[str], '"ok"', RootModel[str]('ok')),
    ('set_of_str', set[str], '["a", "b"]', {'a', 'b'}),
    ('frozenset_of_int', frozenset[int], '[1, 2]', frozenset({1, 2})),
    ('pep695_alias', VendorListAlias, f'[{VENDOR_JSON}]', [Vendor(vendor_name='Acme', risk_score=12)]),
    ('dict', dict[str, int], '{"a": 1}', {'a': 1}),
    ('literal', Literal['APPROVE', 'ESCALATE'], '"ESCALATE"', 'ESCALATE'),
    ('annotated', Annotated[int, Field(gt=0)], '7', 7),
    ('dataclass', Vendor, VENDOR_JSON, Vendor(vendor_name='Acme', risk_score=12)),
    ('pydantic_dataclass', PydanticVendor, VENDOR_JSON, PydanticVendor(vendor_name='Acme', risk_score=12)),
    ('typed_dict', VendorDict, VENDOR_JSON, {'vendor_name': 'Acme', 'risk_score': 12}),
    ('enum', Triage, '"ESCALATE"', Triage.ESCALATE),
    ('str_enum', TriageStr, '"ESCALATE"', TriageStr.ESCALATE),
    ('decimal', Decimal, '14250.75', Decimal('14250.75')),
    ('uuid', uuid.UUID, f'"{UID}"', uuid.UUID(UID)),
    (
        'datetime',
        datetime.datetime,
        '"2026-10-15T14:30:00Z"',
        datetime.datetime(2026, 10, 15, 14, 30, tzinfo=datetime.timezone.utc),
    ),
    ('optional_model', InvoiceItem | None, '{"sku": "CH-1", "quantity": 2}', InvoiceItem(sku='CH-1', quantity=2)),
    ('union_of_models', Pickup | Delivery, '{"address": "1 Main St"}', Delivery(address='1 Main St')),
    (
        'nested_dataclass',
        Contact,
        f'{{"vendor": {VENDOR_JSON}, "emails": ["ap@acme.test"]}}',
        Contact(vendor=Vendor(vendor_name='Acme', risk_score=12), emails=['ap@acme.test']),
    ),
]


@pytest.mark.asyncio
@pytest.mark.parametrize(('schema', 'reply', 'expected'), [c[1:] for c in CASES], ids=[c[0] for c in CASES])
async def test_generate_hydrates_output_schema(schema: Any, reply: str, expected: Any) -> None:  # noqa: ANN401
    response = await _ai_with_reply(reply).generate(prompt='hi', output_schema=schema)

    assert response.error is None
    assert response.output == expected
    assert type(response.output) is type(expected)


@pytest.mark.asyncio
async def test_dict_of_decimal_hydrates_json_number() -> None:
    response = await _ai_with_reply('{"amount": 0.1}').generate(prompt='hi', output_schema=dict[str, Decimal])

    assert response.output == {'amount': Decimal('0.1')}


@pytest.mark.asyncio
async def test_wrong_shape_is_invalid_output_for_non_model_schema() -> None:
    response = await _ai_with_reply('[{"sku": "CH-1"}]').generate(prompt='hi', output_schema=list[InvoiceItem])

    assert response.output is None
    assert response.error is not None
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}


@pytest.mark.asyncio
async def test_scalar_root_model_streams_and_rejects_invalid_scalars() -> None:
    ai = _ai_with_reply('"hello"', chunks=['"hel', 'lo"'])
    stream = ai.generate_stream(prompt='hi', output_schema=LabelRoot)
    outputs = [chunk.output async for chunk in stream.stream]
    response = await stream.response

    assert outputs == [None, LabelRoot('hello')]
    assert response.output == LabelRoot('hello')
    assert hydrate_partial(schema=LabelRoot, data='hel', closed=False) == LabelRoot('hel')
    assert hydrate_partial(schema=InvoiceList, data=[{'sku': 'CH-1'}], closed=False) == InvoiceList([
        InvoiceItem.model_construct(sku='CH-1', quantity=None)
    ])

    wrong_type = await _ai_with_reply('"not-an-int"').generate(prompt='hi', output_schema=ScoreRoot)
    assert wrong_type.output is None
    assert wrong_type.error is not None
    assert wrong_type.error.details == {'reason': 'INVALID_OUTPUT'}

    # Odd integer passes JSON schema ('type': 'integer') and must still fail Pydantic validation.
    odd_score = await _ai_with_reply('3').generate(prompt='hi', output_schema=EvenScoreRoot)
    assert odd_score.output is None
    assert odd_score.error is not None
    assert odd_score.error.details == {'reason': 'INVALID_OUTPUT'}


@pytest.mark.asyncio
async def test_enum_format_returns_member() -> None:
    response = await _ai_with_reply('ESCALATE').generate(prompt='hi', output_schema=TriageStr, output_format='enum')

    assert response.output is TriageStr.ESCALATE


@pytest.mark.asyncio
async def test_enum_format_stream_waits_when_chunk_is_prefix_of_longer_member() -> None:
    stream_plus = _ai_with_reply('PRO_PLUS', chunks=['PRO', '_PLUS']).generate_stream(
        prompt='hi', output_schema=Plan, output_format='enum'
    )
    plus_chunks = [chunk.output async for chunk in stream_plus.stream]
    plus_response = await stream_plus.response

    assert plus_chunks == [None, Plan.PRO_PLUS]
    assert plus_response.output is Plan.PRO_PLUS

    stream_pro = _ai_with_reply('PRO', chunks=['PRO']).generate_stream(
        prompt='hi', output_schema=Plan, output_format='enum'
    )
    pro_chunks = [chunk.output async for chunk in stream_pro.stream]
    pro_response = await stream_pro.response

    assert pro_chunks == [None]
    assert pro_response.output is Plan.PRO


@pytest.mark.asyncio
async def test_array_format_with_list_schema_hydrates_chunks_and_response() -> None:
    ai = _ai_with_reply(
        '[{"sku": "CH-1", "quantity": 2}, {"sku": "DK-2", "quantity": 5}]',
        chunks=['[{"sku": "CH-1", "quantity": 2}, ', '{"sku": "DK-2", "quantity": 5}]'],
    )
    stream = ai.generate_stream(prompt='hi', output_schema=list[InvoiceItem], output_format='array')
    outputs = [chunk.output async for chunk in stream.stream]
    response = await stream.response

    assert outputs == [[InvoiceItem(sku='CH-1', quantity=2)], [InvoiceItem(sku='DK-2', quantity=5)]]
    assert response.output == [InvoiceItem(sku='CH-1', quantity=2), InvoiceItem(sku='DK-2', quantity=5)]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('schema', 'expected_items'),
    [
        (list[Vendor], [Vendor('Acme', 12), Vendor('Beta', 5)]),
        (list[VendorDict], [{'vendor_name': 'Acme', 'risk_score': 12}, {'vendor_name': 'Beta', 'risk_score': 5}]),
        (Annotated[list[Vendor], Field(min_length=2)], [Vendor('Acme', 12), Vendor('Beta', 5)]),
    ],
    ids=['dataclass', 'typed_dict', 'annotated_min_length'],
)
async def test_array_format_hydrates_per_item_deltas(schema: Any, expected_items: list[Any]) -> None:  # noqa: ANN401
    full_text = f'[{VENDOR_JSON}, {BETA_VENDOR_JSON}]'
    ai = _ai_with_reply(
        full_text,
        chunks=['[{"vendor_name": "Ac', 'me", "risk_score": 12}, ', BETA_VENDOR_JSON, ']'],
    )
    stream = ai.generate_stream(prompt='hi', output_schema=schema, output_format='array')
    outputs = [chunk.output async for chunk in stream.stream]
    response = await stream.response

    assert outputs == [[], [expected_items[0]], [expected_items[1]], []]
    assert response.output == expected_items


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('schema', 'expected_items'),
    [
        (list[Vendor], [Vendor('Acme', 12), Vendor('Beta', 5)]),
        (list[VendorDict], [{'vendor_name': 'Acme', 'risk_score': 12}, {'vendor_name': 'Beta', 'risk_score': 5}]),
        (Annotated[list[Vendor], Field(min_length=2)], [Vendor('Acme', 12), Vendor('Beta', 5)]),
    ],
    ids=['dataclass', 'typed_dict', 'annotated_min_length'],
)
async def test_jsonl_format_hydrates_per_item_deltas(schema: Any, expected_items: list[Any]) -> None:  # noqa: ANN401
    full_text = f'{VENDOR_JSON}\n{BETA_VENDOR_JSON}\n'
    ai = _ai_with_reply(
        full_text,
        chunks=[f'{VENDOR_JSON}\n{{"vendor_name": "Be', 'ta", "risk_score": 5}\n'],
    )
    stream = ai.generate_stream(prompt='hi', output_schema=schema, output_format='jsonl')
    outputs = [chunk.output async for chunk in stream.stream]
    response = await stream.response

    assert outputs == [[expected_items[0]], [expected_items[1]]]
    assert response.output == expected_items


@pytest.mark.asyncio
async def test_array_format_still_enforces_list_constraints_on_final_response() -> None:
    ai = _ai_with_reply(f'[{VENDOR_JSON}]', chunks=[f'[{VENDOR_JSON}', ']'])
    stream = ai.generate_stream(
        prompt='hi',
        output_schema=Annotated[list[Vendor], Field(min_length=2)],
        output_format='array',
    )
    outputs = [chunk.output async for chunk in stream.stream]
    response = await stream.response

    assert outputs == [[Vendor('Acme', 12)], []]
    assert response.output is None
    assert response.error is not None
    assert response.error.details == {'reason': 'INVALID_OUTPUT'}


@pytest.mark.asyncio
async def test_stream_model_with_nested_dataclass_and_typed_dict() -> None:
    reply = f'{{"lines": [{VENDOR_JSON}], "contact": {{"vendor": {VENDOR_JSON}, "emails": []}}}}'
    ai = _ai_with_reply(reply, chunks=[reply[:30], reply[30:]])
    stream = ai.generate_stream(prompt='hi', output_schema=PurchaseOrder)
    outputs = [chunk.output async for chunk in stream.stream]

    # Open: the model is partial; the open TypedDict line stays a raw dict.
    assert isinstance(outputs[0], PurchaseOrder)
    assert outputs[0].lines == [{'vendor_name': 'Ac'}]
    assert outputs[0].contact is None
    # Closed: the nested dataclass is hydrated, same as the final response.
    expected = PurchaseOrder(
        lines=[{'vendor_name': 'Acme', 'risk_score': 12}],
        contact=Contact(vendor=Vendor(vendor_name='Acme', risk_score=12), emails=[]),
    )
    assert outputs[1] == expected
    assert (await stream.response).output == expected


@pytest.mark.asyncio
async def test_stream_chunks_of_list_schema_are_models() -> None:
    ai = _ai_with_reply(
        '[{"sku": "CH-1", "quantity": 2}]',
        chunks=['[{"sku": "CH-1", "qu', 'antity": 2}]'],
    )
    stream = ai.generate_stream(prompt='hi', output_schema=list[InvoiceItem])
    outputs = [chunk.output async for chunk in stream.stream]

    # Open: the half-written item is an InvoiceItem with holes.
    first = outputs[0]
    assert first is not None
    assert isinstance(first[0], InvoiceItem)
    assert first[0].sku == 'CH-1'
    assert first[0].quantity is None
    # Closed: validated like the final response.
    assert outputs[1] == [InvoiceItem(sku='CH-1', quantity=2)]
    assert (await stream.response).output == outputs[1]


@pytest.mark.asyncio
async def test_stream_set_and_frozenset_schemas() -> None:
    set_stream = _ai_with_reply('["a", "b"]', chunks=['["a", "b', '"]']).generate_stream(
        prompt='hi', output_schema=set[str]
    )
    set_chunks = [chunk.output async for chunk in set_stream.stream]
    assert set_chunks == [{'a', 'b'}, {'a', 'b'}]
    assert (await set_stream.response).output == {'a', 'b'}

    frozen_stream = _ai_with_reply('[1, 2]', chunks=['[1, ', '2]']).generate_stream(
        prompt='hi', output_schema=frozenset[int]
    )
    frozen_chunks = [chunk.output async for chunk in frozen_stream.stream]
    assert frozen_chunks == [frozenset({1}), frozenset({1, 2})]
    assert (await frozen_stream.response).output == frozenset({1, 2})


@pytest.mark.asyncio
async def test_define_prompt_accepts_generic_alias_schema() -> None:
    ai = _ai_with_reply('[{"sku": "CH-1", "quantity": 2}]')
    prompt = ai.define_prompt(name='invoice', prompt='List items.', output_schema=list[InvoiceItem])

    response = await prompt()

    assert response.output == [InvoiceItem(sku='CH-1', quantity=2)]


@pytest.mark.asyncio
async def test_looked_up_prompt_keeps_caller_generic_alias_schema() -> None:
    ai = _ai_with_reply('[{"sku": "CH-1", "quantity": 2}]')
    ai.define_prompt(name='invoice', prompt='List items.', output_schema={'type': 'array'})

    prompt = ai.prompt('invoice', output_schema=list[InvoiceItem])
    response = await prompt()

    assert response.output == [InvoiceItem(sku='CH-1', quantity=2)]


class Unschemable:
    pass


@pytest.mark.asyncio
@pytest.mark.parametrize('bad_schema', [[InvoiceItem], 123, Unschemable], ids=['list_instance', 'int', 'plain_class'])
async def test_invalid_output_schema_raises_invalid_output_schema_error(bad_schema: Any) -> None:  # noqa: ANN401
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{}')]))]

    with pytest.raises(InvalidOutputSchemaError) as exc_info:
        await ai.generate(prompt='hi', output_schema=bad_schema)

    assert isinstance(exc_info.value, GenkitError)
    assert exc_info.value.status == 'INVALID_ARGUMENT'
    assert exc_info.value.reason is ErrorReason.INVALID_SCHEMA
    assert pm.request_count == 0

    with pytest.raises(InvalidOutputSchemaError):
        to_json_schema(bad_schema)

    with pytest.raises(InvalidOutputSchemaError):
        output_adapter(bad_schema)


class Dish(BaseModel):
    name: str

    @field_validator('name')
    @classmethod
    def _capitalized(cls, v: str) -> str:
        if not v[:1].isupper():
            raise ValueError('must be capitalized')
        return v


def test_construct_partial_closed_raises_instead_of_model_construct() -> None:
    with pytest.raises(ValidationError):
        construct_partial(schema_type=Dish, data={'name': 'salmon'}, closed=True)


def test_construct_partial_open_still_builds() -> None:
    out = construct_partial(schema_type=Dish, data={'name': 'salm'})

    assert isinstance(out, Dish)
    assert out.name == 'salm'


def test_hydrate_partial_enum_prefix_is_none_until_whole() -> None:
    assert hydrate_partial(schema=TriageStr, data='ESC', closed=False) is None
    assert hydrate_partial(schema=TriageStr, data='ESCALATE', closed=False) is TriageStr.ESCALATE
    assert hydrate_partial(schema=Plan, data='PRO', closed=False) is None
    assert hydrate_partial(schema=Plan, data='PRO_PLUS', closed=False) is Plan.PRO_PLUS
    assert hydrate_partial(schema=Plan, data='PRO', closed=True) is Plan.PRO


def test_hydrate_partial_dataclass_stays_raw_until_closed() -> None:
    assert hydrate_partial(schema=Vendor, data={'vendor_name': 'Ac'}, closed=False) == {'vendor_name': 'Ac'}
    assert hydrate_partial(schema=Vendor, data={'vendor_name': 'Acme', 'risk_score': 1}, closed=True) == Vendor(
        vendor_name='Acme', risk_score=1
    )


def test_unwrap_annotated_unwraps_pep695_type_aliases() -> None:
    nested = Annotated[VendorListAlias, Field(min_length=1)]
    assert unwrap_annotated(annotation=nested) == list[Vendor]

    std_alias_cls = getattr(typing, 'TypeAliasType', None)
    if std_alias_cls is not None and sys.version_info >= (3, 12):
        ns: dict[str, Any] = {'InvoiceItem': InvoiceItem}
        exec('type Items = list[InvoiceItem]\ntype Box[T] = list[T]', ns)  # noqa: S102
        assert unwrap_annotated(annotation=ns['Items']) == list[InvoiceItem]
        assert unwrap_annotated(annotation=ns['Box'][InvoiceItem]) == list[InvoiceItem]
        partial = hydrate_partial(schema=ns['Items'], data=[{'sku': 'CH-1'}], closed=False)
        assert partial == [InvoiceItem.model_construct(sku='CH-1', quantity=None)]


def test_anchor_type_skips_stdlib_value_modules() -> None:
    assert anchor_type(list[uuid.UUID]) is object
    assert anchor_type(list[Decimal]) is object
    assert anchor_type(list[datetime.datetime]) is object
    assert anchor_type(list[pathlib.Path]) is object
    assert anchor_type(list[ipaddress.IPv4Address]) is object
    assert anchor_type(dict[uuid.UUID, InvoiceItem]) is InvoiceItem
    assert anchor_type(VendorListAlias) is Vendor
