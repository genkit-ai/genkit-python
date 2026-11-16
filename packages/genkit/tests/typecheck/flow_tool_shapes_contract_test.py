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

# pyright: reportUnnecessaryTypeIgnoreComment=true

"""Type-checker contract for every flow and tool handler shape.

Every shape the ``@ai.flow`` / ``@ai.tool`` / ``define_flow`` /
``define_tool`` overloads cover is pinned here with its exact type, so a
shape that resolves to the wrong overload fails CI. Shapes Genkit doesn't
support (a keyword-only parameter, two inputs) fail at the decorator in all
three checkers and raise ``TypeError`` at definition. The runtime checks
assert the built object is a ``DefaultedFlow`` / ``DefaultedTool`` exactly
when the input has a default, so ``isinstance`` agrees with the static type.
"""

import copy
from typing import Any

import pytest
from pydantic import BaseModel
from typing_extensions import assert_type

from genkit import Flow, FlowContext, Genkit, GenkitError, Tool, ToolContext
from genkit._ai._tools import define_tool
from genkit._core._flow import DefaultedFlow, define_flow
from genkit._core._registry import Registry
from genkit._core._tool import DefaultedTool
from genkit.plugin_api import ActionContext


class Diet(BaseModel):
    vegan: bool = False


class Order(BaseModel):
    dish: str


# Flow handlers, one per shape.


async def recommend_dish(diet: Diet) -> str:
    return 'tofu larb' if diet.vegan else 'pork larb'


async def todays_special(cuisine: str = 'italian') -> str:
    return f'{cuisine}: risotto'


async def lookup_reservation(res_id: str | None = None) -> str:
    return res_id or 'walk-in'


async def staff_on_shift(ctx: FlowContext) -> str:
    del ctx
    return 'chef, two cooks'


async def kitchen_status() -> str:
    return 'open'


async def plan_menu(ctx: FlowContext, diet: Diet) -> str:
    del ctx
    return 'tasting menu' if diet.vegan else 'prix fixe'


async def plate_dish(diet: Diet, ctx: FlowContext) -> str:
    del ctx
    return 'plated' if diet.vegan else 'plated with jus'


async def chef_pick(ctx: FlowContext, cuisine: str = 'thai') -> str:
    del ctx
    return f'{cuisine}: chef pick'


async def special(cuisine: str = 'thai', ctx: FlowContext | None = None) -> str:
    del ctx
    return cuisine


async def streamed_special(cuisine: str = 'thai', ctx: FlowContext[str] | None = None) -> str:
    if ctx is not None:
        ctx.send_chunk('plating')
    return cuisine


async def open_tables(ctx: ActionContext) -> str:
    del ctx
    return '4 tables'


async def echo_order(order: Any) -> str:  # noqa: ANN401 - pins the Any-input shape
    return str(order)


# Unsupported: a keyword-only context, and two inputs.


async def kwonly_special(cuisine: str = 'thai', *, ctx: FlowContext) -> str:
    del ctx
    return cuisine


async def pair_dishes(main: str, side: str) -> str:
    return f'{main} with {side}'


# Tool handlers, one per shape.


async def place_order(order: Order) -> str:
    return f'ordered {order.dish}'


async def check_allergens(dish: str = 'pad thai') -> str:
    return f'{dish}: peanuts'


async def find_table(res_id: str | None = None) -> str:
    return res_id or 'bar seat'


async def whoami(ctx: ToolContext) -> str:
    del ctx
    return 'server 3'


async def list_specials() -> str:
    return 'soup, tartine'


async def fire_order(ctx: ToolContext, order: Order) -> str:
    del ctx
    return f'fired {order.dish}'


async def void_order(order: Order, ctx: ToolContext) -> str:
    del ctx
    return f'voided {order.dish}'


async def suggest_wine(ctx: ToolContext, dish: str = 'steak') -> str:
    del ctx
    return f'{dish}: malbec'


async def soup_of_day(day: str = 'monday', ctx: ToolContext | None = None) -> str:
    del ctx
    return f'{day}: bouillabaisse'


async def echo_ticket(ticket: Any) -> str:  # noqa: ANN401 - pins the Any-input shape
    return str(ticket)


# Unsupported tool shapes.


async def kwonly_soup(day: str = 'monday', *, ctx: ToolContext) -> str:
    del ctx
    return day


async def split_bill(total: str, guests: str) -> str:
    return f'{total} / {guests}'


def test_ai_flow_shapes() -> None:
    """``@ai.flow`` types every handler shape exactly; defaulted inputs build a ``DefaultedFlow``."""
    ai = Genkit()

    # 1. Required, defaulted and Optional-typed inputs
    assert_type(ai.flow(recommend_dish), Flow[Diet, str])
    assert_type(ai.flow(todays_special), DefaultedFlow[str, str])
    assert_type(ai.flow(lookup_reservation), DefaultedFlow[str | None, str])

    # 2. No input
    assert_type(ai.flow(staff_on_shift), Flow[None, str])
    assert_type(ai.flow(kitchen_status), Flow[None, str])
    assert_type(ai.flow(open_tables), Flow[None, str])

    # 3. Input plus context, in either order
    assert_type(ai.flow(plan_menu), Flow[Diet, str])
    assert_type(ai.flow(plate_dish), Flow[Diet, str])
    assert_type(ai.flow(chef_pick), DefaultedFlow[str, str])
    assert_type(ai.flow(special), DefaultedFlow[str, str])
    assert_type(ai.flow(streamed_special), DefaultedFlow[str, str, str])

    # 4. Any input; ty types it Unknown (see _FlowDecorator)
    assert_type(ai.flow(echo_order), Flow[Any, str])  # ty: ignore[type-assertion-failure]


def test_ai_flow_called_shapes() -> None:
    """``@ai.flow()`` (the decorator object) types the same shapes the same way."""
    ai = Genkit()
    flow = ai.flow()

    assert_type(flow(recommend_dish), Flow[Diet, str])
    assert_type(flow(todays_special), DefaultedFlow[str, str])
    assert_type(flow(lookup_reservation), DefaultedFlow[str | None, str])
    assert_type(flow(staff_on_shift), Flow[None, str])
    assert_type(flow(kitchen_status), Flow[None, str])
    assert_type(flow(open_tables), Flow[None, str])
    assert_type(flow(plan_menu), Flow[Diet, str])
    assert_type(flow(plate_dish), Flow[Diet, str])
    assert_type(flow(chef_pick), DefaultedFlow[str, str])
    assert_type(flow(special), DefaultedFlow[str, str])
    assert_type(flow(streamed_special), DefaultedFlow[str, str, str])
    assert_type(flow(echo_order), Flow[Any, str])  # ty: ignore[type-assertion-failure]


def test_define_flow_shapes() -> None:
    """``define_flow`` types the same shapes the same way."""
    registry = Registry()

    assert_type(define_flow(registry, recommend_dish), Flow[Diet, str])
    assert_type(define_flow(registry, todays_special), DefaultedFlow[str, str])
    assert_type(define_flow(registry, lookup_reservation), DefaultedFlow[str | None, str])
    assert_type(define_flow(registry, staff_on_shift), Flow[None, str])
    assert_type(define_flow(registry, kitchen_status), Flow[None, str])
    assert_type(define_flow(registry, open_tables), Flow[None, str])
    assert_type(define_flow(registry, plan_menu), Flow[Diet, str])
    assert_type(define_flow(registry, plate_dish), Flow[Diet, str])
    assert_type(define_flow(registry, chef_pick), DefaultedFlow[str, str])
    assert_type(define_flow(registry, special), DefaultedFlow[str, str])
    assert_type(define_flow(registry, streamed_special), DefaultedFlow[str, str, str])
    assert_type(define_flow(registry, echo_order), Flow[Any, str])  # ty: ignore[type-assertion-failure]


def test_ai_tool_shapes() -> None:
    """``@ai.tool`` types every handler shape exactly; defaulted inputs build a ``DefaultedTool``."""
    ai = Genkit()

    assert_type(ai.tool(place_order), Tool[Order, str])
    assert_type(ai.tool(check_allergens), DefaultedTool[str, str])
    assert_type(ai.tool(find_table), DefaultedTool[str | None, str])
    assert_type(ai.tool(whoami), Tool[None, str])
    assert_type(ai.tool(list_specials), Tool[None, str])
    assert_type(ai.tool(fire_order), Tool[Order, str])
    assert_type(ai.tool(void_order), Tool[Order, str])
    assert_type(ai.tool(suggest_wine), DefaultedTool[str, str])
    assert_type(ai.tool(soup_of_day), DefaultedTool[str, str])
    assert_type(ai.tool(echo_ticket), Tool[Any, str])  # ty: ignore[type-assertion-failure]


def test_ai_tool_called_shapes() -> None:
    """``@ai.tool()`` (the decorator object) types the same shapes the same way."""
    ai = Genkit()
    tool = ai.tool()

    assert_type(tool(place_order), Tool[Order, str])
    assert_type(tool(check_allergens), DefaultedTool[str, str])
    assert_type(tool(find_table), DefaultedTool[str | None, str])
    assert_type(tool(whoami), Tool[None, str])
    assert_type(tool(list_specials), Tool[None, str])
    assert_type(tool(fire_order), Tool[Order, str])
    assert_type(tool(void_order), Tool[Order, str])
    assert_type(tool(suggest_wine), DefaultedTool[str, str])
    assert_type(tool(soup_of_day), DefaultedTool[str, str])
    assert_type(tool(echo_ticket), Tool[Any, str])  # ty: ignore[type-assertion-failure]


def test_define_tool_shapes() -> None:
    """``define_tool`` types the same shapes the same way."""
    registry = Registry()

    assert_type(define_tool(registry, place_order), Tool[Order, str])
    assert_type(define_tool(registry, check_allergens), DefaultedTool[str, str])
    assert_type(define_tool(registry, find_table), DefaultedTool[str | None, str])
    assert_type(define_tool(registry, whoami), Tool[None, str])
    assert_type(define_tool(registry, list_specials), Tool[None, str])
    assert_type(define_tool(registry, fire_order), Tool[Order, str])
    assert_type(define_tool(registry, void_order), Tool[Order, str])
    assert_type(define_tool(registry, suggest_wine), DefaultedTool[str, str])
    assert_type(define_tool(registry, soup_of_day), DefaultedTool[str, str])
    assert_type(define_tool(registry, echo_ticket), Tool[Any, str])  # ty: ignore[type-assertion-failure]


def test_unsupported_shapes_fail_at_the_decorator() -> None:
    """A keyword-only parameter or a second input fails every checker at the decorator, and raises at definition."""
    ai = Genkit()

    # 1. Keyword-only context
    with pytest.raises(TypeError, match="flow 'kwonly_special' parameter 'ctx' is keyword-only"):
        ai.flow(kwonly_special)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match="tool 'kwonly_soup' parameter 'ctx' is keyword-only"):
        ai.tool()(kwonly_soup)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]

    # 2. Two inputs
    with pytest.raises(TypeError, match="flow 'pair_dishes' takes one input, but 'side' is a second parameter"):
        ai.flow(pair_dishes)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]
    with pytest.raises(TypeError, match="tool 'split_bill' takes one input, but 'guests' is a second parameter"):
        ai.tool()(split_bill)  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


def test_runtime_class_matches_the_static_type() -> None:
    """A defaulted input builds a ``DefaultedFlow`` / ``DefaultedTool``; anything else builds a plain one."""
    ai = Genkit()
    registry = Registry()

    # 1. Flows: a defaulted input, with or without a context, in any position
    for fn in (todays_special, lookup_reservation, chef_pick, special, streamed_special):
        assert isinstance(ai.flow(fn), DefaultedFlow), fn.__name__
        assert isinstance(define_flow(registry, fn), DefaultedFlow), fn.__name__
    for fn in (recommend_dish, staff_on_shift, kitchen_status, open_tables, plan_menu, plate_dish, echo_order):
        built = define_flow(registry, fn)
        assert isinstance(built, Flow), fn.__name__
        assert not isinstance(built, DefaultedFlow), fn.__name__

    # 2. Tools, the same way
    for fn in (check_allergens, find_table, suggest_wine, soup_of_day):
        assert isinstance(ai.tool(fn), DefaultedTool), fn.__name__
        assert isinstance(define_tool(registry, fn), DefaultedTool), fn.__name__
    for fn in (place_order, whoami, list_specials, fire_order, void_order, echo_ticket):
        built_tool = define_tool(registry, fn)
        assert isinstance(built_tool, Tool), fn.__name__
        assert not isinstance(built_tool, DefaultedTool), fn.__name__

    # 3. Flow(fn) / Tool(fn) built directly stay plain, matching their static type
    assert type(Flow(todays_special)) is Flow
    assert type(Tool(check_allergens)) is Tool


@pytest.mark.asyncio
async def test_defaulted_flow_calls() -> None:
    """A ``DefaultedFlow`` runs with no input or a ``T``; ``None`` and wrong types are errors."""
    ai = Genkit()

    @ai.flow
    async def todays_special(cuisine: str = 'italian') -> str:
        return f'{cuisine}: risotto'

    @ai.flow
    async def lookup_reservation(res_id: str | None = None) -> str:
        return res_id or 'walk-in'

    @ai.flow
    async def recommend_dish(diet: Diet) -> str:
        return 'tofu larb' if diet.vegan else 'pork larb'

    @ai.flow
    async def kitchen_status() -> str:
        return 'open'

    @ai.flow
    async def special(cuisine: str = 'thai', ctx: FlowContext | None = None) -> str:
        del ctx
        return cuisine

    # 1. Omit the input to use the default, or pass a str
    assert_type(await todays_special(), str)
    assert await todays_special() == 'italian: risotto'
    assert await todays_special('thai') == 'thai: risotto'
    assert (await todays_special.run()).response == 'italian: risotto'
    assert await todays_special.stream().response == 'italian: risotto'
    assert await todays_special.stream('thai').response == 'thai: risotto'

    # 2. None is a value when the input type includes it
    assert await lookup_reservation(None) == 'walk-in'
    assert await lookup_reservation() == 'walk-in'

    # 3. No-input flows, and a defaulted input before a context
    assert await kitchen_status() == 'open'
    assert await special() == 'thai'
    assert await special('isan') == 'isan'

    # 4. None isn't a str; neither is 3
    with pytest.raises(GenkitError):
        await todays_special(None)  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]
    with pytest.raises(GenkitError):
        await todays_special.run(None)  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]
    with pytest.raises(GenkitError):
        await todays_special(3)  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]

    # 5. A required input can't be omitted
    with pytest.raises(GenkitError):
        await recommend_dish()  # pyright: ignore[reportCallIssue]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]

    # 6. A no-input flow takes no input; at runtime the extra input is ignored
    assert await kitchen_status('x') == 'open'  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[no-matching-overload]


@pytest.mark.asyncio
async def test_defaulted_tool_calls() -> None:
    """A ``DefaultedTool`` runs with no input or a ``T``; ``None`` is an error."""
    ai = Genkit()

    @ai.tool()
    async def check_allergens(dish: str = 'pad thai') -> str:
        return f'{dish}: peanuts'

    assert await check_allergens() == 'pad thai: peanuts'
    assert await check_allergens('larb') == 'larb: peanuts'

    with pytest.raises(GenkitError):
        await check_allergens(None)  # pyright: ignore[reportArgumentType]  # pyrefly: ignore[no-matching-overload]  # ty: ignore[invalid-argument-type]


def test_defaulted_flows_fit_flow_annotations() -> None:
    """A ``DefaultedFlow`` / ``DefaultedTool`` is assignable wherever a ``Flow[T, O]`` / ``Tool[T, O]`` is."""
    ai = Genkit()

    @ai.flow
    async def todays_special(cuisine: str = 'italian') -> str:
        return f'{cuisine}: risotto'

    @ai.tool()
    async def check_allergens(dish: str = 'pad thai') -> str:
        return f'{dish}: peanuts'

    def menu_flow(f: Flow[str, str]) -> str:
        return f.name

    def menu_tool(t: Tool[str, str]) -> str:
        return t.name

    # 1. Annotate as Flow[str, str]
    by_cuisine: Flow[str, str] = todays_special
    allergens: Tool[str, str] = check_allergens

    # 2. Pass to a Flow[str, str] / Tool[str, str] parameter
    assert menu_flow(todays_special) == 'todays_special'
    assert menu_tool(check_allergens) == 'check_allergens'
    assert by_cuisine is todays_special
    assert allergens is check_allergens


@pytest.mark.asyncio
async def test_defaulted_tool_survives_copy_and_lookup() -> None:
    """``copy``, ``deepcopy``, ``Tool.from_action`` and registry lookup keep a ``DefaultedTool`` working."""
    ai = Genkit()

    @ai.tool()
    async def check_allergens(dish: str = 'pad thai') -> str:
        return f'{dish}: peanuts'

    # 1. Shallow and deep copies keep the class and still run with the default
    shallow = copy.copy(check_allergens)
    deep = copy.deepcopy(check_allergens)
    assert isinstance(shallow, DefaultedTool)
    assert isinstance(deep, DefaultedTool)
    assert await shallow() == 'pad thai: peanuts'
    assert await deep() == 'pad thai: peanuts'

    # 2. Tool.from_action returns the same object
    assert Tool.from_action(check_allergens) is check_allergens

    # 3. The registry hands back the DefaultedTool
    found = await ai.lookup_tool('check_allergens')
    assert found is check_allergens
