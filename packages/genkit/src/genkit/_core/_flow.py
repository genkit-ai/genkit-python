# Copyright 2025 Google LLC
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

"""Flow action class and registration for Genkit."""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from typing import Any, overload

from typing_extensions import TypeVar

from genkit._core._action import (
    Action,
    ActionKind,
    ChunkT,
    CtxDefaultInputFn,
    DefaultedAction,
    DefaultInputCtxFn,
    DefaultInputFn,
    FlowContext,
    InputT,
    OutputT,
    get_func_description,
    input_has_default,
)
from genkit._core._async_call import is_async_callable
from genkit._core._registry import Registry

_FlowInputT = TypeVar('_FlowInputT')
_FlowOutputT = TypeVar('_FlowOutputT')
_FlowChunkT = TypeVar('_FlowChunkT')


class Flow(Action[InputT, OutputT, ChunkT, None]):
    """A typed, traced, streamable flow action.

    ``@ai.flow`` constructs and registers a ``Flow`` on the ``Genkit`` instance.
    Constructing ``Flow(fn)`` directly builds a standalone (unregistered) flow.
    """

    # `(ctx, x: Any)` would match the (input, ctx) shape with the ctx as the
    # input; catch it first. InputT isn't bound here, so it defaults to Any.
    @overload
    def __init__(
        self,
        func: Callable[[FlowContext[ChunkT], object], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    # The defaulted-input shapes are self-typed to DefaultedFlow, so they
    # apply when define_flow builds one. Flow(fn) itself stays a Flow and
    # skips them. `(x = ..., ctx = None)` also fits (input, ctx); catch it first.
    @overload
    def __init__(
        self: DefaultedFlow[_FlowInputT, _FlowOutputT, _FlowChunkT],
        func: DefaultInputCtxFn[FlowContext[_FlowChunkT], _FlowInputT, _FlowOutputT],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[InputT, FlowContext[ChunkT]], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: DefaultedFlow[_FlowInputT, _FlowOutputT, _FlowChunkT],
        func: CtxDefaultInputFn[FlowContext[_FlowChunkT], _FlowInputT, _FlowOutputT],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[FlowContext[ChunkT], InputT], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[FlowContext[ChunkT]], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: DefaultedFlow[_FlowInputT, _FlowOutputT, Any],
        func: DefaultInputFn[_FlowInputT, _FlowOutputT],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[InputT], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    @overload
    def __init__(
        self,
        func: Callable[[], Awaitable[OutputT]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None: ...

    def __init__(
        self,
        func: Callable[..., Awaitable[Any]],
        *,
        name: str | None = None,
        description: str | None = None,
    ) -> None:
        """Initialize a ``Flow`` from ``func``.

        Args:
            func: Async flow handler taking at most one input and an optional
                ``FlowContext[ChunkT]`` parameter.
            name: Optional flow name. Defaults to ``func.__name__``.
            description: Optional description. Defaults to ``func.__doc__``.
        """
        if not is_async_callable(func):
            raise TypeError(f'Flow must be async: {getattr(func, "__name__", repr(func))}')

        flow_name = name or getattr(func, '__name__', None) or type(func).__name__
        super().__init__(
            kind=ActionKind.FLOW,
            name=flow_name,
            fn=func,
            description=get_func_description(func, description),
            span_metadata={'flow:name': flow_name},
            _strict_io=True,
        )


class DefaultedFlow(DefaultedAction[InputT, OutputT, ChunkT, None], Flow[InputT, OutputT, ChunkT]):
    """A ``Flow`` whose function gives the input a default.

    ``@ai.flow`` and ``define_flow`` return one for ``async def
    todays_special(cuisine: str = 'italian')``. ``await todays_special()``
    runs with the default; ``await todays_special(None)`` is a type error.
    It's still a ``Flow``, so a ``Flow[str, str]`` annotation accepts it.
    """


# Same order as _FlowDecorator.__call__; the `object` shapes catch
# `(x: Any)` / `(x: object)` before the ctx-only shape can.
@overload
def define_flow(
    registry: Registry,
    func: Callable[[FlowContext[_FlowChunkT], object], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[Any, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: DefaultInputCtxFn[FlowContext[_FlowChunkT], _FlowInputT, _FlowOutputT],
    name: str | None = None,
    description: str | None = None,
) -> DefaultedFlow[_FlowInputT, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[_FlowInputT, FlowContext[_FlowChunkT]], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[_FlowInputT, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: CtxDefaultInputFn[FlowContext[_FlowChunkT], _FlowInputT, _FlowOutputT],
    name: str | None = None,
    description: str | None = None,
) -> DefaultedFlow[_FlowInputT, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[FlowContext[_FlowChunkT], _FlowInputT], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[_FlowInputT, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[object], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[Any, _FlowOutputT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[FlowContext[_FlowChunkT]], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[None, _FlowOutputT, _FlowChunkT]: ...


@overload
def define_flow(
    registry: Registry,
    func: DefaultInputFn[_FlowInputT, _FlowOutputT],
    name: str | None = None,
    description: str | None = None,
) -> DefaultedFlow[_FlowInputT, _FlowOutputT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[_FlowInputT], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[_FlowInputT, _FlowOutputT]: ...


@overload
def define_flow(
    registry: Registry,
    func: Callable[[], Awaitable[_FlowOutputT]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[None, _FlowOutputT]: ...


def define_flow(
    registry: Registry,
    func: Callable[..., Awaitable[Any]],
    name: str | None = None,
    description: str | None = None,
) -> Flow[Any, Any, Any]:
    """Register an async function as a flow action.

    A function whose input has a default becomes a ``DefaultedFlow``.
    """
    cls = DefaultedFlow if input_has_default(func, ActionKind.FLOW) else Flow
    flow = cls(func, name=name, description=description)
    registry.register_action_from_instance(flow)
    return flow
