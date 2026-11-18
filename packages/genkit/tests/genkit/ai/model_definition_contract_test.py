#!/usr/bin/env python3
#
# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""define_model accepts ModelRequest / ModelRequest[Cfg] and rejects the rest."""

import functools
import sys
import typing
from collections.abc import Awaitable, Callable
from typing import Annotated, Any, Optional, TypeVar, cast

import pytest
import typing_extensions
from pydantic import BaseModel
from typing_extensions import ParamSpec

from genkit import Genkit, Part
from genkit._core._error import GenkitError
from genkit._core._model import Message, ModelRequest, ModelResponse
from genkit._core._typing import Role
from genkit.model import ModelContext


class Cfg(BaseModel):
    """Sample typed plugin config."""

    temperature: float | None = None


OK = ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('ok')]))


@pytest.fixture
def ai() -> Genkit:
    return Genkit()


# --- allowed shapes -----------------------------------------------------------


def test_typed_annotation_allowed(ai: Genkit) -> None:
    """define_model(fn with ModelRequest[Cfg]) registers."""

    async def fn(request: ModelRequest[Cfg], ctx: ModelContext) -> ModelResponse:
        return OK

    ai.define_model(name='typed', fn=fn)


def test_bare_annotation_allowed(ai: Genkit) -> None:
    """define_model(fn with ModelRequest) registers. Config stays a dict."""

    async def fn(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        return OK

    ai.define_model(name='bare', fn=fn)


def test_unannotated_allowed(ai: Genkit) -> None:
    """define_model(fn with no request annotation) still registers."""

    async def fn(request, ctx: ModelContext) -> ModelResponse:  # noqa: ANN001
        return OK

    ai.define_model(name='unannotated', fn=fn)


def test_annotated_wrapper_unwrapped_and_allowed(ai: Genkit) -> None:
    """Annotated[ModelRequest[Cfg], ...] is treated as ModelRequest[Cfg]."""

    async def fn(request: Annotated[ModelRequest[Cfg], 'doc'], ctx: ModelContext) -> ModelResponse:
        return OK

    ai.define_model(name='annotated', fn=fn)


def test_any_parametrization_allowed(ai: Genkit) -> None:
    """ModelRequest[Any] registers like bare ModelRequest; no config class to check."""

    async def fn(request: ModelRequest[Any], ctx: ModelContext) -> ModelResponse:
        return OK

    ai.define_model(name='param_any', fn=fn)


# --- rejected antipatterns ----------------------------------------------------


class StdlibTypedDictConfig(typing.TypedDict, total=False):
    """Pydantic itself rejects this before Python 3.12; ours fires first."""

    temperature: float


class ExtensionsTypedDictConfig(typing_extensions.TypedDict, total=False):
    temperature: float


@pytest.mark.parametrize(
    'config_type',
    [StdlibTypedDictConfig, ExtensionsTypedDictConfig, dict, dict[str, Any], Cfg | None],
    ids=['typing.TypedDict', 'typing_extensions.TypedDict', 'dict', 'dict[str, Any]', 'union'],
)
def test_non_model_config_type_rejected(config_type: object) -> None:
    """ModelRequest[X] raises where the annotation is evaluated unless X is a pydantic model."""
    with pytest.raises(GenkitError, match='config type must be a pydantic BaseModel subclass') as exc:
        cast(Any, ModelRequest)[config_type]

    assert exc.value.status == 'INVALID_ARGUMENT'


# A user module with a module-level TypedDict config. Under the future import
# (and by default from Python 3.14) the annotation is read at define_model.
_TYPED_DICT_CONFIG_MODULE = """
from __future__ import annotations

from typing import TypedDict


class RestaurantConfig(TypedDict, total=False):
    temperature: float


async def menu_model(request: ModelRequest[RestaurantConfig], ctx: ModelContext) -> ModelResponse:
    return OK
"""


@pytest.mark.parametrize(
    'postponed',
    [
        True,
        pytest.param(
            False,
            marks=pytest.mark.skipif(sys.version_info < (3, 14), reason='annotations are evaluated lazily from 3.14'),
        ),
    ],
    ids=['future_import', 'lazy_3_14'],
)
def test_typed_dict_config_rejected_through_define_model(ai: Genkit, postponed: bool) -> None:
    """define_model raises the config-type error for ModelRequest[RestaurantConfig] when annotations are deferred."""
    source = _TYPED_DICT_CONFIG_MODULE
    if not postponed:
        source = source.replace('from __future__ import annotations\n', '')
    module_globals: dict[str, Any] = {
        'ModelRequest': ModelRequest,
        'ModelResponse': ModelResponse,
        'ModelContext': ModelContext,
        'OK': OK,
    }
    exec(source, module_globals)  # noqa: S102 - builds a user module

    with pytest.raises(
        GenkitError, match=r'ModelRequest\[RestaurantConfig\]: the config type must be a pydantic'
    ) as exc:
        ai.define_model(name='menu', fn=module_globals['menu_model'])

    assert exc.value.status == 'INVALID_ARGUMENT'


def test_union_with_none_rejected(ai: Genkit) -> None:
    """ModelRequest[Cfg] | None is rejected — generate never passes None."""

    async def fn(request: ModelRequest[Cfg] | None, ctx: ModelContext) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='union', fn=fn)


def test_optional_spelling_rejected(ai: Genkit) -> None:
    """Optional[ModelRequest[Cfg]] is the same reject as ModelRequest[Cfg] | None."""

    async def fn(request: Optional[ModelRequest[Cfg]], ctx: ModelContext) -> ModelResponse:  # noqa: UP045
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='optional', fn=fn)


def test_dict_annotation_rejected(ai: Genkit) -> None:
    """A handler annotated dict is rejected. The request is a ModelRequest."""

    async def fn(request: dict, ctx: ModelContext) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='rawdict', fn=fn)


def test_arbitrary_class_rejected(ai: Genkit) -> None:
    """A handler annotated with some other class is rejected. The request is a ModelRequest."""

    class NotARequest(BaseModel):
        pass

    async def fn(request: NotARequest, ctx: ModelContext) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='arbitrary', fn=fn)


# --- context first ------------------------------------------------------------


def test_ctx_first_union_with_none_rejected(ai: Genkit) -> None:
    """The request check finds the request after the context: ``(ctx, req: ModelRequest[Cfg] | None)`` is rejected."""

    async def fn(ctx: ModelContext, request: ModelRequest[Cfg] | None) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match=r"Model 'ctx_union': the request parameter must be annotated") as exc:
        ai.define_model(name='ctx_union', fn=fn)
    assert 'ModelContext' not in str(exc.value)


def test_ctx_first_dict_annotation_rejected(ai: Genkit) -> None:
    """``(ctx, req: dict)`` is rejected the same as ``(req: dict, ctx)``."""

    async def fn(ctx: ModelContext, request: dict) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match='must be annotated as ModelRequest'):
        ai.define_model(name='ctx_rawdict', fn=fn)


# --- decorated model fns ------------------------------------------------------

P = ParamSpec('P')
R = TypeVar('R')


def _timed_without_wraps(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await fn(*args, **kwargs)

    return wrapper


def _timed(fn: Callable[P, Awaitable[R]]) -> Callable[P, Awaitable[R]]:
    @functools.wraps(fn)
    async def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
        return await fn(*args, **kwargs)

    return wrapper


def test_wrapper_without_functools_wraps_rejected(ai: Genkit) -> None:
    """A decorator that drops the signature raises at definition instead of failing every generate()."""

    @_timed_without_wraps
    async def chef(request: ModelRequest[Cfg], ctx: ModelContext) -> ModelResponse:
        return OK

    with pytest.raises(GenkitError, match=r"Model 'timed': the function only takes \*args / \*\*kwargs") as exc:
        ai.define_model(name='timed', fn=chef)
    assert exc.value.status == 'INVALID_ARGUMENT'
    assert 'functools.wraps' in str(exc.value)


@pytest.mark.asyncio
async def test_wrapper_with_functools_wraps_runs(ai: Genkit) -> None:
    """With functools.wraps the wrapper keeps the signature, so it defines and answers generate."""

    @_timed
    async def chef(request: ModelRequest[Cfg], ctx: ModelContext) -> ModelResponse:
        return OK

    ai.define_model(name='timed', fn=chef)

    assert (await ai.generate(model='timed', prompt='hi')).text == 'ok'


def test_no_parameters_still_allowed(ai: Genkit) -> None:
    """A model fn with no parameters ignores the request; it still registers."""

    async def fn() -> ModelResponse:
        return OK

    ai.define_model(name='no_params', fn=cast(Any, fn))
