#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Dotprompt file loading: frontmatter output settings and per-file isolation."""

import tempfile
from pathlib import Path

import pytest

from genkit import Genkit, Message, ModelResponse, Part
from genkit._ai._prompt import strip_frontmatter
from genkit._core._typing import ConstrainedSupport, Role
from genkit.model import ModelContext, ModelInfo, ModelRequest, ModelSupports
from genkit.testing import define_scripted_model

_SCHEMA_PROMPT = """---
model: menuModel
output:
  schema:
    title: string
  constrained: {constrained}
---
Suggest a dish.
"""


def _define_menu_model(ai: Genkit) -> list[ModelRequest]:
    """A model that supports native JSON mode and records what it receives."""
    seen: list[ModelRequest] = []

    async def menu_model(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
        seen.append(request)
        return ModelResponse(message=Message(role=Role.MODEL, content=[Part.from_text('{"title": "Tartine"}')]))

    ai.define_model(
        name='menuModel', fn=menu_model, info=ModelInfo(supports=ModelSupports(constrained=ConstrainedSupport.ALL))
    )
    return seen


@pytest.mark.asyncio
@pytest.mark.parametrize('constrained', [False, True])
async def test_prompt_file_output_constrained_reaches_the_model(constrained: bool) -> None:
    """`constrained: false` in frontmatter turns native JSON mode off on the model request."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        text = _SCHEMA_PROMPT.format(constrained=str(constrained).lower())
        (Path(tmp_dir) / 'dish.prompt').write_text(text)
        ai = Genkit(prompt_dir=tmp_dir)
        seen = _define_menu_model(ai)

        await ai.prompt('dish')()

        assert seen[0].output is not None
        assert seen[0].output.constrained is constrained
        assert seen[0].output.schema is not None


@pytest.mark.asyncio
async def test_prompt_file_schema_without_constrained_defaults_to_native_json() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'dish.prompt').write_text(
            '---\nmodel: menuModel\noutput:\n  schema:\n    title: string\n---\nSuggest a dish.\n'
        )
        ai = Genkit(prompt_dir=tmp_dir)
        seen = _define_menu_model(ai)

        await ai.prompt('dish')()

        assert seen[0].output is not None
        assert seen[0].output.constrained is True


@pytest.mark.asyncio
async def test_prompt_file_content_type_reaches_the_model_without_a_format() -> None:
    """With no format or schema, frontmatter contentType is what the model sees."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'menu.prompt').write_text(
            '---\nmodel: menuModel\noutput:\n  contentType: text/markdown\n---\nWrite the menu.\n'
        )
        ai = Genkit(prompt_dir=tmp_dir)
        seen = _define_menu_model(ai)

        await ai.prompt('menu')()

        assert seen[0].output is not None
        assert seen[0].output.content_type == 'text/markdown'


@pytest.mark.asyncio
async def test_one_bad_prompt_file_does_not_drop_its_siblings(capfd: pytest.CaptureFixture[str]) -> None:
    """A YAML typo in one file is logged with its path and traceback; the rest of the folder loads."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        root = Path(tmp_dir)
        (root / 'a_broken.prompt').write_text('---\ndescription: Note: always greet\n  bad: [\n---\nHi\n')
        (root / 'b_greet.prompt').write_text('---\nmodel: scriptedModel\n---\nHello\n')
        (root / 'sub').mkdir()
        (root / 'sub' / 'menu.prompt').write_text('---\nmodel: scriptedModel\n---\nMenu\n')

        ai = Genkit(prompt_dir=tmp_dir, model='scriptedModel')
        define_scripted_model(ai)

        assert (await ai.prompt('b_greet').render()).messages
        assert (await ai.prompt('sub/menu').render()).messages
        out, err = capfd.readouterr()
        assert 'a_broken.prompt' in out + err
        assert 'Traceback' in out + err


@pytest.mark.asyncio
async def test_prompt_variant_keeps_every_dot_after_the_name() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        (Path(tmp_dir) / 'scorer.v2.1.prompt').write_text('---\nmodel: scriptedModel\n---\nScore it.\n')
        ai = Genkit(prompt_dir=tmp_dir, model='scriptedModel')
        define_scripted_model(ai)

        rendered = await ai.prompt('scorer', variant='v2.1').render()

        assert rendered.messages


def test_strip_frontmatter_keeps_markdown_dividers_in_the_body() -> None:
    assert strip_frontmatter('---\nname: x\n---\nIntro\n\n---\n\nMore') == 'Intro\n\n---\n\nMore'
    assert strip_frontmatter('Intro\n\n---\n\nMore') == 'Intro\n\n---\n\nMore'
    assert strip_frontmatter('---- not a delimiter ---\nbody') == '---- not a delimiter ---\nbody'
