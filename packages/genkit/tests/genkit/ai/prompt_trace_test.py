# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""What the Dev UI trace shows when a prompt runs."""

from collections.abc import Sequence
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient
from pydantic import BaseModel

from genkit import Genkit, Message, ModelResponse, ModelResponseChunk, Part, Role
from genkit._core._reflection import create_reflection_asgi_app
from genkit._core._telemetry._attrs import Attr
from genkit._core._telemetry._http import ActiveSpan
from genkit.model import ModelContext, ModelRequest

MODEL = 'fake'


async def _fake_model(request: ModelRequest, ctx: ModelContext) -> ModelResponse:
    user_text = next((m.text for m in reversed(request.messages) if m.role == Role.USER), '')
    if user_text.endswith('boom'):
        raise Exception('fake model failure')
    last = request.messages[-1]
    if any(p.tool_response is not None for p in last.content):
        content = [Part.from_text('done')]
    elif request.tools:
        content = [Part.from_tool_request(name=request.tools[0].name, input='Paris', ref='t1')]
    else:
        if ctx.is_streaming:
            for piece in ['echo: ', user_text]:
                ctx.send_chunk(ModelResponseChunk(role=Role.MODEL, index=0, content=[Part.from_text(piece)]))
        content = [Part.from_text(f'echo: {user_text}')]
    return ModelResponse(message=Message(role=Role.MODEL, content=content), finish_reason='stop')


def _app(prompt_dir: Path | None = None) -> Genkit:
    ai = Genkit(prompt_dir=prompt_dir) if prompt_dir is not None else Genkit()
    ai.define_model(name=MODEL, fn=_fake_model)
    return ai


def _tree(spans: Sequence[ActiveSpan]) -> list[tuple[str, str | None, object]]:
    """(name, parent name, genkit:type) per span, in start order."""
    by_id = {s.span_id: s for s in spans}
    return [
        (
            s.name,
            by_id[s.parent_span_id].name if s.parent_span_id in by_id else None,
            s.attributes.get(Attr.TYPE),
        )
        for s in spans
    ]


def _descends_from(span: ActiveSpan, ancestor: ActiveSpan, spans: Sequence[ActiveSpan]) -> bool:
    by_id = {s.span_id: s for s in spans}
    parent = by_id.get(span.parent_span_id or '')
    while parent is not None:
        if parent.span_id == ancestor.span_id:
            return True
        parent = by_id.get(parent.parent_span_id or '')
    return False


class Greeting(BaseModel):
    name: str


PROMPT_TREE = [
    ('prompt_flow', None, 'action'),
    ('greeting_prompt', 'prompt_flow', 'dotprompt'),
    ('render', 'greeting_prompt', 'promptTemplate'),
    ('generate', 'greeting_prompt', 'util'),
    (MODEL, 'generate', 'action'),
]


@pytest.mark.asyncio
async def test_calling_a_defined_prompt_in_a_flow_shows_prompt_with_render_and_generate(exporter) -> None:
    """`await greeting(input={'name': 'Ada'})` in a flow shows greeting_prompt with render and generate under it."""
    ai = _app()
    greeting = ai.define_prompt(name='greeting_prompt', model=MODEL, input_schema=Greeting, prompt='Say hi to {{name}}')

    @ai.flow(name='prompt_flow')
    async def prompt_flow(name: str) -> str:
        return (await greeting(input={'name': name})).text

    assert await prompt_flow('Ada') == 'echo: Say hi to Ada'

    spans = exporter.get_finished_spans()
    assert _tree(spans) == PROMPT_TREE
    prompt_span = next(s for s in spans if s.name == 'greeting_prompt')
    assert prompt_span.attributes[Attr.INPUT] == '{"name": "Ada"}'
    assert prompt_span.attributes[Attr.STATE] == 'success'
    assert 'echo: Say hi to Ada' in str(prompt_span.attributes[Attr.OUTPUT])
    assert Attr.SUBTYPE not in prompt_span.attributes
    render_span = next(s for s in spans if s.name == 'render')
    assert render_span.attributes['genkit:metadata:promptName'] == 'greeting_prompt'
    assert render_span.attributes[Attr.INPUT] == '{"name": "Ada"}'
    assert 'Say hi to Ada' in str(render_span.attributes[Attr.OUTPUT])


@pytest.mark.asyncio
async def test_calling_a_prompt_file_in_a_flow_shows_prompt_with_render_and_generate(exporter, tmp_path: Path) -> None:
    """`await ai.prompt('file_prompt')(input=...)` in a flow shows file_prompt with render and generate under it."""
    (tmp_path / 'file_prompt.prompt').write_text(
        f'---\nmodel: {MODEL}\ninput:\n  schema:\n    name: string\n---\nHello from file to {{{{name}}}}\n'
    )
    ai = _app(prompt_dir=tmp_path)

    @ai.flow(name='file_prompt_flow')
    async def file_prompt_flow(name: str) -> str:
        return (await ai.prompt('file_prompt')(input={'name': name})).text

    assert await file_prompt_flow('Ada') == 'echo: Hello from file to Ada'

    spans = exporter.get_finished_spans()
    assert _tree(spans) == [
        ('file_prompt_flow', None, 'action'),
        ('file_prompt', 'file_prompt_flow', 'dotprompt'),
        ('render', 'file_prompt', 'promptTemplate'),
        ('generate', 'file_prompt', 'util'),
        (MODEL, 'generate', 'action'),
    ]
    render_span = next(s for s in spans if s.name == 'render')
    assert render_span.attributes['genkit:metadata:promptName'] == 'file_prompt'


@pytest.mark.asyncio
async def test_streaming_a_prompt_shows_the_same_prompt_tree(exporter) -> None:
    """`greeting.stream(input=...)` in a flow shows the same greeting_prompt tree as awaiting it."""
    ai = _app()
    greeting = ai.define_prompt(name='greeting_prompt', model=MODEL, input_schema=Greeting, prompt='Say hi to {{name}}')

    @ai.flow(name='prompt_flow')
    async def prompt_flow(name: str) -> str:
        result = greeting.stream(input={'name': name})
        chunks = [chunk.text async for chunk in result.stream]
        assert ''.join(chunks) == 'echo: Say hi to Ada'
        return (await result.response).text

    assert await prompt_flow('Ada') == 'echo: Say hi to Ada'

    assert _tree(exporter.get_finished_spans()) == PROMPT_TREE


@pytest.mark.asyncio
async def test_prompt_with_tools_nests_the_tool_loop_under_the_prompt(exporter) -> None:
    """A prompt whose model calls a tool shows the tool and every model call inside the prompt span."""
    ai = _app()

    @ai.tool()
    async def get_weather(city: str) -> str:
        return f'sunny in {city}'

    greeting = ai.define_prompt(
        name='greeting_prompt', model=MODEL, input_schema=Greeting, prompt='Weather for {{name}}', tools=['get_weather']
    )

    @ai.flow(name='prompt_flow')
    async def prompt_flow(name: str) -> str:
        return (await greeting(input={'name': name})).text

    assert await prompt_flow('Ada') == 'done'

    spans = exporter.get_finished_spans()
    prompt_span = next(s for s in spans if s.name == 'greeting_prompt')
    tool_spans = [s for s in spans if s.name == 'get_weather']
    model_spans = [s for s in spans if s.name == MODEL]
    assert len(tool_spans) == 1
    assert len(model_spans) == 2
    assert all(_descends_from(s, prompt_span, spans) for s in [*tool_spans, *model_spans])
    assert ('render', 'greeting_prompt', 'promptTemplate') in _tree(spans)


@pytest.mark.asyncio
async def test_prompt_whose_model_fails_shows_the_cause_on_the_prompt_span(exporter) -> None:
    """A prompt whose model raises marks the prompt span as failed and still returns the failed response."""
    ai = _app()
    greeting = ai.define_prompt(name='greeting_prompt', model=MODEL, input_schema=Greeting, prompt='Say hi to {{name}}')

    @ai.flow(name='prompt_flow')
    async def prompt_flow(name: str) -> str:
        response = await greeting(input={'name': name})
        return str(response.finish_reason)

    assert await prompt_flow('boom') == 'failed'

    spans = exporter.get_finished_spans()
    assert _tree(spans) == PROMPT_TREE
    prompt_span = next(s for s in spans if s.name == 'greeting_prompt')
    assert prompt_span.attributes[Attr.STATE] == 'error'
    assert prompt_span.status_code == 2
    assert '"finishReason":"failed"' in str(prompt_span.attributes[Attr.OUTPUT])


@pytest.mark.asyncio
async def test_rendering_a_prompt_from_the_dev_ui_records_a_render_span(exporter) -> None:
    """Running greeting_prompt from the Dev UI shows a render span under the prompt action."""
    ai = _app()
    ai.define_prompt(name='greeting_prompt', model=MODEL, input_schema=Greeting, prompt='Say hi to {{name}}')

    client = AsyncClient(transport=ASGITransport(app=create_reflection_asgi_app(ai._registry)), base_url='http://test')
    try:
        for key in ['/executable-prompt/greeting_prompt', '/prompt/greeting_prompt']:
            exporter.clear()
            response = await client.post('/api/runAction', json={'key': key, 'input': {'name': 'Ada'}})

            assert response.status_code == 200
            assert 'Say hi to Ada' in response.text
            spans = exporter.get_finished_spans()
            assert _tree(spans) == [
                ('greeting_prompt', None, 'action'),
                ('render', 'greeting_prompt', 'promptTemplate'),
            ]
    finally:
        await client.aclose()


@pytest.mark.asyncio
async def test_prompt_variant_names_the_prompt_and_render_spans_with_the_variant(exporter) -> None:
    """A variant prompt uses `name.variant` for both the prompt span and render's promptName, like JS."""
    ai = _app()
    formal = ai.define_prompt(
        name='greeting_prompt', variant='formal', model=MODEL, input_schema=Greeting, prompt='Good day, {{name}}'
    )

    assert (await formal(input={'name': 'Ada'})).text == 'echo: Good day, Ada'

    spans = exporter.get_finished_spans()
    assert ('greeting_prompt.formal', None, 'dotprompt') in _tree(spans)
    render_span = next(s for s in spans if s.name == 'render')
    assert render_span.attributes['genkit:metadata:promptName'] == 'greeting_prompt.formal'
