#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Turn-loop invariants: max_turns, restart input, system= with history, docs=, and resume with prompt=."""

import numpy as np
import pytest

from genkit import (
    Document,
    Genkit,
    Interrupt,
    Message,
    ModelResponse,
    Part,
    ToolContext,
)
from genkit._ai._generate import CallScope, augment_with_context, generate_action, resolved_interrupt_part
from genkit._core import _tool as tool_mod
from genkit._core._error import ErrorReason, GenkitError
from genkit._core._model import GenerateActionOptions, ModelRequest, reject_unanswered_interrupts
from genkit._core._typing import FinishReason, Role, ToolRequest
from genkit.testing import ScriptedModel, define_scripted_model


def _answers(text: str) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(role=Role.MODEL, content=[Part.from_text(text)]),
    )


def _transfer_request(amount: int) -> ModelResponse:
    return ModelResponse(
        finish_reason=FinishReason.STOP,
        message=Message(
            role=Role.MODEL,
            content=[Part(tool_request=ToolRequest(name='transfer', input={'days': amount}, ref='t1'))],
        ),
    )


def _hitl_app() -> tuple[Genkit, ScriptedModel, list[object]]:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    ran_with: list[object] = []

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolContext) -> str:
        if not ctx.is_resumed():
            raise Interrupt({'reason': 'needs approval', 'days': inp['days']})
        ran_with.append(inp)
        return f'booked {inp["days"]} days'

    return ai, pm, ran_with


@pytest.mark.asyncio
async def test_restart_replace_input_is_the_input_history_keeps() -> None:
    """The model's next turn and the audit trail see the approved input, not the original."""
    ai, pm, ran_with = _hitl_app()
    pm.responses = [_transfer_request(7), _answers('booked')]

    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    assert paused.finish_reason == FinishReason.INTERRUPTED
    intr = paused.interrupts[0]
    assert intr.metadata is not None
    assert intr.metadata['interrupt'] == {'reason': 'needs approval', 'days': 7}

    done = await ai.generate(
        messages=paused.messages,
        tools=['transfer'],
        resume_restart=intr.restart(replace_input={'days': 5}),
    )

    assert ran_with == [{'days': 5}]
    assert done.text == 'booked'
    model_turn = next(m for m in done.messages if m.role == Role.MODEL and m.content[0].tool_request is not None)
    part = model_turn.content[0]
    assert part.tool_request is not None
    assert part.tool_request.input == {'days': 5}
    assert part.metadata is not None
    # On a tool request in history, replacedInput would read as a restart.
    assert 'replacedInput' not in part.metadata
    assert part.metadata['resolvedInterrupt'] == {'reason': 'needs approval', 'days': 7}
    assert 'interrupt' not in part.metadata
    assert 'resumed' not in part.metadata
    # The model reads the approved input on the turn after the restart.
    assert pm.last_request is not None
    sent = next(m for m in pm.last_request.messages if m.role == Role.MODEL)
    assert sent.content[0].tool_request is not None
    assert sent.content[0].tool_request.input == {'days': 5}


@pytest.mark.asyncio
async def test_restart_without_replace_input_keeps_the_model_input() -> None:
    ai, pm, ran_with = _hitl_app()
    pm.responses = [_transfer_request(7), _answers('booked')]

    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    done = await ai.generate(
        messages=paused.messages,
        tools=['transfer'],
        resume_restart=paused.interrupts[0].restart(),
    )

    assert ran_with == [{'days': 7}]
    model_turn = next(m for m in done.messages if m.role == Role.MODEL and m.content[0].tool_request is not None)
    assert model_turn.content[0].tool_request is not None
    assert model_turn.content[0].tool_request.input == {'days': 7}
    assert 'replacedInput' not in (model_turn.content[0].metadata or {})


@pytest.mark.asyncio
async def test_restart_that_pauses_again_keeps_the_replaced_input() -> None:
    """A second approval step sees the input the restart ran with."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolContext) -> str:
        raise Interrupt({'reason': 'second approver', 'days': inp['days']})

    pm.responses = [_transfer_request(7)]
    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    again = await ai.generate(
        messages=paused.messages,
        tools=['transfer'],
        resume_restart=paused.interrupts[0].restart(replace_input={'days': 5}),
    )

    assert again.finish_reason == FinishReason.INTERRUPTED
    part = again.interrupts[0]
    assert part.tool_request is not None
    assert part.tool_request.input == {'days': 5}
    assert part.metadata is not None
    assert part.metadata['interrupt'] == {'reason': 'second approver', 'days': 5}
    assert 'replacedInput' not in part.metadata


def test_resolved_interrupt_part_strips_interrupt_and_replaced_input() -> None:
    """History keeps the approved input and resolvedInterrupt, never interrupt or replacedInput."""
    original = Part(
        tool_request=ToolRequest(name='transfer', input={'days': 7}, ref='t1'),
        metadata={'interrupt': {'reason': 'approve'}, 'replacedInput': {'days': 9}, 'note': 'kept'},
    )

    resolved = resolved_interrupt_part(original=original, tool_input={'days': 5})

    assert resolved.tool_request is not None
    assert resolved.tool_request.input == {'days': 5}
    assert resolved.metadata == {'note': 'kept', 'resolvedInterrupt': {'reason': 'approve'}}


@pytest.mark.asyncio
async def test_system_with_saved_history_goes_first_and_keeps_history() -> None:
    """system= is prepended; the saved chat's own system message is sent as stored."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('Bonjour'), _answers('Bill: $42')]

    first = await ai.generate(system='You are the host.', prompt='Hi')
    await ai.generate(system='You are the cashier.', messages=first.messages, prompt='Check please')

    assert pm.last_request is not None
    sent = pm.last_request.messages
    assert [m.role for m in sent] == [Role.SYSTEM, Role.SYSTEM, Role.USER, Role.MODEL, Role.USER]
    assert sent[0].text == 'You are the cashier.'
    assert sent[1].text == 'You are the host.'


@pytest.mark.asyncio
async def test_empty_system_adds_nothing_and_keeps_history() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('Bonjour'), _answers('ok')]

    first = await ai.generate(system='You are the host.', prompt='Hi')
    await ai.generate(system='', messages=first.messages, prompt='Thanks')

    assert pm.last_request is not None
    sent = pm.last_request.messages
    assert [m.role for m in sent] == [Role.SYSTEM, Role.USER, Role.MODEL, Role.USER]
    assert sent[0].text == 'You are the host.'


@pytest.mark.asyncio
async def test_omitted_system_keeps_the_saved_persona() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('Bonjour'), _answers('ok')]

    first = await ai.generate(system='You are the host.', prompt='Hi')
    await ai.generate(messages=first.messages, prompt='Thanks')

    assert pm.last_request is not None
    assert pm.last_request.messages[0].role == Role.SYSTEM
    assert pm.last_request.messages[0].text == 'You are the host.'


def test_docs_context_is_text_only_and_leaves_metadata_out() -> None:
    """Citation line is `- [ref]: text`; metadata and media parts stay out of the prompt."""
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('Which dishes are vegan?')])],
        docs=[
            Document(
                content=[
                    Part.from_text('Lentil soup'),
                    Part.from_media('gs://menu/soup.png', content_type='image/png'),
                ],
                metadata={'id': 'menu-7', 'acl': ['user-7'], 'embedding': [0.1, 0.2]},
            )
        ],
    )

    augmented = augment_with_context(request)

    user = augmented.messages[-1]
    context = [p for p in user.content if (p.metadata or {}).get('purpose') == 'context']
    assert len(context) == 1
    assert context[0].text is not None
    assert '\n- [menu-7]: Lentil soup\n' in context[0].text
    assert 'acl' not in context[0].text
    assert 'embedding' not in context[0].text
    assert 'user-7' not in context[0].text
    assert all(p.media is None for p in user.content)


def test_docs_context_without_extra_metadata_is_unchanged() -> None:
    request = ModelRequest(
        messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
        docs=[Document.from_text('Lentil soup', metadata={'ref': 'r1'})],
    )

    augmented = augment_with_context(request)

    text = augmented.messages[-1].content[-1].text
    assert text is not None
    assert '- [r1]: Lentil soup\n' in text


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [True, False, float('nan'), float('inf'), 2.5])
async def test_max_turns_rejects_non_whole_numbers(bad: object) -> None:
    """max_turns=True used to run as 1; nan leaked a raw ValidationError."""
    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError) as raised:
        await ai.generate(prompt='hi', max_turns=bad)  # type: ignore[arg-type]

    assert raised.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_max_turns_accepts_a_whole_float_from_dotprompt() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok')]

    response = await generate_action(
        CallScope(ai),
        GenerateActionOptions(
            model='scriptedModel',
            messages=[Message(role=Role.USER, content=[Part.from_text('hi')])],
            max_turns=3.0,
        ),
    )

    assert response.text == 'ok'


@pytest.mark.asyncio
async def test_prompt_with_resume_fails_before_the_model_runs() -> None:
    """Same boxed FAILED_PRECONDITION as any resume whose last message isn't a tool request."""
    ai, pm, ran_with = _hitl_app()
    pm.responses = [_transfer_request(7)]
    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    calls = pm.request_count

    response = await ai.generate(
        prompt='also add breakfast',
        messages=paused.messages,
        tools=['transfer'],
        resume_restart=paused.interrupts[0].restart(),
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is ErrorReason.INVALID_RESUME
    assert response.error.message.startswith(
        "Cannot 'resume' generation unless the previous message is a model message"
    )
    assert 'prompt=' in response.error.message
    assert pm.request_count == calls
    assert ran_with == []


@pytest.mark.asyncio
async def test_resume_without_a_paused_tool_request_does_not_mention_prompt() -> None:
    ai, pm, _ = _hitl_app()

    response = await ai.generate(
        messages=[Message(role=Role.USER, content=[Part.from_text('book 7 days')])],
        tools=['transfer'],
        resume_restart=Part(tool_request=ToolRequest(name='transfer', input={}, ref='t1'), metadata={'resumed': True}),
    )

    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is ErrorReason.INVALID_RESUME
    assert 'prompt=' not in response.error.message
    assert pm.request_count == 0


@pytest.mark.asyncio
async def test_bare_restart_still_reads_as_resumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """restart() with no metadata resumes with {}; the tool span still records resumed."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    seen: list[object] = []
    span_attrs: list[dict[str, object]] = []
    orig_set_attrs = tool_mod.set_custom_metadata_attributes

    def _record_attrs(attrs: dict[str, object]) -> None:
        span_attrs.append(dict(attrs))
        orig_set_attrs(attrs)

    monkeypatch.setattr(tool_mod, 'set_custom_metadata_attributes', _record_attrs)

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolContext) -> str:
        if not ctx.is_resumed():
            raise Interrupt({'reason': 'approve'})
        seen.append(ctx.resumed_metadata)
        return 'ok'

    pm.responses = [_transfer_request(7), _answers('done')]
    paused = await ai.generate(prompt='book', tools=['transfer'])
    await ai.generate(messages=paused.messages, tools=['transfer'], resume_restart=paused.interrupts[0].restart())

    assert len(seen) == 1
    assert {'resumed': {}} in span_attrs


@pytest.mark.asyncio
async def test_resume_with_user_turn_after_paused_tool_does_not_mention_prompt() -> None:
    """History ending in a user turn after a paused tool request must not blame prompt=."""
    ai, pm, _ = _hitl_app()
    pm.responses = [_transfer_request(7)]
    paused = await ai.generate(prompt='book 7 days', tools=['transfer'])
    calls = pm.request_count

    response = await ai.generate(
        messages=[*paused.messages, Message(role=Role.USER, content=[Part.from_text('also add breakfast')])],
        tools=['transfer'],
        resume_restart=paused.interrupts[0].restart(),
    )

    assert response.finish_reason == FinishReason.FAILED
    assert response.error is not None
    assert response.error.status == 'FAILED_PRECONDITION'
    assert response.error.reason is ErrorReason.INVALID_RESUME
    assert 'prompt=' not in response.error.message
    assert pm.request_count == calls


def test_reject_unanswered_interrupts_sets_invalid_resume_reason() -> None:
    paused = Part(
        tool_request=ToolRequest(name='transfer', input={'days': 7}, ref='t1'),
        metadata={'interrupt': {'reason': 'approve'}},
    )

    with pytest.raises(GenkitError) as restart_err:
        reject_unanswered_interrupts(restart=[paused])
    assert restart_err.value.status == 'INVALID_ARGUMENT'
    assert restart_err.value.reason is ErrorReason.INVALID_RESUME

    with pytest.raises(GenkitError) as respond_err:
        reject_unanswered_interrupts(respond=[paused])
    assert respond_err.value.status == 'INVALID_ARGUMENT'
    assert respond_err.value.reason is ErrorReason.INVALID_RESUME


@pytest.mark.asyncio
@pytest.mark.parametrize('bad', [True, False, 2.5])
async def test_max_turns_rejects_bool_and_fraction_across_entry_points(bad: object) -> None:
    from genkit._ai._agents._base import define_agent

    ai = Genkit(model='scriptedModel')
    define_scripted_model(ai)

    with pytest.raises(GenkitError) as gen_err:
        ai.generate_stream(prompt='hi', max_turns=bad)  # type: ignore[arg-type]
    assert gen_err.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError) as prompt_err:
        ai.define_prompt(name=f'p_{bad}', prompt='hi', max_turns=bad)  # type: ignore[arg-type]
    assert prompt_err.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError) as agent_err:
        define_agent(ai, name=f'a_{bad}', max_turns=bad)  # type: ignore[arg-type]
    assert agent_err.value.status == 'INVALID_ARGUMENT'

    with pytest.raises(GenkitError) as opts_err:
        GenerateActionOptions(messages=[], max_turns=bad)  # type: ignore[arg-type]
    assert opts_err.value.status == 'INVALID_ARGUMENT'


@pytest.mark.asyncio
async def test_max_turns_accepts_numpy_integral() -> None:
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok'), _answers('ok')]

    res = await ai.generate(prompt='hi', max_turns=np.int64(3))  # type: ignore[arg-type]
    assert res.text == 'ok'

    prompt = ai.define_prompt(name='np_turns', prompt='hi', max_turns=np.int64(3))  # type: ignore[arg-type]
    prompt_res = await prompt()
    assert prompt_res.text == 'ok'
