#!/usr/bin/env python3
#
# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Turn-loop invariants: max_turns, restart input, system= with history, docs=, and resume with prompt=."""

import asyncio
from collections.abc import Awaitable, Callable

import pytest

from genkit import (
    Document,
    Genkit,
    Interrupt,
    Message,
    ModelResponse,
    Part,
    Tool,
    ToolContext,
)
from genkit._ai._agents._client import agent_interrupts_from_message
from genkit._ai._generate import CallScope, augment_with_context, generate_action, restarted_tool_request
from genkit._core._action import ActionRunContext
from genkit._core._error import ErrorReason, GenkitError
from genkit._core._model import GenerateActionOptions, ModelRequest, MultipartToolResponse
from genkit._core._typing import FinishReason, Role, ToolRequest
from genkit.middleware import Middleware, MiddlewareContext, ModelHookParams, ToolHookParams
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


def test_restarted_tool_request_never_carries_replaced_input() -> None:
    """Neither the restart part nor a stale history part can put replacedInput back on history."""
    original = Part(
        tool_request=ToolRequest(name='transfer', input={'days': 7}, ref='t1'),
        metadata={'interrupt': {'reason': 'approve'}, 'replacedInput': {'days': 9}, 'note': 'kept'},
    )
    restart = Part(
        tool_request=ToolRequest(name='transfer', input={'days': 5}, ref='t1'),
        metadata={'resumed': True, 'replacedInput': {'days': 7}},
    )

    ran, metadata, interrupt = restarted_tool_request(original=original, restart=restart)

    assert ran.input == {'days': 5}
    assert metadata == {'note': 'kept'}
    assert interrupt == {'reason': 'approve'}


def test_agent_interrupt_exposes_the_interrupt_payload() -> None:
    message = Message(
        role=Role.MODEL,
        content=[
            Part(
                tool_request=ToolRequest(name='transfer', input={'days': 7}, ref='t1'),
                metadata={'interrupt': {'reason': 'needs approval'}},
            )
        ],
    )

    [intr] = agent_interrupts_from_message(message)

    assert intr.metadata == {'reason': 'needs approval'}


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
async def test_child_flows_get_their_own_copy_of_the_context() -> None:
    """Concurrent child flows writing ctx.context don't clobber each other or the parent."""
    ai = Genkit()
    seen: dict[str, object] = {}

    @ai.flow()
    async def tag_order(table: str, ctx: ActionRunContext) -> str:
        ctx.context['table'] = table
        await asyncio.sleep(0.01)
        seen[table] = ctx.context['table']
        return table

    @ai.flow()
    async def seat_party(n: int, ctx: ActionRunContext) -> list[str]:
        tables = await asyncio.gather(*(tag_order(f't{i}') for i in range(n)))
        seen['parent'] = ctx.context.get('table')
        return list(tables)

    await seat_party(3, context={'auth': {'uid': 'host-1'}})

    assert seen == {'t0': 't0', 't1': 't1', 't2': 't2', 'parent': None}


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
async def test_bare_restart_still_reads_as_resumed() -> None:
    """restart() with no metadata resumes with {}; the tool must still see a resume."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    seen: list[object] = []

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


@pytest.mark.asyncio
async def test_tool_accepts_an_async_callable_instance() -> None:
    """A class with `async def __call__` is an async tool; sync callables still fail."""

    class AllergyCheck:
        async def __call__(self, dish: str) -> str:
            return f'{dish}: nut-free'

    class SyncCheck:
        def __call__(self, dish: str) -> str:
            return dish

    tool = Tool(AllergyCheck(), name='allergy_check')
    assert tool.name == 'allergy_check'
    assert tool.input_schema == {'type': 'string'}
    assert (await tool.run('peanuts')).response == 'peanuts: nut-free'

    with pytest.raises(TypeError, match='must be async'):
        Tool(SyncCheck(), name='sync_check')  # pyright: ignore[reportCallIssue, reportArgumentType]  # pyrefly: ignore[no-matching-overload]


@pytest.mark.asyncio
async def test_middleware_sees_the_context_passed_to_generate() -> None:
    """Genkit.current_context() inside a hook is the generate call's context."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok')]
    seen: list[object] = []

    class AuditTenant(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            seen.append(Genkit.current_context())
            return await next_fn(params, ctx)

    await ai.generate(prompt='hi', context={'tenant': 'bistro-9'}, use=[AuditTenant()])

    assert seen == [{'tenant': 'bistro-9'}]
    assert Genkit.current_context() is None


@pytest.mark.asyncio
async def test_middleware_cleared_context_does_not_leak_to_tool() -> None:
    """Clearing ctx.context in wrap_tool hides caller secrets from the tool."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    tool_seen: list[object] = []

    @ai.tool(name='transfer')
    async def transfer(inp: dict, ctx: ToolContext) -> str:
        tool_seen.append(dict(ctx.context))
        return 'ok'

    class StripContext(Middleware):
        async def wrap_tool(
            self,
            params: ToolHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ToolHookParams, MiddlewareContext], Awaitable[MultipartToolResponse]],
        ) -> MultipartToolResponse:
            ctx.context.clear()
            return await next_fn(params, ctx)

    pm.responses = [_transfer_request(1), _answers('done')]
    await ai.generate(prompt='go', tools=['transfer'], context={'secrets': 'top-secret'}, use=[StripContext()])

    assert tool_seen == [{}]


@pytest.mark.asyncio
async def test_middleware_context_mutations_sync_with_current_context_and_isolate_flow() -> None:
    """Middleware mutations are visible via current_context() and do not mutate parent flow context."""
    ai = Genkit(model='scriptedModel')
    pm, _ = define_scripted_model(ai)
    pm.responses = [_answers('ok')]
    seen_in_second: list[object] = []

    class RewriteTenant(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            ctx.context['tenant'] = 'rewritten'
            return await next_fn(params, ctx)

    class ReadCurrentContext(Middleware):
        async def wrap_model(
            self,
            params: ModelHookParams,
            ctx: MiddlewareContext,
            next_fn: Callable[[ModelHookParams, MiddlewareContext], Awaitable[ModelResponse]],
        ) -> ModelResponse:
            current = Genkit.current_context()
            seen_in_second.append(dict(current) if current is not None else None)
            if current is not None:
                current['leaked'] = True
            return await next_fn(params, ctx)

    @ai.flow()
    async def parent_flow(ctx: ActionRunContext) -> dict[str, object]:
        await ai.generate(prompt='hi', use=[RewriteTenant(), ReadCurrentContext()])
        return dict(ctx.context)

    flow_ctx = await parent_flow(context={'tenant': 'original'})

    assert seen_in_second == [{'tenant': 'rewritten'}]
    assert flow_ctx == {'tenant': 'original'}
