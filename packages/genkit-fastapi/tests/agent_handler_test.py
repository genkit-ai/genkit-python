# Copyright 2026 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""Tests for serve_agent in genkit_fastapi.exp."""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import StreamingResponse
from fastapi.testclient import TestClient
from genkit_fastapi import handle_genkit_request
from genkit_fastapi.exp import serve_agent

from genkit import (
    ErrorReason,
    FinishReason,
    GenkitError,
    Message,
    ModelResponse,
    ModelResponseChunk,
    Part,
    PublicError,
    Role,
)

# error_from_http is private: this test pins the client-side HTTP error mapping.
from genkit._ai._agents._client import error_from_http  # noqa: TID251
from genkit.exp import Genkit
from genkit.exp.agent import (
    AgentContext,
    AgentInit,
    AgentInput,
    AgentResult,
    InMemorySessionStore,
    SessionRunner,
    SnapshotStatus,
    TurnContext,
    TurnResult,
)
from genkit.plugin_api import provider_error
from genkit.testing import define_scripted_model
from genkit.web import to_sse


def build_agent(name: str, *, server_managed: bool = True) -> Any:
    """A prompt agent whose model replies with a fixed line; server-backed unless told otherwise."""
    ai = Genkit()
    define_scripted_model(
        ai,
        name='scriptedModel',
        responses=[
            ModelResponse(
                finish_reason=FinishReason.STOP,
                message=Message(role=Role.MODEL, content=[Part.from_text('Hi there!')]),
            )
        ],
        chunks=[[ModelResponseChunk(role=Role.MODEL, content=[Part.from_text('Hi there!')])]],
    )
    ai.define_prompt(name=name, model='scriptedModel', system='You echo things.')
    return ai.define_prompt_agent(name=name, store=InMemorySessionStore() if server_managed else None)


def sse_events(text: str) -> list[dict[str, Any]]:
    records = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('data: '):
            records.append(json.loads(line[6:].strip()))
        elif line.startswith('error: '):
            records.append(json.loads(line[7:].strip()))
    return records


def client(agent: Any, **kwargs: Any) -> TestClient:
    """Mount ``agent`` under ``/api`` and return a test client."""
    app = FastAPI()
    app.include_router(serve_agent(agent, base_path='/chat', **kwargs), prefix='/api')
    return TestClient(app)


def test_turn_streams_sse_and_final_result() -> None:
    """A turn returns SSE events and a final {"result": <AgentOutput>}."""
    client_obj = client(build_agent('echoAgent'))

    response = client_obj.post('/api/chat?stream=true', json={'message': 'Hi'})

    assert response.status_code == 200
    assert response.headers['content-type'].startswith('text/event-stream')
    records = sse_events(response.text)
    assert 'result' in records[-1]
    # The reply text lands in the settled AgentOutput.
    assert 'Hi there!' in json.dumps(records[-1]['result'])


def test_base_path_defaults_to_agent_name() -> None:
    """Omitting base_path mounts the turn route at /<agent name>."""
    app = FastAPI()
    app.include_router(serve_agent(build_agent('weatherAgent')), prefix='/api')  # no base_path
    client_obj = TestClient(app)

    response = client_obj.post('/api/weatherAgent', json={'message': 'Hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(response.json()['result'])


def test_turn_data_envelope_matches_shorthand() -> None:
    """The {"data": ..., "init": ...} wire shape works the same as the shorthand."""
    client_obj = client(build_agent('wireAgent'))

    body = {'data': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}, 'init': {}}
    response = client_obj.post('/api/chat', json=body)

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(response.json()['result'])


def test_agent_input_key_body_is_400() -> None:
    """POST {"input": "hi"} to an agent route is a 400 telling the caller to wrap in data."""
    response = client(build_agent('inputKeyAgent')).post('/api/chat', json={'input': 'hi'})

    assert response.status_code == 400
    assert response.json() == {
        'message': 'Flow request must be wrapped in {"data": ...}',
        'status': 'INVALID_ARGUMENT',
    }


def test_agent_non_object_body_is_400_with_flow_wording() -> None:
    """POST a bare JSON string to an agent route gets the same 400 as a flow route."""
    response = client(build_agent('bareBodyAgent')).post('/api/chat', json='hi')

    assert response.status_code == 400
    assert response.json() == {
        'message': 'Flow request must be wrapped in {"data": ...}',
        'status': 'INVALID_ARGUMENT',
    }


def test_get_snapshot_missing_returns_404() -> None:
    """getSnapshot for an unknown snapshot id returns 404."""
    client_obj = client(build_agent('snapAgent'))

    response = client_obj.post('/api/chat/getSnapshot', json={'snapshotId': 'does-not-exist'})

    assert response.status_code == 404


def test_context_dependency_gates_the_turn() -> None:
    """A context_dependency that raises stops the turn before it streams."""

    async def deny() -> dict[str, object]:
        raise HTTPException(status_code=401, detail='no token')

    client_obj = client(build_agent('depAuthAgent'), context_dependency=deny)

    response = client_obj.post('/api/chat', json={'message': 'Hi'})

    assert response.status_code == 401


def test_context_dependency_allows_the_turn() -> None:
    """A resolved context_dependency lets the turn run and stream normally."""

    async def allow() -> dict[str, object]:
        return {'uid': 'user-123'}

    client_obj = client(build_agent('depOkAgent'), context_dependency=allow)

    response = client_obj.post('/api/chat?stream=true', json={'message': 'Hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(sse_events(response.text)[-1]['result'])


def test_serve_agent_message_body_starts_a_turn() -> None:
    """POST {"message": "hi"} to serve_agent starts a turn."""
    client_obj = client(build_agent('msgAgent'))

    response = client_obj.post('/api/chat', json={'message': 'hi'})

    assert response.status_code == 200
    assert 'Hi there!' in json.dumps(response.json()['result'])


def test_serve_agent_session_id_query_param_continues_session() -> None:
    """POST /chat?session_id=s1 runs the turn in session s1."""
    client_obj = client(build_agent('sessionAgent'))

    response = client_obj.post('/api/chat?session_id=s1', json={'message': 'Hi'})

    assert response.status_code == 200
    result = response.json()['result']
    assert result['sessionId'] == 's1'
    assert 'Hi there!' in json.dumps(result)

    snap = client_obj.post('/api/chat/getSnapshot', json={'sessionId': 's1'})
    assert snap.status_code == 200
    assert snap.json()['result']['sessionId'] == 's1'


def test_handle_genkit_request_agent_route_with_data_envelope_runs_turn() -> None:
    """A custom route that calls handle_genkit_request with {"data": ...} runs a turn."""
    agent = build_agent('handRolledAgent')
    app = FastAPI()

    @app.post('/custom')
    async def custom(request: Request) -> Response:
        # A real app would build this context and init from its own Depends params.
        return await handle_genkit_request(
            request,
            action=agent,
            context={'uid': 'user-123'},
            init=AgentInit(session_id='session-789'),
        )

    client_obj = TestClient(app)

    response = client_obj.post(
        '/custom',
        json={'data': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}},
    )

    assert response.status_code == 200
    result = response.json()['result']
    assert result['sessionId'] == 'session-789'
    assert 'Hi there!' in json.dumps(result)


@pytest.mark.parametrize(
    'path, body, message',
    [
        pytest.param('/api/chat', {'foo': 'bar'}, 'Flow request must be wrapped in {"data": ...}', id='turn'),
        pytest.param(
            '/api/chat/getSnapshot',
            {'snapshotId': 's1', 'sessionId': 'x1'},
            "getSnapshot requires exactly one of 'snapshotId' (or 'snapshot_id') or 'sessionId' (or 'session_id').",
            id='get-snapshot',
        ),
        pytest.param(
            '/api/chat/abort', {'data': {}}, "abort requires 'snapshotId' (or 'snapshot_id') in input.", id='abort'
        ),
    ],
)
def test_serve_agent_bad_input_returns_400_with_its_message(path: str, body: dict[str, Any], message: str) -> None:
    """Agent-route input errors are fixed adapter text, so the caller sees what to fix."""
    response = client(build_agent('badInputAgent')).post(path, json=body)

    assert response.status_code == 400
    assert response.json() == {'message': message, 'status': 'INVALID_ARGUMENT'}


@pytest.mark.parametrize(
    'server_managed, init, message, reason',
    [
        pytest.param(
            True,
            {'state': {'custom': {'table': 4}}},
            "Cannot send 'state' to agent 'initAgent': this agent uses a server-managed store. "
            "Send 'snapshotId' or 'sessionId' instead.",
            None,
            id='state-to-server-managed',
        ),
        pytest.param(
            False,
            {'snapshotId': 'snap-1'},
            "Cannot use 'snapshotId' with agent 'initAgent': this agent has no store configured "
            "(client-managed state). Send 'state' instead.",
            ErrorReason.SESSION_STORE_NOT_CONFIGURED,
            id='snapshot-id-without-store',
        ),
    ],
)
def test_serve_agent_init_mismatch_returns_agent_init_error(
    server_managed: bool, init: dict[str, Any], message: str, reason: ErrorReason | None
) -> None:
    """AgentInitError is a PublicError, so a remote client sees the status, message, and reason it would in-process."""
    response = client(build_agent('initAgent', server_managed=server_managed)).post(
        '/api/chat', json={'data': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}, 'init': init}
    )

    assert response.status_code == 400
    expected: dict[str, Any] = {'message': message, 'status': 'FAILED_PRECONDITION'}
    if reason is not None:
        expected['details'] = {'reason': reason.value}
    assert response.json() == expected

    err = error_from_http(status_code=response.status_code, body=response.text)
    assert err.status == 'FAILED_PRECONDITION'
    assert err.reason is reason


def test_serve_agent_stream_failure_log_names_the_agent(caplog: pytest.LogCaptureFixture) -> None:
    """A streamed agent failure logs the agent's action name via to_sse."""
    with caplog.at_level(logging.WARNING):
        response = client(build_agent('orderAgent', server_managed=True)).post(
            '/api/chat?stream=true',
            json={
                'data': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}},
                'init': {'state': {'custom': {'table': 4}}},
            },
        )

    msg = (
        "Cannot send 'state' to agent 'orderAgent': this agent uses a server-managed store. "
        "Send 'snapshotId' or 'sessionId' instead."
    )
    assert sse_events(response.text) == [{'error': {'message': msg, 'status': 'FAILED_PRECONDITION'}}]
    assert [(r.levelname, r.getMessage(), r.exc_info is not None) for r in caplog.records] == [
        ('WARNING', f"served action 'orderAgent' stream failed: FAILED_PRECONDITION: {msg}", False),
    ]


_UUID = re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')


def frames(text: str) -> list[str]:
    """SSE data lines with session / snapshot uuids masked, so frames from separate runs compare whole."""
    return [_UUID.sub('<id>', line) for line in text.splitlines() if line.startswith('data: ')]


def test_hand_rolled_routes_send_the_same_frames_as_serve_agent() -> None:
    """handle_genkit_request and genkit.web.to_sse(turn) send serve_agent's frames, payloads included."""
    served = client(build_agent('servedFrames')).post('/api/chat?stream=true', json={'message': 'Hi'})

    via_request = build_agent('requestFrames')
    via_to_sse = build_agent('toSseFrames')
    app = FastAPI()

    @app.post('/request')
    async def request_route(request: Request) -> Response:
        return await handle_genkit_request(request, action=via_request)

    @app.post('/to_sse')
    async def to_sse_route() -> StreamingResponse:
        return StreamingResponse(to_sse(via_to_sse.chat().send_stream('Hi')), media_type='text/event-stream')

    hand = TestClient(app)
    wire_body = {'data': {'message': {'role': 'user', 'content': [{'text': 'Hi'}]}}}

    expected = [
        'data: {"message":{"modelChunk":{"role":"model","index":0.0,"content":[{"text":"Hi there!"}]}}}',
        'data: {"message":{"turnEnd":{"snapshotId":"<id>","finishReason":"stop"}}}',
        'data: {"result":{"sessionId":"<id>","snapshotId":"<id>",'
        '"message":{"role":"model","content":[{"text":"Hi there!"}]},"artifacts":[],"finishReason":"stop"}}',
    ]
    assert frames(served.text) == expected
    assert frames(hand.post('/request?stream=true', json=wire_body).text) == expected
    assert frames(hand.post('/to_sse').text) == expected


class _ProviderHttpError(Exception):
    """Stands in for an SDK's HTTP error whose text carries the request URL."""


def _failing_agent(name: str, exc: Exception) -> Any:
    """A server-managed custom agent whose every turn raises ``exc``."""
    ai = Genkit()

    async def fn(session_runner: SessionRunner, _ctx: AgentContext) -> AgentResult:
        async def turn(_inp: AgentInput, _tc: TurnContext) -> TurnResult | None:
            raise exc

        await session_runner.run(turn)
        return await session_runner.result()

    return ai.define_custom_agent(name=name, fn=fn, store=InMemorySessionStore())


@pytest.mark.parametrize(
    ('exc', 'expected'),
    [
        (ValueError('db password=hunter2'), {'status': 'INTERNAL', 'message': 'Internal Error'}),
        (
            PublicError('PERMISSION_DENIED', 'table 12 is reserved'),
            {'status': 'PERMISSION_DENIED', 'message': 'table 12 is reserved'},
        ),
        (
            GenkitError(status='NOT_FOUND', message='no row in tenant_secrets for key=hunter2'),
            {'status': 'NOT_FOUND', 'message': 'Not Found'},
        ),
        (
            provider_error(_ProviderHttpError('429 https://api.example.com/v1?key=hunter2'), http_status=429),
            {'status': 'RESOURCE_EXHAUSTED', 'message': 'Resource Exhausted'},
        ),
    ],
    ids=['foreign', 'public', 'genkit_not_found', 'provider_429'],
)
@pytest.mark.parametrize('stream', [False, True], ids=['json', 'sse'])
def test_failed_turn_sends_only_public_error_text(exc: Exception, expected: dict[str, str], stream: bool) -> None:
    """A failed turn is a 200; its error follows error_body's rule, so only a PublicError's text goes out."""
    response = client(_failing_agent('ordersDb', exc)).post(
        f'/api/chat?stream={str(stream).lower()}', json={'message': 'refund A-17'}
    )

    assert response.status_code == 200
    result = sse_events(response.text)[-1]['result'] if stream else response.json()['result']
    assert result['finishReason'] == 'failed'
    assert result['error'] == expected
    assert 'hunter2' not in response.text


@pytest.mark.asyncio
async def test_failed_turn_keeps_error_text_in_process_and_in_the_store() -> None:
    """Redaction is at the wire: res.error and the stored snapshot keep the exception's text."""
    agent = _failing_agent('ordersDb', RuntimeError('pos sync failed: dsn=pg://admin@10.0.0.4'))

    # 1. Detach so the background finalize writes the failure into the store.
    out = await agent.run(
        input=AgentInput(message=Message(role=Role.USER, content=[Part.from_text('refund A-17')]), detach=True),
        init=AgentInit(),
    )
    snapshot_id = out.response.snapshot_id
    assert snapshot_id is not None
    for _ in range(200):
        snap = await agent.get_snapshot_data(snapshot_id=snapshot_id)
        if snap is not None and snap.status == SnapshotStatus.FAILED:
            break
        await asyncio.sleep(0.01)
    else:
        raise AssertionError('detached turn never finalized')

    # 2. In process, the stored snapshot keeps the exception's text.
    assert snap.error is not None
    assert snap.error.message == 'pos sync failed: dsn=pg://admin@10.0.0.4'

    # 3. Over HTTP, getSnapshot sends the generic message.
    served = client(agent).post('/api/chat/getSnapshot', json={'snapshotId': snapshot_id})
    assert served.json()['result']['error'] == {'status': 'INTERNAL', 'message': 'Internal Error'}
