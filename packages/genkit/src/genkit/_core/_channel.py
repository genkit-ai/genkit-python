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

"""Channel for async streaming with final value, and uvloop-aware runner."""

from __future__ import annotations

import asyncio
import contextlib
import sys
import weakref
from collections.abc import AsyncIterable, AsyncIterator, Awaitable, Callable, Coroutine
from typing import Any, Generic, TypeVar

from typing_extensions import Self, TypeVar as TypeVarExt, override

from genkit._core._logger import get_logger

if sys.version_info >= (3, 13):
    # Reuse the stdlib exception so a queue closed via native shutdown() and one
    # closed via the emulated path raise the exact same type, and so callers
    # can catch either interchangeably.
    from asyncio import QueueShutDown
else:

    class QueueShutDown(Exception):  # noqa: N818
        """Raised when interacting with a closed CloseableQueue."""


logger = get_logger(__name__)

T = TypeVar('T')
T_co = TypeVarExt('T_co')
R = TypeVarExt('R', default=Any)
# No defaults, so they can sit in any order in the helpers' signatures.
_ChunkT = TypeVar('_ChunkT')
_ResultT = TypeVar('_ResultT')
_PickedT = TypeVar('_PickedT')


class Channel(Generic[T_co, R]):
    """Async channel for streaming values with a final result when closed."""

    def __init__(self) -> None:
        self.queue: asyncio.Queue[T_co] = asyncio.Queue()
        self.closed: asyncio.Future[R] = asyncio.Future()
        self._close_future: asyncio.Future[R] | None = None

    def __aiter__(self) -> AsyncIterator[T_co]:
        return self

    def _cleanup_pop_task(self, pop_task: asyncio.Future[T_co]) -> None:
        if not pop_task.done():
            pop_task.cancel()
            return
        if pop_task.cancelled():
            return
        if pop_task.exception() is None:
            self.queue._queue.appendleft(pop_task.result())  # type: ignore[attr-defined]

    async def __anext__(self) -> T_co:
        if not self.queue.empty():
            return self.queue.get_nowait()

        pop_task = asyncio.ensure_future(self._pop())
        if not self._close_future:
            try:
                return await pop_task
            except BaseException:
                self._cleanup_pop_task(pop_task)
                raise

        try:
            finished, _ = await asyncio.wait(
                [pop_task, self._close_future],
                return_when=asyncio.FIRST_COMPLETED,
            )
        except BaseException:
            self._cleanup_pop_task(pop_task)
            raise

        if pop_task in finished:
            return pop_task.result()

        self._cleanup_pop_task(pop_task)
        if self._close_future in finished:
            if self._close_future.cancelled():
                if not self.queue.empty():
                    return self.queue.get_nowait()
                raise StopAsyncIteration
            # generate_stream's async for is how callers see the run. If the
            # generate task died, that has to show up here, not as an empty
            # stream that only fails later on .response.
            await self._close_future
            if not self.queue.empty():
                return self.queue.get_nowait()
            raise StopAsyncIteration

        return await pop_task

    def send(self, value: T_co) -> None:
        """Send a value into the channel."""
        self.queue.put_nowait(value)

    def set_close_future(self, future: asyncio.Future[R]) -> None:
        """Set a future that closes the channel when completed."""
        if future is None:  # pyright: ignore[reportUnnecessaryComparison]
            raise ValueError('Cannot set a None future')  # pyright: ignore[reportUnreachable]

        def _handle_done(v: asyncio.Future[R]) -> None:
            if v.cancelled():
                _ = self.closed.cancel()
            elif (exc := v.exception()) is not None:
                self.closed.set_exception(exc)
                # `closed` only mirrors the run. The error reaches the caller
                # through `async for` and the response future, so mark this
                # copy retrieved or GC logs "Future exception was never
                # retrieved" even when the caller handled it.
                _ = self.closed.exception()
            else:
                self.closed.set_result(v.result())

        self._close_future = asyncio.ensure_future(future)
        if self._close_future is not None:  # pyright: ignore[reportUnnecessaryComparison]
            self._close_future.add_done_callback(_handle_done)

    async def _pop(self) -> T_co:
        r = await self.queue.get()
        self.queue.task_done()
        if r is None:
            raise StopAsyncIteration
        return r


def start_stream(
    run: Callable[[Callable[[_ChunkT], None]], Coroutine[Any, Any, _ResultT]],
) -> tuple[Channel[_ChunkT, _ResultT], asyncio.Task[_ResultT]]:
    """Run ``run(send_chunk)`` as a task that feeds a Channel; return both.

    The channel ends when the task does and re-raises its error in the
    ``async for``. Every StreamResponse is built from this pair: models hand
    out the task as ``.response``, actions a ``mirror_result`` of it.
    """
    channel: Channel[_ChunkT, _ResultT] = Channel()
    task = asyncio.create_task(run(channel.send))
    channel.set_close_future(task)
    return channel, task


def mirror_result(task: asyncio.Future[_ResultT], pick: Callable[[_ResultT], _PickedT]) -> asyncio.Future[_PickedT]:
    """A future that settles like ``task``, with ``pick`` applied to a result.

    Cancelling an awaiter of the mirror doesn't cancel the run. The mirror's
    error is marked retrieved: a caller that only ran the ``async for``
    already saw it there.
    """
    mirror: asyncio.Future[_PickedT] = asyncio.get_running_loop().create_future()

    def _settle(done: asyncio.Future[_ResultT]) -> None:
        if mirror.done():
            return
        if done.cancelled():
            _ = mirror.cancel()
        elif (exc := done.exception()) is not None:
            mirror.set_exception(exc)
            _ = mirror.exception()
        else:
            mirror.set_result(pick(done.result()))

    task.add_done_callback(_settle)
    return mirror


_pending_stops: set[asyncio.Task[None]] = set()


def _spawn_stop(stop: Callable[[], Awaitable[None]], *, on_done: Callable[[], None] | None = None) -> None:
    task = asyncio.ensure_future(stop())
    _pending_stops.add(task)

    def _finish(done: asyncio.Task[None]) -> None:
        _pending_stops.discard(done)
        if on_done is not None:
            on_done()

    task.add_done_callback(_finish)


class StoppableStream(AsyncIterator[T]):
    """Single-pass ``AsyncIterator`` over a run's chunks, with explicit ``aclose()``.

    ``__aiter__`` returns ``self`` (``aiter(s) is s``). Closing via
    ``async with`` or ``await s.aclose()`` stops the run if it hasn't finished
    and waits for cleanup. Without a close, the run is also stopped when:

    - the task reading it ends cancelled or with an unhandled exception before
      the stream finishes (a web adapter cancelling the flow on client
      disconnect, a failing ``TaskGroup``, or an exception raised inside the
      ``async for`` body). A timeout around one ``__anext__()`` or ``anext()``
      (an SSE heartbeat tick) doesn't count: the reader carries on.
    - the reader started iterating, then dropped the stream unfinished (a bare
      ``async for`` that hits ``break``, then lets the stream go out of scope).

    Both stops are scheduled on the loop; use ``async with`` to wait for
    cleanup. Reading zero chunks leaves the run going, so
    ``await s.response`` alone still works.
    """

    def __init__(self, *, chunks: AsyncIterable[T], stop: Callable[[], Awaitable[None]] | None) -> None:
        self._chunks: AsyncIterable[T] | None = chunks
        self._source: AsyncIterator[T] | None = None
        self._stop = stop
        self._finished = False
        self._stopped = False
        self._loop_ref: weakref.ref[asyncio.AbstractEventLoop] | None = None
        self._watched: weakref.WeakSet[asyncio.Task[Any]] = weakref.WeakSet()

    @override
    def __aiter__(self) -> Self:
        return self

    @override
    async def __anext__(self) -> T:
        if self._finished:
            raise StopAsyncIteration
        if self._source is None:
            assert self._chunks is not None
            self._source = self._chunks.__aiter__()
            self._loop_ref = weakref.ref(asyncio.get_running_loop())
        self._watch_reader()
        try:
            return await self._source.__anext__()
        except StopAsyncIteration:
            self._finished = True
            self._chunks = None
            self._source = None
            self._stop = None
            self._loop_ref = None
            raise

    def _watch_reader(self) -> None:
        """Stop the run once a reader task dies cancelled or with an unhandled exception.

        A cancelled or crashing `async for` without `async with` never calls
        aclose(), so the run would keep generating (and calling tools) with
        nobody reading. The decision waits for the task to end because a
        timeout around a single __anext__() or anext() cancels only that await,
        and the reader keeps going. Uses a weakref to ``self`` so the task's
        done-callback list does not pin a dropped stream until the task ends.
        """
        task = asyncio.current_task()
        if self._stop is None or task is None or task in self._watched:
            return
        # A task whose whole job is this one __anext__() (wait_for before
        # 3.12, ensure_future(anext(s)), or 2-arg anext(s, default)) is a poll,
        # not the reader.
        coro = task.get_coro()
        if getattr(coro, 'cr_code', None) is _RAW_ANEXT_CODE or type(coro).__name__ == 'anext_awaitable':
            return
        self._watched.add(task)
        stream_ref = weakref.ref(self)

        def _on_reader_done(done_task: asyncio.Task[Any]) -> None:
            if done_task.cancelled() or done_task.exception() is not None:
                stream = stream_ref()
                if stream is not None:
                    stream._stop_run()

        task.add_done_callback(_on_reader_done)

    def _clear_stop_ref(self) -> None:
        self._stop = None

    def _stop_run(self) -> None:
        stop = self._stop
        if stop is None or self._stopped or self._finished:
            return
        self._stopped = True
        self._loop_ref = None
        stream_ref = weakref.ref(self)
        _spawn_stop(stop, on_done=lambda: (s := stream_ref()) is not None and s._clear_stop_ref())  # type: ignore[func-returns-value]

    def __del__(self) -> None:
        # A bare `async for` that breaks never calls aclose(). Once the reader
        # lets go of a stream it started reading, stop the run.
        stop = self._stop
        loop = self._loop_ref() if self._loop_ref is not None else None
        if stop is None or self._stopped or self._finished or loop is None or loop.is_closed():
            return
        self._stopped = True
        self._stop = None
        self._loop_ref = None
        with contextlib.suppress(RuntimeError):  # loop already closed
            loop.call_soon_threadsafe(_spawn_stop, stop)

    async def aclose(self) -> None:
        """Stop the run if it hasn't finished, and wait for it to end."""
        if self._finished:
            return
        self._stopped = True
        self._loop_ref = None
        target = self._source if self._source is not None else self._chunks
        self._chunks = None
        self._source = None
        source_aclose = getattr(target, 'aclose', None)
        stop = self._stop
        try:
            if source_aclose is not None:
                await source_aclose()
            if stop is not None:
                await stop()
        finally:
            self._finished = True
            self._stop = None

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.aclose()


_RAW_ANEXT_CODE = StoppableStream.__anext__.__code__


def run_loop(coro: Coroutine[object, object, T], *, debug: bool | None = None) -> T:
    """Run a coroutine using uvloop if available, otherwise asyncio."""
    try:
        import uvloop  # noqa: PLC0415

        logger.debug('Using uvloop (recommended)')
        return uvloop.run(coro, debug=debug)
    except ImportError as e:
        logger.debug('Using asyncio (install uvloop for better performance)', error=e)
        return asyncio.run(coro, debug=debug)


class CloseableQueue(asyncio.Queue[T]):
    """An asyncio.Queue subclass with a synchronous, idempotent close().

    Once closed, put()/put_nowait() raise QueueShutDown immediately, while
    get()/get_nowait() drain any buffered items first and then raise
    QueueShutDown once the queue is empty. close() also wakes coroutines that
    are already blocked in get() (and put() on a bounded queue). Supports async
    iteration via ``async for``.

    Python 3.13+ ships this as Queue.shutdown(), so we delegate to it there. On
    3.10-3.12 the same behavior is emulated by hand.
    """

    def __init__(self, maxsize: int = 0) -> None:
        super().__init__(maxsize=maxsize)
        self.closed = False

    def close(self) -> None:
        """Close the queue synchronously and idempotently.

        Stops accepting new items, lets buffered items drain, then makes blocked
        and future getters raise QueueShutDown. Must be called on the event loop
        thread: asyncio.Queue is loop-affine and not thread-safe.
        """
        if self.closed:
            return
        self.closed = True

        if sys.version_info >= (3, 13):
            # native shutdown rejects new puts, leaves buffered items to drain,
            # and wakes blocked getters so they raise once the queue is empty.
            super().shutdown(immediate=False)
            return

        # wake anyone blocked in get() so they observe the closed-and-empty
        # state; the base queue tracks its waiters as plain deques.
        getters = getattr(self, '_getters', None)
        while getters:
            getter = getters.popleft()
            if not getter.done():
                getter.set_exception(QueueShutDown())

        # wake anyone blocked in put() on a bounded queue.
        putters = getattr(self, '_putters', None)
        while putters:
            putter = putters.popleft()
            if not putter.done():
                putter.set_exception(QueueShutDown())

    def is_closed(self) -> bool:
        return self.closed

    async def put(self, item: T) -> None:
        if self.closed:
            raise QueueShutDown('Queue is closed')
        await super().put(item)

    def put_nowait(self, item: T) -> None:
        if self.closed:
            raise QueueShutDown('Queue is closed')
        super().put_nowait(item)

    async def get(self) -> T:
        # If the queue is closed and empty, raise QueueShutDown to signal end of stream
        if self.closed and self.empty():
            raise QueueShutDown('Queue is closed and empty')
        return await super().get()

    def get_nowait(self) -> T:
        if self.closed and self.empty():
            raise QueueShutDown('Queue is closed and empty')
        return super().get_nowait()

    def __aiter__(self) -> AsyncIterator[T]:
        return self

    async def __anext__(self) -> T:
        try:
            return await self.get()
        except QueueShutDown:
            raise StopAsyncIteration from None
