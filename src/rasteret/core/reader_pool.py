# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright Terrafloww Labs, Inc.

from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

from rasteret.fetch.cog import COGReader

logger = logging.getLogger(__name__)

T = TypeVar("T")


class AsyncCOGReaderPool:
    """Run a persistent ``COGReader`` on a background asyncio loop.

    Lifetime: call :meth:`close` (directly or via ``with``) to shut down the
    background loop and release the reader's HTTP connection pool.  The
    background thread is a daemon, so the interpreter can still exit cleanly
    if :meth:`close` is never called — but the obstore session may then emit
    "unclosed" warnings.  :class:`~rasteret.core.collection.Collection` owns
    its pool and closes it from ``Collection._close_reader_pool``.
    """

    def __init__(self, *, max_concurrent: int, backend: object | None = None) -> None:
        self.max_concurrent = max_concurrent
        self._backend = backend
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._reader: COGReader | None = None
        self._ready = threading.Event()
        self._error: BaseException | None = None
        self._closed = False
        self._start()

    def _start(self) -> None:
        if self._thread is not None:
            return
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        async def init() -> None:
            self._reader = COGReader(
                max_concurrent=self.max_concurrent,
                backend=self._backend,
            )
            await self._reader.__aenter__()

        try:
            loop.run_until_complete(init())
            self._loop = loop
        except BaseException as exc:
            self._error = exc
            self._loop = loop
        finally:
            self._ready.set()

        if self._error is not None:
            try:
                loop.close()
            except Exception:
                pass
            return

        try:
            loop.run_forever()
        finally:

            async def shutdown() -> None:
                if self._reader is not None:
                    await self._reader.__aexit__(None, None, None)

            try:
                loop.run_until_complete(shutdown())
            finally:
                loop.close()

    def run(self, coro: Coroutine[Any, Any, T]) -> T:
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(
                "Failed to initialize async COG reader pool"
            ) from self._error
        if self._loop is None:
            raise RuntimeError("Event loop not initialized in COG reader pool")
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    @property
    def reader(self) -> COGReader:
        self._ready.wait()
        if self._error is not None:
            raise RuntimeError(
                "Failed to initialize async COG reader pool"
            ) from self._error
        if self._reader is None:
            raise RuntimeError("COG reader not initialized in pool")
        return self._reader

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._ready.wait()
        loop = self._loop
        if loop is None:
            return
        loop.call_soon_threadsafe(loop.stop)
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=5)
            if self._thread.is_alive():
                logger.warning(
                    "COG reader pool thread did not join within 5 s; resources may leak"
                )

    def __enter__(self) -> AsyncCOGReaderPool:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
