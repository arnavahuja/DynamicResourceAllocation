from __future__ import annotations

import asyncio
from collections import defaultdict
from typing import Any


class MetricsBus:
    """In-process pub/sub for live training metrics keyed by run_id.

    Trainers (running in worker threads) push dicts via `publish_threadsafe`;
    WebSocket handlers `subscribe(run_id)` and consume from an asyncio.Queue.
    The bus also keeps a small replay buffer per run so a late-joining
    websocket gets the most recent points.
    """

    def __init__(self, replay_size: int = 200):
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subs: dict[str, list[asyncio.Queue]] = defaultdict(list)
        self._replay: dict[str, list[dict[str, Any]]] = defaultdict(list)
        self._replay_size = replay_size
        self._closed: set[str] = set()

    def attach_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop

    def subscribe(self, run_id: str) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue(maxsize=1024)
        # backfill recent points
        for msg in self._replay.get(run_id, []):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                break
        if run_id in self._closed:
            q.put_nowait({"event": "closed"})
        self._subs[run_id].append(q)
        return q

    def unsubscribe(self, run_id: str, q: asyncio.Queue) -> None:
        if q in self._subs.get(run_id, []):
            self._subs[run_id].remove(q)

    async def _publish(self, run_id: str, msg: dict[str, Any]) -> None:
        buf = self._replay[run_id]
        buf.append(msg)
        if len(buf) > self._replay_size:
            del buf[: len(buf) - self._replay_size]
        for q in list(self._subs.get(run_id, [])):
            try:
                q.put_nowait(msg)
            except asyncio.QueueFull:
                pass  # drop on overflow rather than block trainer

    def publish_threadsafe(self, run_id: str, msg: dict[str, Any]) -> None:
        """Safe to call from worker threads."""
        if self._loop is None or self._loop.is_closed():
            # No loop yet (or shutting down) — buffer for replay only.
            buf = self._replay[run_id]
            buf.append(msg)
            if len(buf) > self._replay_size:
                del buf[: len(buf) - self._replay_size]
            return
        asyncio.run_coroutine_threadsafe(self._publish(run_id, msg), self._loop)

    def close_run(self, run_id: str) -> None:
        self._closed.add(run_id)
        self.publish_threadsafe(run_id, {"event": "closed"})


bus = MetricsBus()
