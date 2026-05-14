from __future__ import annotations

import asyncio

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from backend.core.websocket_manager import bus

router = APIRouter(tags=["metrics"])


@router.websocket("/ws/metrics/{run_id}")
async def metrics_ws(websocket: WebSocket, run_id: str) -> None:
    await websocket.accept()
    queue = bus.subscribe(run_id)
    try:
        while True:
            try:
                msg = await asyncio.wait_for(queue.get(), timeout=30.0)
            except asyncio.TimeoutError:
                # heartbeat to keep connection alive
                await websocket.send_json({"event": "ping"})
                continue
            await websocket.send_json(msg)
            if msg.get("event") in ("completed", "failed", "closed"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        bus.unsubscribe(run_id, queue)
