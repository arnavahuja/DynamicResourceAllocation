"""End-to-end backend smoke tests using FastAPI's TestClient + a brief
poll loop. Uses heuristic agents (round_robin) so tests run in <2s.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Isolate DB per test
    monkeypatch.setenv("DATABASE_URL", f"sqlite:///{tmp_path}/test.db")
    # Reload settings so the new DB URL takes effect
    import backend.core.config as bcfg
    bcfg.settings.DATABASE_URL = f"sqlite:///{tmp_path}/test.db"

    from fastapi.testclient import TestClient

    from backend.main import app

    with TestClient(app) as c:
        yield c


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_train_and_list(client):
    r = client.post(
        "/train",
        json={
            "agent": "round_robin",
            "n_servers": 4,
            "episodes": 3,
            "episode_length": 30,
            "eval_episodes": 2,
            "seed": 0,
        },
    )
    assert r.status_code == 200
    run_id = r.json()["run_id"]
    assert run_id.startswith("round_robin_")

    # Poll until completion (heuristic on tiny episodes — finishes fast)
    for _ in range(200):
        s = client.get(f"/train/{run_id}/status").json()
        if s["status"] in ("completed", "failed"):
            break
        time.sleep(0.05)
    assert s["status"] == "completed", s
    assert s["progress"] == 1.0

    runs = client.get("/experiments").json()
    assert any(r["run_id"] == run_id for r in runs)

    res = client.get(f"/experiments/{run_id}/results").json()
    assert res["summary"]["run_id"] == run_id
    assert len(res["episodes"]) == 3
    assert res["eval"] is not None


def test_simulate(client):
    r = client.post(
        "/simulate",
        json={
            "agent": "round_robin",
            "n_servers": 4,
            "episode_length": 20,
            "seed": 0,
        },
    )
    assert r.status_code == 200
    body = r.json()
    assert len(body["steps"]) == 20
    assert all("server_utilizations" in s for s in body["steps"])


def test_websocket_streams_episodes(client):
    r = client.post(
        "/train",
        json={
            "agent": "round_robin",
            "n_servers": 3,
            "episodes": 2,
            "episode_length": 20,
            "eval_episodes": 0,
            "seed": 0,
        },
    )
    run_id = r.json()["run_id"]

    received = []
    with client.websocket_connect(f"/ws/metrics/{run_id}") as ws:
        # We may join late; bus replays buffered messages.
        for _ in range(40):
            msg = ws.receive_json()
            received.append(msg)
            if msg.get("event") in ("completed", "closed", "failed"):
                break
    events = [m.get("event") for m in received]
    assert "episode" in events or "completed" in events
