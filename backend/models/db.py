"""Tiny SQLite-backed experiment store.

Pure stdlib `sqlite3` — no ORM. Two tables: `experiments` (one row per run)
and `episodes` (one row per training episode, foreign-keyed to a run).
"""
from __future__ import annotations

import json
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

from backend.core.config import settings

_DB_LOCK = threading.Lock()


def _db_path() -> str:
    url = settings.DATABASE_URL
    if url.startswith("sqlite:///"):
        return url.replace("sqlite:///", "", 1)
    if url.startswith("sqlite://"):
        return url.replace("sqlite://", "", 1)
    return url


@contextmanager
def get_conn():
    path = _db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with _DB_LOCK:
        conn = sqlite3.connect(path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()


def init_db() -> None:
    with get_conn() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                run_id TEXT PRIMARY KEY,
                agent TEXT NOT NULL,
                status TEXT NOT NULL,
                created_at TEXT NOT NULL,
                config_json TEXT NOT NULL,
                summary_json TEXT,
                eval_json TEXT,
                error TEXT
            );
            CREATE TABLE IF NOT EXISTS episodes (
                run_id TEXT NOT NULL,
                episode INTEGER NOT NULL,
                reward REAL NOT NULL,
                power REAL NOT NULL,
                sla_violations INTEGER NOT NULL,
                steps INTEGER NOT NULL,
                PRIMARY KEY (run_id, episode),
                FOREIGN KEY (run_id) REFERENCES experiments(run_id) ON DELETE CASCADE
            );
            CREATE INDEX IF NOT EXISTS idx_episodes_run ON episodes(run_id);
            """
        )


def insert_experiment(
    run_id: str, agent: str, status: str, created_at: str, config: dict
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT INTO experiments (run_id, agent, status, created_at, config_json) "
            "VALUES (?, ?, ?, ?, ?)",
            (run_id, agent, status, created_at, json.dumps(config)),
        )


def update_experiment_status(
    run_id: str,
    status: str,
    summary: dict | None = None,
    eval_data: dict | None = None,
    error: str | None = None,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "UPDATE experiments SET status = ?, "
            "summary_json = COALESCE(?, summary_json), "
            "eval_json = COALESCE(?, eval_json), "
            "error = COALESCE(?, error) "
            "WHERE run_id = ?",
            (
                status,
                json.dumps(summary) if summary else None,
                json.dumps(eval_data) if eval_data else None,
                error,
                run_id,
            ),
        )


def insert_episode(
    run_id: str, episode: int, reward: float, power: float,
    sla_violations: int, steps: int,
) -> None:
    with get_conn() as conn:
        conn.execute(
            "INSERT OR REPLACE INTO episodes "
            "(run_id, episode, reward, power, sla_violations, steps) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, episode, reward, power, sla_violations, steps),
        )


def list_experiments() -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT * FROM experiments ORDER BY created_at DESC"
        ).fetchall()
        return [dict(r) for r in rows]


def get_experiment(run_id: str) -> dict | None:
    with get_conn() as conn:
        row = conn.execute(
            "SELECT * FROM experiments WHERE run_id = ?", (run_id,)
        ).fetchone()
        return dict(row) if row else None


def get_episodes(run_id: str) -> list[dict]:
    with get_conn() as conn:
        rows = conn.execute(
            "SELECT episode, reward, power, sla_violations, steps "
            "FROM episodes WHERE run_id = ? ORDER BY episode",
            (run_id,),
        ).fetchall()
        return [dict(r) for r in rows]


def delete_experiment(run_id: str) -> None:
    with get_conn() as conn:
        conn.execute("DELETE FROM episodes WHERE run_id = ?", (run_id,))
        conn.execute("DELETE FROM experiments WHERE run_id = ?", (run_id,))
