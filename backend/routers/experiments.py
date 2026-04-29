from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.models import db

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _row_to_summary(row: dict) -> dict:
    cfg = json.loads(row.get("config_json") or "{}")
    summary = json.loads(row.get("summary_json") or "{}") if row.get("summary_json") else {}
    eval_d = json.loads(row.get("eval_json") or "{}") if row.get("eval_json") else {}
    return {
        "run_id": row["run_id"],
        "agent": row["agent"],
        "status": row["status"],
        "created_at": row["created_at"],
        "n_servers": cfg.get("n_servers"),
        "episodes": cfg.get("episodes"),
        "mean_reward_last10": summary.get("mean_reward_last10"),
        "mean_power": eval_d.get("mean_power"),
        "sla_violation_rate": eval_d.get("sla_violation_rate"),
    }


@router.get("")
def list_all() -> list[dict]:
    return [_row_to_summary(r) for r in db.list_experiments()]


@router.get("/{run_id}/results")
def get_results(run_id: str) -> dict:
    row = db.get_experiment(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return {
        "summary": _row_to_summary(row),
        "config": json.loads(row.get("config_json") or "{}"),
        "episodes": db.get_episodes(run_id),
        "eval": json.loads(row["eval_json"]) if row.get("eval_json") else None,
        "error": row.get("error"),
    }


@router.delete("/{run_id}")
def delete(run_id: str) -> dict:
    db.delete_experiment(run_id)
    return {"deleted": run_id}
