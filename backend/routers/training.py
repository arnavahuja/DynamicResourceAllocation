from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException

from backend.models import db
from backend.models.schemas import (
    RunStatusResponse,
    TrainRequest,
    TrainResponse,
)
from backend.services import training_service

router = APIRouter(prefix="/train", tags=["training"])


@router.post("", response_model=TrainResponse)
def start_train(req: TrainRequest) -> TrainResponse:
    run_id = training_service.start_training(req)
    return TrainResponse(run_id=run_id)


@router.get("/{run_id}/status", response_model=RunStatusResponse)
def status(run_id: str) -> RunStatusResponse:
    job = training_service.get_status(run_id)
    if job is not None:
        return RunStatusResponse(
            run_id=run_id,
            status=job["status"],
            progress=job.get("progress", 0.0),
            eta_seconds=job.get("eta_seconds"),
            current_episode=job.get("current_episode", 0),
            total_episodes=job.get("total_episodes", 0),
            last_reward=job.get("last_reward"),
            error=job.get("error"),
        )

    # Fallback: in-memory job tracker is empty (typical after a backend
    # restart) but the run still exists in SQLite. Return persisted state
    # so MonitorPage stops 404-spamming the logs on every poll.
    row = db.get_experiment(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="run_id not found")

    cfg = json.loads(row.get("config_json") or "{}")
    summary = json.loads(row["summary_json"]) if row.get("summary_json") else {}
    total_eps = int(cfg.get("episodes", 0))
    completed_eps = int(summary.get("n_episodes", 0)) if summary else 0
    persisted_status = row.get("status") or "unknown"
    progress = 1.0 if persisted_status == "completed" else (
        completed_eps / total_eps if total_eps else 0.0
    )
    return RunStatusResponse(
        run_id=run_id,
        status=persisted_status,
        progress=progress,
        eta_seconds=None,
        current_episode=completed_eps,
        total_episodes=total_eps,
        last_reward=summary.get("mean_reward_last10") if summary else None,
        error=row.get("error"),
    )


@router.delete("/{run_id}")
def cancel(run_id: str) -> dict:
    ok = training_service.cancel(run_id)
    if not ok:
        raise HTTPException(
            status_code=400, detail="cannot cancel — unknown or already finished"
        )
    return {"cancelled": run_id}
