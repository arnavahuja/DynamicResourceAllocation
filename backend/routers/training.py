from __future__ import annotations

from fastapi import APIRouter, HTTPException

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
    if job is None:
        raise HTTPException(status_code=404, detail="run_id not found")
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


@router.delete("/{run_id}")
def cancel(run_id: str) -> dict:
    ok = training_service.cancel(run_id)
    if not ok:
        raise HTTPException(
            status_code=400, detail="cannot cancel — unknown or already finished"
        )
    return {"cancelled": run_id}
