from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter

from backend.models.schemas import OfflineTrainRequest, TrainResponse
from backend.services import training_service

router = APIRouter(prefix="/offline", tags=["offline"])


@router.post("/train", response_model=TrainResponse)
def start_offline_train(req: OfflineTrainRequest) -> TrainResponse:
    run_id = training_service.start_offline_training(req)
    return TrainResponse(run_id=run_id)


@router.get("/dataset_exists")
def dataset_exists(path: str = "data/processed/offline.parquet") -> dict:
    """Helper for the frontend to know whether auto-generation will fire."""
    repo_root = Path(__file__).resolve().parents[2]
    p = (repo_root / path).resolve() if not Path(path).is_absolute() else Path(path)
    return {"path": str(p), "exists": p.exists()}
