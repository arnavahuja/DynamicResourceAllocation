from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.models.schemas import SweepRequest, SweepResponse
from backend.services import sweep_service

router = APIRouter(prefix="/sweep", tags=["sweep"])


@router.post("", response_model=SweepResponse)
def start_sweep(req: SweepRequest) -> SweepResponse:
    try:
        return sweep_service.start_sweep(req)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/sweepable")
def sweepable() -> dict:
    """Frontend asks: which params can I sweep in each mode?"""
    return {
        "online": sorted(sweep_service.ONLINE_SWEEPABLE),
        "offline": sorted(sweep_service.OFFLINE_SWEEPABLE),
    }


@router.get("")
def list_all() -> list[dict]:
    return sweep_service.list_all_sweeps()


@router.get("/{sweep_id}")
def get_sweep(sweep_id: str) -> dict:
    return sweep_service.list_sweep(sweep_id)
