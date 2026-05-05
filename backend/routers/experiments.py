from __future__ import annotations

import json
import re

from fastapi import APIRouter, HTTPException

from backend.models import db
from backend.models.schemas import ExperimentSummary
from backend.services import training_service

# Run IDs are auto-generated from `agent_workload_..._<uuid8>`. Allow only
# the characters that generator can produce — guards `shutil.rmtree` and
# `glob` against `..` / `/` traversal even if the DB ever contained a
# crafted run_id.
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_\-]+$")

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _optimal_reward(cfg: dict) -> float | None:
    """Theoretical reward ceiling for one episode.

    The reward formula now uses ACTIVE power (above idle), so an oracle
    with zero load and zero SLA violations would score exactly 0.0.
    This makes `gap_pct` directly interpretable: it's the cumulative
    cost the agent's policy added on top of doing nothing, expressed
    as a fraction of |best_real_score|.
    """
    return 0.0


def _row_to_summary(row: dict) -> dict:
    cfg = json.loads(row.get("config_json") or "{}")
    summary = json.loads(row.get("summary_json") or "{}") if row.get("summary_json") else {}
    eval_d = json.loads(row.get("eval_json") or "{}") if row.get("eval_json") else {}
    optimal = _optimal_reward(cfg)
    last10 = summary.get("mean_reward_last10")
    gap_pct = None
    # With active-power reward, optimal = 0, so |gap| = |last10|. Normalize
    # by episode_length so the number is comparable across run lengths and
    # has an intuitive scale (avg per-step cost).
    ep_len = cfg.get("episode_length")
    if last10 is not None and ep_len:
        gap_pct = abs(last10) / ep_len * 100.0
    return {
        "run_id": row["run_id"],
        "agent": row["agent"],
        "cluster_type": cfg.get("cluster_type", "homogeneous"),
        "status": row["status"],
        "created_at": row["created_at"],
        "n_servers": cfg.get("n_servers"),
        "episodes": cfg.get("episodes"),
        "mean_reward_last10": last10,
        "mean_reward_eval": eval_d.get("mean_reward"),
        "mean_power": eval_d.get("mean_power"),
        "mean_active_power": eval_d.get("mean_active_power"),
        "mean_asleep_servers": eval_d.get("mean_asleep_servers"),
        "sla_violation_rate": eval_d.get("sla_violation_rate"),
        "optimal_reward": optimal,
        "gap_pct": gap_pct,
        "sweep_id": cfg.get("sweep_id"),
        "sweep_param": cfg.get("sweep_param"),
        "sweep_value": cfg.get("sweep_value"),
    }


@router.get("", response_model=list[ExperimentSummary])
def list_all() -> list[dict]:
    return [_row_to_summary(r) for r in db.list_experiments()]


@router.get("/{run_id}/results")
def get_results(run_id: str) -> dict:
    row = db.get_experiment(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    cfg = json.loads(row.get("config_json") or "{}")
    return {
        "summary": _row_to_summary(row),
        "summary_full": json.loads(row["summary_json"]) if row.get("summary_json") else None,
        "config": cfg,
        "optimal_reward": _optimal_reward(cfg),
        "episodes": db.get_episodes(run_id),
        "eval": json.loads(row["eval_json"]) if row.get("eval_json") else None,
        "error": row.get("error"),
    }


@router.get("/{run_id}/cmdp_iterations")
def get_cmdp_iterations(run_id: str) -> dict:
    row = db.get_experiment(run_id)
    if row is None:
        raise HTTPException(status_code=404, detail="experiment not found")
    return {"run_id": run_id, "iterations": db.get_cmdp_iterations(run_id)}


@router.delete("/{run_id}")
def delete(run_id: str) -> dict:
    if not _RUN_ID_RE.match(run_id):
        raise HTTPException(status_code=400, detail="invalid run_id")

    # 1. If still running, request cancellation so the worker thread stops
    #    writing to a row we're about to delete.
    training_service.cancel(run_id)

    # 2. Drop DB rows (cascades to episodes + cmdp_iterations).
    db.delete_experiment(run_id)

    # 3. Drop on-disk artefacts (checkpoints + logs + any sub-agent files).
    artefacts = training_service.cleanup_artifacts(run_id)

    # 4. Clear in-memory job tracker + WebSocket bus state.
    training_service.purge_job_state(run_id)

    return {"deleted": run_id, "artefacts": artefacts}
