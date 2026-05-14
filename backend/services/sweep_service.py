"""Parameter-sweep orchestrator.

Fires N training runs through the existing training_service, sharing a
`sweep_id` tag persisted in each run's config_json so the frontend can
group + visualise them together.
"""
from __future__ import annotations

import json
import uuid

from backend.models import db
from backend.models.schemas import (
    OfflineTrainRequest,
    SweepRequest,
    SweepResponse,
    TrainRequest,
)
from backend.services import training_service

# Numeric-only parameters that make sense to sweep over. Restricting the
# allow-list both prevents accidental string sweeps and gives the frontend
# a fixed dropdown without server roundtripping for schema introspection.
ONLINE_SWEEPABLE = {
    "alpha", "beta", "n_servers", "episodes", "episode_length",
    "total_steps", "seed", "n_train_seeds", "n_test_seeds",
}
OFFLINE_SWEEPABLE = {
    "sla_budget", "cql_alpha", "iterations", "dataset_random_eps",
    "behavior_weight_rr", "behavior_weight_sjf", "behavior_weight_ffd",
    "batch_size", "target_update_freq", "dual_update_freq",
    "lambda_init", "lambda_lr", "dual_eval_seeds", "dual_eval_steps",
    "dataset_episodes", "n_test_seeds",
}


def start_sweep(req: SweepRequest) -> SweepResponse:
    if req.mode == "online":
        if req.base_online is None:
            raise ValueError("mode=online requires base_online")
        if req.sweep_param not in ONLINE_SWEEPABLE:
            raise ValueError(
                f"sweep_param '{req.sweep_param}' not in ONLINE_SWEEPABLE"
            )
    else:
        if req.base_offline is None:
            raise ValueError("mode=offline requires base_offline")
        if req.sweep_param not in OFFLINE_SWEEPABLE:
            raise ValueError(
                f"sweep_param '{req.sweep_param}' not in OFFLINE_SWEEPABLE"
            )

    sweep_id = f"sweep_{uuid.uuid4().hex[:8]}"
    run_ids: list[str] = []

    for v in req.sweep_values:
        # Cast to int when the underlying field is an int — pydantic will
        # otherwise reject a float sweep value for an int field.
        if req.mode == "online":
            base = req.base_online
            field_info = TrainRequest.model_fields[req.sweep_param]
        else:
            base = req.base_offline
            field_info = OfflineTrainRequest.model_fields[req.sweep_param]

        # crude type cast: if annotation is `int`, round to int.
        ann = field_info.annotation
        cast_v: float | int = int(round(v)) if ann is int else float(v)

        update = {
            req.sweep_param: cast_v,
            "sweep_id": sweep_id,
            "sweep_param": req.sweep_param,
            "sweep_value": float(v),
        }
        new_req = base.model_copy(update=update)

        if req.mode == "online":
            run_id = training_service.start_training(new_req)
        else:
            run_id = training_service.start_offline_training(new_req)
        run_ids.append(run_id)

    return SweepResponse(
        sweep_id=sweep_id, run_ids=run_ids,
        sweep_param=req.sweep_param, sweep_values=req.sweep_values,
    )


def list_sweep(sweep_id: str) -> dict:
    """Return all runs tagged with this sweep_id, with their swept value
    and current eval metrics."""
    out = []
    for row in db.list_experiments():
        cfg = json.loads(row.get("config_json") or "{}")
        if cfg.get("sweep_id") != sweep_id:
            continue
        ev = json.loads(row.get("eval_json") or "{}") if row.get("eval_json") else {}
        summ = json.loads(row.get("summary_json") or "{}") if row.get("summary_json") else {}
        out.append({
            "run_id": row["run_id"],
            "agent": row["agent"],
            "status": row["status"],
            "sweep_value": cfg.get("sweep_value"),
            "sweep_param": cfg.get("sweep_param"),
            "mean_reward_eval": ev.get("mean_reward"),
            "mean_reward_last10": summ.get("mean_reward_last10"),
            "mean_power": ev.get("mean_power"),
            "sla_violation_rate": ev.get("sla_violation_rate"),
            "n_episodes": ev.get("n_episodes"),
            "lambda_final": summ.get("lambda_final"),
            "constraint_slack": summ.get("constraint_slack"),
        })
    # Sort by sweep_value for chart-friendly order.
    out.sort(key=lambda r: (r["sweep_value"] if r["sweep_value"] is not None else 0))
    return {"sweep_id": sweep_id, "runs": out}


def list_all_sweeps() -> list[dict]:
    """Return distinct sweep_ids with summary counts."""
    seen: dict[str, dict] = {}
    for row in db.list_experiments():
        cfg = json.loads(row.get("config_json") or "{}")
        sid = cfg.get("sweep_id")
        if not sid:
            continue
        if sid not in seen:
            seen[sid] = {
                "sweep_id": sid,
                "sweep_param": cfg.get("sweep_param"),
                "mode": "offline" if row["agent"] == "cmdp" else "online",
                "n_runs": 0,
                "n_completed": 0,
                "created_at": row["created_at"],
            }
        seen[sid]["n_runs"] += 1
        if row["status"] == "completed":
            seen[sid]["n_completed"] += 1
        # Keep the earliest created_at as the sweep's start time.
        if row["created_at"] < seen[sid]["created_at"]:
            seen[sid]["created_at"] = row["created_at"]
    return sorted(seen.values(), key=lambda s: s["created_at"], reverse=True)
