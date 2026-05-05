"""Backfill eval_json for runs that completed without an eval pass.

When a run was launched with `n_test_seeds>0, eval_episodes=0` BUT got its
seed pool wiped (the pre-fix bug for trace_sampled), the eval branch fell
through to `evaluate(n_episodes=0)` and stored NaN/0 metrics — or skipped
eval entirely. This script reads per-episode rows from the `episodes` table
and computes mean-power + an SLA-violations-per-step proxy directly from
the training trajectory, then writes them into `experiments.eval_json` so
the frontend table populates without re-running the experiment.

Caveats:
    1. mean_power here is the TRAINING-time per-episode mean, not held-out
       eval power. For RR/SJF/FFD that's identical (no learning). For
       DQN/PPO/Agentic it includes early-exploration episodes where power
       is typically higher than the converged policy's.
    2. We don't persist jobs_completed per episode, so we can't compute the
       real SLA *rate* (violations / jobs_completed). We store
       `sla_violation_rate_per_step = sum(violations) / sum(steps)` as a
       proxy. Documented as such in the eval payload.

Usage:
    # Backfill every completed run that has no eval_json yet
    python -m scripts.backfill_eval --all

    # Backfill a specific run
    python -m scripts.backfill_eval --run-id ppo_het_n50_s0_tr50te50_abc123

    # Dry-run (show what would change)
    python -m scripts.backfill_eval --all --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.models import db


def compute_eval_from_episodes(episodes: list[dict]) -> dict | None:
    if not episodes:
        return None
    powers = np.array([e["power"] for e in episodes], dtype=float)
    slas = np.array([e["sla_violations"] for e in episodes], dtype=float)
    steps = np.array([e["steps"] for e in episodes], dtype=float)

    total_steps = float(steps.sum())
    sla_per_step = float(slas.sum() / total_steps) if total_steps > 0 else 0.0

    return {
        "agent": "backfilled-from-episodes",
        "mean_reward": float(np.mean([e["reward"] for e in episodes])),
        "std_reward": float(np.std([e["reward"] for e in episodes])),
        "mean_power": float(powers.mean()),
        "std_power": float(powers.std()),
        "sla_violation_rate": sla_per_step,
        "sla_violation_rate_kind": "per_step",  # not per job — see script docstring
        "mean_jobs_completed": None,
        "n_episodes": len(episodes),
        "_backfilled": True,
    }


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run-id", default=None,
                   help="Single run to backfill (default: all completed runs missing eval).")
    p.add_argument("--all", action="store_true",
                   help="Backfill every completed run that has no eval_json.")
    p.add_argument("--force", action="store_true",
                   help="Overwrite eval_json even if already populated.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print what would change; don't write to DB.")
    args = p.parse_args()

    if not args.run_id and not args.all:
        p.error("pass --run-id <id> or --all")

    if args.run_id:
        rows = [db.get_experiment(args.run_id)]
        rows = [r for r in rows if r is not None]
    else:
        rows = [r for r in db.list_experiments() if r["status"] == "completed"]

    n_changed = 0
    n_skipped = 0
    for row in rows:
        run_id = row["run_id"]
        existing = row.get("eval_json")
        if existing and not args.force:
            print(f"[skip] {run_id}: eval_json already present")
            n_skipped += 1
            continue

        episodes = db.get_episodes(run_id)
        if not episodes:
            print(f"[skip] {run_id}: no episode rows")
            n_skipped += 1
            continue

        eval_data = compute_eval_from_episodes(episodes)
        if eval_data is None:
            print(f"[skip] {run_id}: empty eval payload")
            n_skipped += 1
            continue

        print(
            f"[fill] {run_id}: "
            f"mean_power={eval_data['mean_power']:.1f} "
            f"sla/step={eval_data['sla_violation_rate']:.4f} "
            f"n_eps={eval_data['n_episodes']}"
        )
        if not args.dry_run:
            db.update_experiment_status(run_id, row["status"], eval_data=eval_data)
        n_changed += 1

    print(f"\nDone. backfilled={n_changed} skipped={n_skipped} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
