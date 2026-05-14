"""Generate an offline RL dataset by rolling out a *behavior policy* through
the simulator and logging every transition to disk.

Behavior policy is a stochastic mix of the three heuristics + uniform
random — this gives the offline learner broad state-action coverage,
which CMDP/FQI methods need to bootstrap reliable Q-values.

Schema (Parquet via pandas):
    state[obs_dim]              float32  — observation at step t
    action                      int32
    action_mask[n_actions]      bool     — legal actions at step t
    next_state[obs_dim]         float32
    next_action_mask[n_actions] bool
    reward                      float32  — env reward (combined)
    cost                        float32  — SLA-cost component (the constraint)
    power                       float32  — raw step power (W)
    done                        bool

Usage:
    python -m data.generate_offline_dataset \\
        --episodes 50 --n-servers 10 --episode-length 500 \\
        --out data/processed/offline.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.base_agent import compute_action_mask
from agents.heuristics import (
    FirstFitDecreasingAgent,
    RoundRobinAgent,
    ShortestJobFirstAgent,
)
from environment import config as env_config
from environment.cluster_env import CloudClusterEnv
from environment.workload.synthetic import SyntheticWorkloadGenerator


def _behavior_action(
    policies: list, weights: np.ndarray, state, mask, rng: np.random.Generator
) -> int:
    """Pick which policy to follow this step, then ask it for an action.
    With small ε-uniform-random injection for state coverage."""
    if rng.random() < 0.10:
        legal = np.flatnonzero(mask)
        return int(rng.choice(legal)) if len(legal) else 0
    idx = int(rng.choice(len(policies), p=weights))
    return policies[idx].select_action(state, mask, greedy=False)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--n-servers", type=int, default=env_config.NUM_SERVERS)
    p.add_argument("--episode-length", type=int, default=env_config.EPISODE_LENGTH)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--out",
        type=str,
        default="data/processed/offline.parquet",
        help="Output Parquet path",
    )
    p.add_argument(
        "--workload",
        choices=["synthetic", "google_v3", "alibaba"],
        default="synthetic",
        help="Workload source. Real-trace options require traces in data/raw/.",
    )
    p.add_argument("--trace-path", default=None,
                   help="Path to trace dir/file when --workload != synthetic")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    np.random.seed(args.seed)

    if args.workload == "synthetic":
        workload = SyntheticWorkloadGenerator(seed=args.seed)
    elif args.workload == "google_v3":
        from environment.workload.google_v3 import GoogleV3WorkloadGenerator
        if args.trace_path is None:
            raise SystemExit("--trace-path required for google_v3")
        workload = GoogleV3WorkloadGenerator(trace_dir=args.trace_path)
    elif args.workload == "alibaba":
        from environment.workload.alibaba import AlibabaWorkloadGenerator
        if args.trace_path is None:
            raise SystemExit("--trace-path required for alibaba")
        workload = AlibabaWorkloadGenerator(trace_path=args.trace_path)
    else:
        raise SystemExit(args.workload)

    env = CloudClusterEnv(
        num_servers=args.n_servers,
        workload_generator=workload,
        episode_length=args.episode_length,
    )
    n_actions = int(env.action_space.n)
    obs_dim = int(np.prod(env.observation_space.shape))

    # Behavior policies + sampling weights
    policies = [
        RoundRobinAgent(n_actions=n_actions),
        ShortestJobFirstAgent(n_servers=n_actions),
        FirstFitDecreasingAgent(n_servers=n_actions),
    ]
    weights = np.array([0.34, 0.33, 0.33])

    # Power normalization to match the env's reward computation
    max_cluster_power = env_config.P_MAX * args.n_servers

    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset()
        mask = compute_action_mask(env)
        # Reset round-robin cursor
        for pol in policies:
            pol.on_episode_end()

        while True:
            action = _behavior_action(policies, weights, obs, mask, rng)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = compute_action_mask(env)
            done = terminated or truncated

            step_power = info.get("step_power", 0.0)
            step_sla = info.get("step_sla_violations", 0)

            rows.append(
                {
                    "state": obs.astype(np.float32).tolist(),
                    "action": int(action),
                    "action_mask": mask.astype(bool).tolist(),
                    "next_state": next_obs.astype(np.float32).tolist(),
                    "next_action_mask": next_mask.astype(bool).tolist(),
                    "reward": float(reward),
                    "cost": float(step_sla),                          # constraint signal
                    "power": float(step_power),
                    "power_norm": float(step_power / max_cluster_power),
                    "done": bool(done),
                }
            )

            obs = next_obs
            mask = next_mask
            if done:
                break

        if (ep + 1) % 5 == 0 or ep == 0:
            print(f"ep {ep+1}/{args.episodes} — transitions so far: {len(rows)}")

    df = pd.DataFrame(rows)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(
        f"\nWrote {len(df):,} transitions to {out} "
        f"(obs_dim={obs_dim}, n_actions={n_actions}, size={out.stat().st_size/1e6:.1f} MB)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
