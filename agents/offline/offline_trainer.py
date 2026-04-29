"""Offline trainer: loads a Parquet dataset of logged transitions and runs
FQI + Lagrangian dual updates on a CMDPAgent. No environment access during
training — the env is only used at evaluation time.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass
class OfflineDataset:
    state: np.ndarray            # (N, obs_dim) float32
    action: np.ndarray           # (N,) int64
    action_mask: np.ndarray      # (N, n_actions) bool
    next_state: np.ndarray       # (N, obs_dim) float32
    next_action_mask: np.ndarray # (N, n_actions) bool
    reward: np.ndarray           # (N,) float32 — used as Q_r target signal
    cost: np.ndarray             # (N,) float32 — used as Q_c target signal
    done: np.ndarray             # (N,) float32

    @property
    def n(self) -> int:
        return self.state.shape[0]

    @property
    def obs_dim(self) -> int:
        return self.state.shape[1]

    @property
    def n_actions(self) -> int:
        return self.action_mask.shape[1]


def load_dataset(path: str | Path, reward_kind: str = "neg_power") -> OfflineDataset:
    """Load a Parquet file produced by `data/generate_offline_dataset.py`.

    `reward_kind`:
        "neg_power" — Q_r learns expected discounted -power (default for CMDP)
        "env"       — Q_r learns the env's combined reward (DQN-style)
    """
    df = pd.read_parquet(path)
    state = np.stack(df["state"].values).astype(np.float32)
    action = df["action"].values.astype(np.int64)
    action_mask = np.stack(df["action_mask"].values).astype(bool)
    next_state = np.stack(df["next_state"].values).astype(np.float32)
    next_action_mask = np.stack(df["next_action_mask"].values).astype(bool)
    cost = df["cost"].values.astype(np.float32)
    done = df["done"].values.astype(np.float32)

    if reward_kind == "neg_power":
        reward = -df["power_norm"].values.astype(np.float32)
    elif reward_kind == "env":
        reward = df["reward"].values.astype(np.float32)
    else:
        raise ValueError(reward_kind)

    return OfflineDataset(
        state=state, action=action, action_mask=action_mask,
        next_state=next_state, next_action_mask=next_action_mask,
        reward=reward, cost=cost, done=done,
    )


def _sample_batch(ds: OfflineDataset, batch_size: int, device: torch.device, rng: np.random.Generator) -> dict[str, torch.Tensor]:
    idx = rng.integers(0, ds.n, size=batch_size)
    return {
        "state": torch.from_numpy(ds.state[idx]).to(device),
        "action": torch.from_numpy(ds.action[idx]).to(device),
        "action_mask": torch.from_numpy(ds.action_mask[idx]).to(device),
        "next_state": torch.from_numpy(ds.next_state[idx]).to(device),
        "next_action_mask": torch.from_numpy(ds.next_action_mask[idx]).to(device),
        "reward": torch.from_numpy(ds.reward[idx]).to(device),
        "cost": torch.from_numpy(ds.cost[idx]).to(device),
        "done": torch.from_numpy(ds.done[idx]).to(device),
    }


def train_offline(
    agent,
    dataset: OfflineDataset,
    iterations: int = 20_000,
    batch_size: int = 256,
    target_update_freq: int = 500,
    dual_update_freq: int = 200,
    log_every: int = 1000,
    seed: int = 0,
) -> list[dict]:
    """FQI + Lagrangian loop. Returns per-log-step metrics."""
    rng = np.random.default_rng(seed)
    metrics: list[dict] = []

    for it in range(1, iterations + 1):
        batch = _sample_batch(dataset, batch_size, agent.device, rng)
        losses = agent.fqi_step(batch)

        if it % dual_update_freq == 0:
            violation = agent.dual_step(batch)
        else:
            violation = None

        if it % target_update_freq == 0:
            agent.hard_update_targets()

        if it % log_every == 0 or it == 1:
            row = {
                "iteration": it,
                "loss_r": losses["loss_r"],
                "loss_c": losses["loss_c"],
                "lambda": agent.lam,
                "constraint_violation": violation,
            }
            metrics.append(row)
            print(
                f"[offline] it {it:6d}/{iterations} "
                f"loss_r={losses['loss_r']:.4f} loss_c={losses['loss_c']:.4f} "
                f"λ={agent.lam:.3f}"
                + (f" viol={violation:+.3f}" if violation is not None else "")
            )

    return metrics
