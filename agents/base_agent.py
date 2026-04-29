from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from environment.cluster_env import CloudClusterEnv


class BaseAgent(ABC):
    """Common interface for RL agents and heuristic baselines."""

    @abstractmethod
    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        """Pick a server index (0 .. N-1) for the head-of-queue job."""

    def on_step(self, *args, **kwargs) -> None:  # noqa: D401
        """Hook called by the trainer after each env step. Override if learning."""

    def on_episode_end(self) -> None:
        """Hook called after each episode."""

    def save(self, path: str) -> None:
        """Persist agent state to disk. No-op by default."""

    def load(self, path: str) -> None:
        """Restore agent state from disk. No-op by default."""


def compute_action_mask(env: CloudClusterEnv) -> np.ndarray:
    """Return bool array of length N_SERVERS — True if the head-of-queue job
    can be assigned to that server given current capacity."""
    mask = np.zeros(env.num_servers, dtype=bool)
    if len(env.job_queue) == 0:
        # No job to place — every action is a no-op; mark all valid so
        # the agent doesn't see a degenerate all-zero mask.
        mask[:] = True
        return mask
    job = env.job_queue[0]
    for i, server in enumerate(env.servers):
        mask[i] = server.can_fit(job)
    if not mask.any():
        # No server can fit the job — return all-True so policy still picks
        # something; the env will apply the invalid-action penalty.
        mask[:] = True
    return mask
