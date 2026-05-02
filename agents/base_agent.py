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
    """Return bool mask of shape (K * N + 1,) for the joint (job, server)
    action space.

    Layout: mask[k * N + n] = True if queue[k] exists AND server[n] can fit it.
            mask[K * N]    = "wait" — only legal when no dispatch is possible
                             (queue empty OR no fitting (job, server) pair).
                             Without this restriction DQN converges to a wait-
                             too-much local optimum, since waiting has zero
                             immediate cost and queue-pressure penalty is
                             heavily discounted.
    """
    N = env.num_servers
    K = env.job_queue_size
    mask = np.zeros(N * K + 1, dtype=bool)
    for k in range(min(K, len(env.job_queue))):
        job = env.job_queue[k]
        for n, server in enumerate(env.servers):
            if server.can_fit(job):
                mask[k * N + n] = True
    # Wait is only legal when no dispatch is.
    if not mask[: N * K].any():
        mask[N * K] = True
    return mask
