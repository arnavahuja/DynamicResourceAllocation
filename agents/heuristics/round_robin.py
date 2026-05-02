from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent


class RoundRobinAgent(BaseAgent):
    """Always dispatch the head-of-queue job (k=0); cycle through servers
    0 → N-1 → 0, skipping any that can't fit. This matches the classic
    head-of-queue RR baseline so comparisons stay fair.

    Action encoding: k * N + n (joint queue/server action space).
    """

    def __init__(self, n_servers: int, queue_size: int):
        self.n_servers = n_servers
        self.queue_size = queue_size
        self.wait_action = n_servers * queue_size
        self._cursor = 0

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        N = self.n_servers
        # Try k=0 (head of queue), rotate through servers.
        for offset in range(N):
            n = (self._cursor + offset) % N
            a = 0 * N + n
            if action_mask[a]:
                self._cursor = (n + 1) % N
                return int(a)
        # Head-of-queue can't fit anywhere → wait this step.
        return int(self.wait_action)

    def on_episode_end(self) -> None:
        self._cursor = 0
