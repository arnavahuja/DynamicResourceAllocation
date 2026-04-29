from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent


class ShortestJobFirstAgent(BaseAgent):
    """Assign the head-of-queue job to the legal server with the *least*
    current CPU load — i.e. the one expected to finish its existing work
    soonest. Greedy latency-minimizing baseline."""

    def __init__(self, n_servers: int):
        self.n_servers = n_servers

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        cpu_utils = state[: 2 * self.n_servers : 2]  # every other entry
        scores = np.where(action_mask, cpu_utils, np.inf)
        return int(np.argmin(scores))
