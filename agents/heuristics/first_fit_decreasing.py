from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent


class FirstFitDecreasingAgent(BaseAgent):
    """Pack jobs onto the *most loaded* legal server (best-fit-decreasing).

    This concentrates load to keep idle servers near zero utilization —
    a power-minimizing baseline that often violates SLAs under bursts.
    """

    def __init__(self, n_servers: int):
        self.n_servers = n_servers

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        cpu_utils = state[: 2 * self.n_servers : 2]
        scores = np.where(action_mask, cpu_utils, -np.inf)
        return int(np.argmax(scores))
