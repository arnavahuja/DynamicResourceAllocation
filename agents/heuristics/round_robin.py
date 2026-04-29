from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent


class RoundRobinAgent(BaseAgent):
    """Cycle through servers 0 → N-1 → 0, skipping any that can't fit the
    current head-of-queue job. Pure baseline — no learning."""

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self._cursor = 0

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        for offset in range(self.n_actions):
            idx = (self._cursor + offset) % self.n_actions
            if action_mask[idx]:
                self._cursor = (idx + 1) % self.n_actions
                return int(idx)
        # No legal action — fall back to current cursor.
        idx = self._cursor
        self._cursor = (self._cursor + 1) % self.n_actions
        return int(idx)

    def on_episode_end(self) -> None:
        self._cursor = 0
