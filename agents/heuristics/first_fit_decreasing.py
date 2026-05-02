from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent


class FirstFitDecreasingAgent(BaseAgent):
    """Pick the LARGEST-CPU job from the queue and pack it onto the most-
    loaded fitting server (best-fit decreasing). Power-minimizing baseline:
    big jobs go first to keep idle servers near zero.

    Layout: state = [4*N server features][4*K job features (cpu, mem, dur, wait)].
    """

    def __init__(self, n_servers: int, queue_size: int):
        self.n_servers = n_servers
        self.queue_size = queue_size
        self.wait_action = n_servers * queue_size

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        N = self.n_servers
        K = self.queue_size
        server_block = state[: 4 * N].reshape(N, 4)
        cpu_utils = server_block[:, 0]
        job_block = state[4 * N : 4 * N + 4 * K].reshape(K, 4)
        cpu_reqs = job_block[:, 0]

        # Prefer large jobs (decreasing order); within a job, prefer the most-
        # loaded server that can still fit.
        best_a = self.wait_action
        best_score = (float("-inf"), float("-inf"))
        for k in range(K):
            for n in range(N):
                a = k * N + n
                if not action_mask[a]:
                    continue
                score = (float(cpu_reqs[k]), float(cpu_utils[n]))
                if score > best_score:
                    best_score = score
                    best_a = a
        return int(best_a)
