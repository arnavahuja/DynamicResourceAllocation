from __future__ import annotations

import numpy as np

from agents.base_agent import BaseAgent


class ShortestJobFirstAgent(BaseAgent):
    """Pick the SHORTEST-DURATION job from the queue and assign it to the
    least-loaded fitting server. Latency-minimizing baseline that exploits
    the new (job, server) action space.

    Reads job durations from the queue features in `state`. Layout (matches
    CloudClusterEnv._get_obs):
        state = [4*N server features][4*K job features (cpu, mem, dur, wait)]
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
        # Server cpu_utilization is feature index 0 of each server block.
        server_block = state[: 4 * N].reshape(N, 4)
        cpu_utils = server_block[:, 0]
        # Job duration is feature index 2 of each job block.
        job_block = state[4 * N : 4 * N + 4 * K].reshape(K, 4)
        durations = job_block[:, 2]  # already normalized by /10 in env

        # Score each (k, n) action: prefer short jobs on idle servers.
        # We pick the legal action with the smallest (duration, util) tuple.
        best_a = self.wait_action
        best_score = (float("inf"), float("inf"))
        for k in range(K):
            for n in range(N):
                a = k * N + n
                if not action_mask[a]:
                    continue
                score = (float(durations[k]), float(cpu_utils[n]))
                if score < best_score:
                    best_score = score
                    best_a = a
        return int(best_a)
