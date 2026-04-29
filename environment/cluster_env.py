from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import config
from environment.job import Job
from environment.power_model import compute_power
from environment.server import Server
from environment.workload.base import WorkloadGenerator
from environment.workload.synthetic import SyntheticWorkloadGenerator


class CloudClusterEnv(gym.Env):
    """Gymnasium environment simulating a cloud data center cluster.

    An RL agent assigns the head-of-queue job to one of N servers at each
    timestep, optimizing for minimal power consumption while meeting SLA
    constraints on job latency.
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        num_servers: int = config.NUM_SERVERS,
        workload_generator: WorkloadGenerator | None = None,
        p_idle: float = config.P_IDLE,
        p_max: float = config.P_MAX,
        power_alpha: float = config.POWER_ALPHA,
        reward_alpha: float = config.REWARD_ALPHA,
        reward_beta: float = config.REWARD_BETA,
        job_queue_size: int = config.JOB_QUEUE_SIZE,
        episode_length: int = config.EPISODE_LENGTH,
        invalid_action_penalty: float = config.INVALID_ACTION_PENALTY,
        render_mode: str | None = None,
    ):
        super().__init__()

        self.num_servers = num_servers
        self.workload = workload_generator or SyntheticWorkloadGenerator(seed=42)
        self.p_idle = p_idle
        self.p_max = p_max
        self.power_alpha = power_alpha
        self.reward_alpha = reward_alpha
        self.reward_beta = reward_beta
        self.job_queue_size = job_queue_size
        self.episode_length = episode_length
        self.invalid_action_penalty = invalid_action_penalty
        self.render_mode = render_mode

        # Observation: [server_cpu_1, server_mem_1, ..., server_cpu_N, server_mem_N,
        #               job1_cpu, job1_mem, job1_dur, job1_wait, ..., jobK_cpu, ...]
        obs_dim = 2 * num_servers + 4 * job_queue_size
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: assign head-of-queue job to server k
        self.action_space = spaces.Discrete(num_servers)

        # Internal state
        self.servers: list[Server] = []
        self.job_queue: deque[Job] = deque()
        self.current_step = 0
        self.total_power = 0.0
        self.total_sla_violations = 0
        self.total_jobs_completed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.servers = [
            Server(server_id=i) for i in range(self.num_servers)
        ]
        self.job_queue = deque()
        self.current_step = 0
        self.total_power = 0.0
        self.total_sla_violations = 0
        self.total_jobs_completed = 0

        self.workload.reset()

        # Fill initial queue
        initial_jobs = self.workload.get_next_jobs(0)
        self.job_queue.extend(initial_jobs)

        obs = self._get_obs()
        info = self._get_info()
        return obs, info

    def step(self, action: int):
        assert self.action_space.contains(action), f"Invalid action {action}"

        reward = 0.0
        sla_violations_this_step = 0

        # --- Assign head-of-queue job to chosen server ---
        if len(self.job_queue) > 0:
            job = self.job_queue[0]
            server = self.servers[action]

            if server.can_fit(job):
                self.job_queue.popleft()
                server.assign(job)
            else:
                # Invalid assignment: job stays in queue, penalty applied
                reward += self.invalid_action_penalty

        # --- Advance simulation by one timestep ---
        self.current_step += 1

        for server in self.servers:
            completed = server.step()
            for job in completed:
                self.total_jobs_completed += 1
                # Check SLA at completion: was latency within deadline?
                latency = self.current_step - job.arrival_time
                if latency > job.sla_deadline:
                    sla_violations_this_step += 1

        # Check SLA violations for jobs still in queue (exceeded deadline while waiting)
        expired_jobs = []
        for job in self.job_queue:
            wait_time = self.current_step - job.arrival_time
            if wait_time > job.sla_deadline:
                sla_violations_this_step += 1
                expired_jobs.append(job)

        # Remove expired jobs from queue (they violated SLA and are dropped)
        for job in expired_jobs:
            self.job_queue.remove(job)
            self.total_jobs_completed += 1  # count as "processed" (failed)

        self.total_sla_violations += sla_violations_this_step

        # --- Pull new arrivals ---
        new_jobs = self.workload.get_next_jobs(self.current_step)
        self.job_queue.extend(new_jobs)

        # --- Compute reward ---
        cluster_power = sum(
            compute_power(s.cpu_utilization, self.p_idle, self.p_max, self.power_alpha)
            for s in self.servers
        )
        self.total_power += cluster_power

        # Normalize power to [0, 1] range for reward computation
        max_possible_power = self.p_max * self.num_servers
        normalized_power = cluster_power / max_possible_power

        reward += -(
            self.reward_alpha * normalized_power
            + self.reward_beta * sla_violations_this_step
        )

        # --- Check termination ---
        terminated = False
        truncated = self.current_step >= self.episode_length

        obs = self._get_obs()
        info = self._get_info()
        info["step_power"] = cluster_power
        info["step_sla_violations"] = sla_violations_this_step

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Server features: [cpu_util, mem_util] per server
        server_features = []
        for s in self.servers:
            server_features.extend([s.cpu_utilization, s.mem_utilization])

        # Job queue features: [cpu_req, mem_req, duration, wait_time] per visible job
        queue_features = []
        for i in range(self.job_queue_size):
            if i < len(self.job_queue):
                job = self.job_queue[i]
                wait_time = self.current_step - job.arrival_time
                queue_features.extend([
                    job.cpu_request,
                    job.mem_request,
                    job.duration / 10.0,  # normalize duration
                    wait_time / 10.0,     # normalize wait time
                ])
            else:
                queue_features.extend([0.0, 0.0, 0.0, 0.0])

        obs = np.array(server_features + queue_features, dtype=np.float32)
        return obs

    def _get_info(self) -> dict:
        utilizations = [s.cpu_utilization for s in self.servers]
        return {
            "power": self.total_power,
            "sla_violations": self.total_sla_violations,
            "jobs_completed": self.total_jobs_completed,
            "queue_length": len(self.job_queue),
            "mean_cpu_utilization": np.mean(utilizations) if utilizations else 0.0,
            "active_servers": sum(1 for u in utilizations if u > 0),
            "timestep": self.current_step,
        }

    def render(self):
        if self.render_mode != "human":
            return
        info = self._get_info()
        utils = [f"{s.cpu_utilization:.1%}" for s in self.servers]
        print(
            f"Step {self.current_step:4d} | "
            f"Queue: {len(self.job_queue):3d} | "
            f"Power: {info['power']:.0f}W | "
            f"SLA viol: {info['sla_violations']} | "
            f"Done: {info['jobs_completed']} | "
            f"Utils: [{', '.join(utils)}]"
        )
