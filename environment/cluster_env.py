from __future__ import annotations

from collections import deque

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import config
from environment.fleet import build_fleet
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
        cluster_type: str = "homogeneous",
        cluster_seed: int = 0,
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
        self.cluster_type = cluster_type
        self.cluster_seed = cluster_seed
        self.render_mode = render_mode

        # Observation per server: [cpu_util, mem_util, p_max_norm, cpu_cap_norm]
        # — p_max_norm and cpu_cap_norm expose server identity (tier) so the
        # agent can prefer efficient servers in heterogeneous clusters. In
        # homogeneous clusters both are constant and the agent learns to
        # ignore them.
        obs_dim = 4 * num_servers + 4 * job_queue_size
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: Discrete(K * N + 1).
        #   a in [0, K*N): k = a // N (queue position), n = a % N (server idx)
        #     → assign queue[k] to server[n]
        #   a == K * N → "wait", dispatch nothing this step (no-op)
        # This lets the agent choose WHICH job (not just where), and gives it
        # the option of skipping a step when no good (job, server) pair exists.
        self.n_actions = num_servers * job_queue_size + 1
        self.wait_action = num_servers * job_queue_size
        self.action_space = spaces.Discrete(self.n_actions)

        # Internal state
        self.servers: list[Server] = []
        self.job_queue: deque[Job] = deque()
        self.current_step = 0
        self.total_power = 0.0
        self.total_sla_violations = 0
        self.total_jobs_completed = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.servers = build_fleet(
            num_servers=self.num_servers,
            cluster_type=self.cluster_type,
            seed=self.cluster_seed,
        )
        self.job_queue = deque()
        self.current_step = 0
        self.total_power = 0.0
        self.total_sla_violations = 0
        self.total_jobs_completed = 0

        # `options={"workload_seed": int}` lets the trainer pick which seed
        # this episode samples from a pool. None → workload uses its own seed.
        workload_seed = (options or {}).get("workload_seed")
        try:
            self.workload.reset(seed=workload_seed)
        except TypeError:
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

        # --- Decode action and dispatch ---
        # action == wait_action → no-op for this step.
        # else: k = action // N (queue position), n = action % N (server idx).
        if action != self.wait_action:
            k = action // self.num_servers
            n = action % self.num_servers
            if k < len(self.job_queue):
                job = self.job_queue[k]
                server = self.servers[n]
                if server.can_fit(job):
                    # Remove from queue at position k (deque doesn't have pop(i)).
                    new_queue = deque()
                    for i, j in enumerate(self.job_queue):
                        if i != k:
                            new_queue.append(j)
                    self.job_queue = new_queue
                    server.assign(job)
                else:
                    reward += self.invalid_action_penalty
            else:
                # Tried to dispatch a queue position that doesn't exist.
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

        # --- Fix 2: continuous SLA pressure ---
        # The discrete violations counted above only fire at expiry boundaries.
        # That's terrible credit assignment — actions taken N steps before the
        # expiry got no signal. Add a continuous per-step pressure term that
        # grows with the *severity* of overdue queued jobs (sum of normalized
        # overdue ratios). This gives the agent gradients during the entire
        # buildup phase, when its actions actually matter.
        queue_pressure = 0.0
        for job in self.job_queue:
            wait = self.current_step - job.arrival_time
            if wait > 0 and job.sla_deadline > 0:
                # Headroom remaining (1.0 = just arrived, 0.0 = at deadline,
                # negative = overdue). Pressure is max(0, -headroom_norm).
                headroom_norm = (job.sla_deadline - wait) / job.sla_deadline
                if headroom_norm < 0:
                    queue_pressure += -headroom_norm  # how-overdue, ratio

        # --- Pull new arrivals ---
        new_jobs = self.workload.get_next_jobs(self.current_step)
        self.job_queue.extend(new_jobs)

        # --- Compute reward ---
        # Per-server power so heterogeneous clusters reflect tier differences.
        cluster_power = sum(
            compute_power(s.cpu_utilization, s.p_idle, s.p_max, s.power_alpha)
            for s in self.servers
        )
        self.total_power += cluster_power

        # Fix 1: reward uses ACTIVE power (above idle), not total cluster power.
        # The idle floor is a constant the agent cannot influence — including it
        # in the reward just adds a per-step bias that drowns out the gradient
        # from the part the agent CAN control (which server to assign).
        #   active_power      = Σ (p_max_i - p_idle_i) · u_i^α
        #   active_ceiling    = Σ (p_max_i - p_idle_i)
        # Normalized active power is in [0, 1] regardless of fleet composition.
        active_ceiling = sum(s.p_max - s.p_idle for s in self.servers)
        idle_total = sum(s.p_idle for s in self.servers)
        active_power = max(0.0, cluster_power - idle_total)
        normalized_active = active_power / active_ceiling if active_ceiling > 0 else 0.0

        # SLA term: discrete violations divided by N_SERVERS (boundary signal)
        # plus continuous queue_pressure (within-deadline gradient).
        sla_signal = (
            sla_violations_this_step / max(1, self.num_servers)
            + queue_pressure / max(1, self.num_servers)
        )
        reward += -(
            self.reward_alpha * normalized_active
            + self.reward_beta * sla_signal
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
        # Server features: [cpu_util, mem_util, p_max_norm, cpu_cap_norm].
        # _norm values are relative to the env defaults so the canonical
        # "standard tier" = 1.0.
        server_features = []
        for s in self.servers:
            p_max_norm = s.p_max / config.P_MAX if config.P_MAX > 0 else 1.0
            cap_norm = s.cpu_capacity / config.SERVER_CPU_CAPACITY if config.SERVER_CPU_CAPACITY > 0 else 1.0
            server_features.extend([s.cpu_utilization, s.mem_utilization, p_max_norm, cap_norm])

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
