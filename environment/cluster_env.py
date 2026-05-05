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
        toggle_penalty: float = config.TOGGLE_PENALTY,
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
        self.toggle_penalty = toggle_penalty
        self.cluster_type = cluster_type
        self.cluster_seed = cluster_seed
        self.render_mode = render_mode

        # Observation per server: [cpu_util, mem_util, p_max_norm, cpu_cap_norm,
        # is_asleep, wakeup_norm]
        # — first four expose tier identity; last two expose lifecycle state
        # so the agent can decide to wake/sleep servers.
        obs_dim = 6 * num_servers + 4 * job_queue_size
        self.observation_space = spaces.Box(
            low=0.0, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )

        # Action: Discrete(K * N + 1 + N).
        #   a in [0, K*N)             : assign queue[k = a // N] to server[n = a % N]
        #   a == K * N                : "wait" — no dispatch this step
        #   a in [K*N+1, K*N+1+N)     : toggle sleep/wake on server (a - K*N - 1)
        #     • awake & idle  → request_sleep
        #     • asleep        → request_wake (begins WAKEUP_DELAY warm-up)
        # The wait action is gated to "no dispatch is legal"; sleep/wake are
        # always legal whenever physically valid (idle awake / asleep), so the
        # agent can sleep underused servers even when other dispatches are open.
        self.n_actions = num_servers * job_queue_size + 1 + num_servers
        self.wait_action = num_servers * job_queue_size
        self.sleep_action_base = num_servers * job_queue_size + 1
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
        if action == self.wait_action:
            pass  # no-op
        elif action >= self.sleep_action_base:
            # Toggle sleep on server (action - sleep_action_base).
            n = action - self.sleep_action_base
            server = self.servers[n]
            if server.can_request_sleep:
                server.request_sleep()
                reward -= self.toggle_penalty
            elif server.can_request_wake:
                server.request_wake()
                # Wake is free: we want the agent to recover capacity
                # immediately when load arrives. Asymmetric cost (sleep paid,
                # wake free) still prevents flicker because back-to-back
                # sleep→wake→sleep pays the sleep penalty twice.
            else:
                # Toggle requested but server has running jobs and is awake,
                # or is mid-wake. Treat as invalid action.
                reward += self.invalid_action_penalty
        else:
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
        # Server.power_draw() handles the three lifecycle states (asleep ⇒
        # standby fraction of p_idle, waking ⇒ full p_idle, awake ⇒ full curve).
        cluster_power = sum(s.power_draw() for s in self.servers)
        self.total_power += cluster_power

        # Active power = power above each server's *own* idle floor for awake
        # servers. Sleeping servers contribute 0 to active power AND drop the
        # cluster_power floor, so the agent now has a real lever: putting a
        # server to sleep saves ~0.95 · p_idle every step it stays asleep.
        active_ceiling = sum(s.p_max - s.p_idle for s in self.servers)
        active_power = sum(
            max(0.0, s.power_draw() - s.p_idle)
            for s in self.servers
            if not s.is_asleep
        )
        normalized_active = active_power / active_ceiling if active_ceiling > 0 else 0.0

        # Idle-floor pressure: only penalize servers that are awake AND
        # underutilized (i.e. could plausibly be put to sleep). Weighting by
        # (1 − utilization) means a fully-loaded server contributes 0 to this
        # term, while a zero-load awake server pays its full p_idle. Without
        # this, the agent gets the same penalty for a busy server as for an
        # idle one and learns to over-sleep — driving SLA violations up while
        # only marginally reducing power.
        sleepable_idle = sum(
            s.p_idle * (1.0 - max(0.0, min(1.0, s.cpu_utilization)))
            for s in self.servers
            if not s.is_asleep
        )
        full_idle_total = sum(s.p_idle for s in self.servers) or 1.0
        normalized_idle = sleepable_idle / full_idle_total  # ∈ [0, 1]

        # SLA term: discrete violations divided by N_SERVERS (boundary signal)
        # plus continuous queue_pressure (within-deadline gradient).
        sla_signal = (
            sla_violations_this_step / max(1, self.num_servers)
            + queue_pressure / max(1, self.num_servers)
        )
        # Power term combines active draw (variable) and sleepable-idle
        # footprint (the lever sleep gives us). Weighted 80/20 so the active
        # signal stays louder than the sleep bonus — otherwise the agent
        # over-sleeps even productive servers.
        power_term = 0.8 * normalized_active + 0.2 * normalized_idle
        reward += -(
            self.reward_alpha * power_term
            + self.reward_beta * sla_signal
        )

        # --- Check termination ---
        terminated = False
        truncated = self.current_step >= self.episode_length

        obs = self._get_obs()
        info = self._get_info()
        info["step_power"] = cluster_power
        info["step_active_power"] = active_power
        info["step_sla_violations"] = sla_violations_this_step
        info["n_asleep"] = sum(1 for s in self.servers if s.is_asleep)
        info["n_waking"] = sum(
            1 for s in self.servers
            if (not s.is_asleep) and s.wakeup_remaining > 0
        )

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        # Server features: [cpu_util, mem_util, p_max_norm, cpu_cap_norm,
        #                   is_asleep, wakeup_norm].
        # _norm values are relative to the env defaults so the canonical
        # "standard tier" = 1.0. wakeup_norm = wakeup_remaining / WAKEUP_DELAY
        # so the agent sees how soon a waking server becomes available.
        server_features = []
        wakeup_denom = max(1, config.SERVER_WAKEUP_DELAY)
        for s in self.servers:
            p_max_norm = s.p_max / config.P_MAX if config.P_MAX > 0 else 1.0
            cap_norm = s.cpu_capacity / config.SERVER_CPU_CAPACITY if config.SERVER_CPU_CAPACITY > 0 else 1.0
            is_asleep_f = 1.0 if s.is_asleep else 0.0
            wakeup_norm = s.wakeup_remaining / wakeup_denom
            server_features.extend([
                s.cpu_utilization, s.mem_utilization,
                p_max_norm, cap_norm,
                is_asleep_f, wakeup_norm,
            ])

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
