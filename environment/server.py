from __future__ import annotations

from environment.job import Job
from environment import config


class Server:
    """A cluster server with three lifecycle states:
    - awake & available  : can accept and run jobs
    - waking             : just got a wake request; unavailable for `wakeup_remaining` steps
    - asleep             : drawing only standby power; cannot accept jobs

    Sleep is the lever that lets the agent actually reduce cluster power
    below the always-on idle floor — without it, ~⅔ of total power is
    un-controllable regardless of policy.
    """

    def __init__(
        self,
        server_id: int,
        cpu_capacity: float = config.SERVER_CPU_CAPACITY,
        mem_capacity: float = config.SERVER_MEM_CAPACITY,
        p_idle: float | None = None,
        p_max: float | None = None,
        power_alpha: float | None = None,
    ):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.p_idle = p_idle if p_idle is not None else config.P_IDLE
        self.p_max = p_max if p_max is not None else config.P_MAX
        self.power_alpha = power_alpha if power_alpha is not None else config.POWER_ALPHA
        self.running_jobs: list[tuple[Job, int]] = []  # (job, remaining_timesteps)
        self._cpu_used = 0.0
        self._mem_used = 0.0
        self.is_asleep: bool = False
        self.wakeup_remaining: int = 0

    def reset(self) -> None:
        self.running_jobs.clear()
        self._cpu_used = 0.0
        self._mem_used = 0.0
        self.is_asleep = False
        self.wakeup_remaining = 0

    @property
    def cpu_utilization(self) -> float:
        return self._cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

    @property
    def mem_utilization(self) -> float:
        return self._mem_used / self.mem_capacity if self.mem_capacity > 0 else 0.0

    @property
    def is_available(self) -> bool:
        """Awake AND finished warming up."""
        return (not self.is_asleep) and self.wakeup_remaining == 0

    @property
    def can_request_sleep(self) -> bool:
        """A server can be put to sleep only when it is currently awake,
        not in the middle of waking, and idle (no running jobs)."""
        return (
            (not self.is_asleep)
            and self.wakeup_remaining == 0
            and len(self.running_jobs) == 0
        )

    @property
    def can_request_wake(self) -> bool:
        return self.is_asleep

    def request_sleep(self) -> bool:
        if not self.can_request_sleep:
            return False
        self.is_asleep = True
        self.wakeup_remaining = 0
        return True

    def request_wake(self) -> bool:
        if not self.can_request_wake:
            return False
        self.is_asleep = False
        self.wakeup_remaining = config.SERVER_WAKEUP_DELAY
        return True

    def can_fit(self, job: Job) -> bool:
        if not self.is_available:
            return False
        return (
            self._cpu_used + job.cpu_request <= self.cpu_capacity + 1e-9
            and self._mem_used + job.mem_request <= self.mem_capacity + 1e-9
        )

    def assign(self, job: Job) -> None:
        if not self.can_fit(job):
            raise ValueError(
                f"Server {self.server_id} cannot fit job {job.job_id}: "
                f"cpu {self._cpu_used}+{job.cpu_request} > {self.cpu_capacity} or "
                f"mem {self._mem_used}+{job.mem_request} > {self.mem_capacity} "
                f"or unavailable (asleep={self.is_asleep}, "
                f"wakeup_remaining={self.wakeup_remaining})"
            )
        self.running_jobs.append((job, job.duration))
        self._cpu_used += job.cpu_request
        self._mem_used += job.mem_request

    def step(self) -> list[Job]:
        """Advance one timestep. Returns list of completed jobs."""
        # Decrement wakeup counter regardless of whether jobs are running.
        if self.wakeup_remaining > 0:
            self.wakeup_remaining -= 1

        completed: list[Job] = []
        # Asleep servers cannot run anything; running_jobs should already be empty.
        if self.is_asleep:
            return completed

        still_running = []
        for job, remaining in self.running_jobs:
            remaining -= 1
            if remaining <= 0:
                completed.append(job)
                self._cpu_used -= job.cpu_request
                self._mem_used -= job.mem_request
            else:
                still_running.append((job, remaining))
        self.running_jobs = still_running
        # Clamp to avoid floating point drift
        self._cpu_used = max(0.0, self._cpu_used)
        self._mem_used = max(0.0, self._mem_used)
        return completed

    def power_draw(self) -> float:
        """Per-step power consumption accounting for sleep / wake state.

        - asleep              : SLEEP_STANDBY_FACTOR · p_idle (small).
        - waking (waiting)    : full p_idle (warming up, no useful work).
        - awake & idle        : p_idle.
        - awake & running     : p_idle + (p_max − p_idle) · u^α.
        """
        if self.is_asleep:
            return config.SLEEP_STANDBY_FACTOR * self.p_idle
        u = max(0.0, min(1.0, self.cpu_utilization))
        return self.p_idle + (self.p_max - self.p_idle) * (u ** self.power_alpha)
