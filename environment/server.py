from __future__ import annotations

from environment.job import Job
from environment import config


class Server:
    def __init__(
        self,
        server_id: int,
        cpu_capacity: float = config.SERVER_CPU_CAPACITY,
        mem_capacity: float = config.SERVER_MEM_CAPACITY,
    ):
        self.server_id = server_id
        self.cpu_capacity = cpu_capacity
        self.mem_capacity = mem_capacity
        self.running_jobs: list[tuple[Job, int]] = []  # (job, remaining_timesteps)
        self._cpu_used = 0.0
        self._mem_used = 0.0

    def reset(self) -> None:
        self.running_jobs.clear()
        self._cpu_used = 0.0
        self._mem_used = 0.0

    @property
    def cpu_utilization(self) -> float:
        return self._cpu_used / self.cpu_capacity if self.cpu_capacity > 0 else 0.0

    @property
    def mem_utilization(self) -> float:
        return self._mem_used / self.mem_capacity if self.mem_capacity > 0 else 0.0

    def can_fit(self, job: Job) -> bool:
        return (
            self._cpu_used + job.cpu_request <= self.cpu_capacity + 1e-9
            and self._mem_used + job.mem_request <= self.mem_capacity + 1e-9
        )

    def assign(self, job: Job) -> None:
        if not self.can_fit(job):
            raise ValueError(
                f"Server {self.server_id} cannot fit job {job.job_id}: "
                f"cpu {self._cpu_used}+{job.cpu_request} > {self.cpu_capacity} or "
                f"mem {self._mem_used}+{job.mem_request} > {self.mem_capacity}"
            )
        self.running_jobs.append((job, job.duration))
        self._cpu_used += job.cpu_request
        self._mem_used += job.mem_request

    def step(self) -> list[Job]:
        """Advance one timestep. Returns list of completed jobs."""
        completed = []
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
