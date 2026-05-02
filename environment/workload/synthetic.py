from __future__ import annotations

import numpy as np

from environment.job import Job
from environment.workload.base import WorkloadGenerator
from environment import config


class SyntheticWorkloadGenerator(WorkloadGenerator):
    """Poisson arrivals with uniform resource demands.

    Useful for testing and debugging without real trace data.
    """

    def __init__(
        self,
        arrival_rate: float = 0.8,
        cpu_range: tuple[float, float] = (0.05, 0.3),
        mem_range: tuple[float, float] = (0.05, 0.3),
        duration_range: tuple[int, int] = (2, 10),
        sla_multiplier: float = 1.6,
        max_jobs: int = 5000,
        seed: int | None = None,
    ):
        self.arrival_rate = arrival_rate
        self.cpu_range = cpu_range
        self.mem_range = mem_range
        self.duration_range = duration_range
        self.sla_multiplier = sla_multiplier
        self.max_jobs = max_jobs
        self._seed = seed
        # rng is re-seeded on every reset() so every episode sees the *same*
        # arrival pattern. This isolates policy effects from arrival noise —
        # the dominant variance source on Poisson workloads.
        self.rng = np.random.default_rng(seed)
        self._job_counter = 0
        self._total_generated = 0

    def reset(self, seed: int | None = None) -> None:
        """Reset arrival counters and re-seed the RNG.

        If `seed` is provided, the workload starts from that seed for this
        episode; otherwise it falls back to the seed passed at construction
        time. This is what enables the "train on a pool of seeds, test on
        a held-out pool" protocol.
        """
        self._job_counter = 0
        self._total_generated = 0
        s = seed if seed is not None else self._seed
        self.rng = np.random.default_rng(s)

    def get_next_jobs(self, timestep: int) -> list[Job]:
        if self._total_generated >= self.max_jobs:
            return []
        n_arrivals = self.rng.poisson(self.arrival_rate)
        n_arrivals = min(n_arrivals, self.max_jobs - self._total_generated)
        jobs = []
        for _ in range(n_arrivals):
            cpu = float(self.rng.uniform(*self.cpu_range))
            mem = float(self.rng.uniform(*self.mem_range))
            dur = int(self.rng.integers(self.duration_range[0], self.duration_range[1] + 1))
            sla = max(dur, int(dur * self.sla_multiplier))
            job = Job(
                job_id=self._job_counter,
                cpu_request=cpu,
                mem_request=mem,
                duration=dur,
                arrival_time=timestep,
                sla_deadline=sla,
            )
            jobs.append(job)
            self._job_counter += 1
            self._total_generated += 1
        return jobs

    def is_exhausted(self) -> bool:
        return self._total_generated >= self.max_jobs
