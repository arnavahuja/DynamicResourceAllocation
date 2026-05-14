from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from environment import config
from environment.job import Job
from environment.workload.base import WorkloadGenerator
from environment.workload.google_v2 import GoogleV2WorkloadGenerator


class TraceSampledWorkloadGenerator(WorkloadGenerator):
    """Hybrid workload: real-trace job-spec distribution + Poisson timing.

    Real-trace replay saturates a small study cluster because production
    traces are bursty and sized for ~10K-machine clusters. This generator
    keeps the *empirical* (cpu, mem, duration) distribution of the trace
    but drives arrivals with a synthetic Poisson process — so the workload
    is realistic in shape but tunable in load.
    """

    def __init__(
        self,
        trace_dir: str | Path,
        arrival_rate: float = config.SYNTHETIC_ARRIVAL_RATE,
        sla_multiplier: float = config.SLA_MULTIPLIER,
        max_jobs: int = 5000,
        seed: int | None = None,
        source: str = "google_v2",
    ):
        self.trace_dir = Path(trace_dir)
        self.arrival_rate = arrival_rate
        self.sla_multiplier = sla_multiplier
        self.max_jobs = max_jobs
        self._seed = seed
        self.source = source

        self._templates: list[tuple[float, float, int]] = []
        self._loaded = False
        self.rng = np.random.default_rng(seed)
        self._job_counter = 0
        self._total_generated = 0

    def _load_templates(self) -> None:
        """Walk the trace once with the underlying parser, harvest job specs."""
        if self.source != "google_v2":
            raise NotImplementedError(
                f"TraceSampledWorkloadGenerator currently only supports "
                f"source='google_v2', got {self.source!r}"
            )
        parser = GoogleV2WorkloadGenerator(trace_dir=self.trace_dir, max_jobs=None)
        parser.reset()
        templates: list[tuple[float, float, int]] = []
        for jobs in parser._jobs_by_timestep.values():
            for j in jobs:
                templates.append((j.cpu_request, j.mem_request, j.duration))
        if not templates:
            raise RuntimeError(f"No job templates extracted from {self.trace_dir}")
        self._templates = templates
        self._loaded = True

    def reset(self, seed: int | None = None) -> None:
        if not self._loaded:
            self._load_templates()
        self._job_counter = 0
        self._total_generated = 0
        s = seed if seed is not None else self._seed
        self.rng = np.random.default_rng(s)

    def get_next_jobs(self, timestep: int) -> list[Job]:
        if not self._loaded:
            self._load_templates()
        if self._total_generated >= self.max_jobs:
            return []
        n_arrivals = self.rng.poisson(self.arrival_rate)
        n_arrivals = min(n_arrivals, self.max_jobs - self._total_generated)
        jobs = []
        for _ in range(n_arrivals):
            idx = int(self.rng.integers(0, len(self._templates)))
            cpu, mem, dur = self._templates[idx]
            sla = max(int(dur), int(dur * self.sla_multiplier))
            job = Job(
                job_id=self._job_counter,
                cpu_request=float(cpu),
                mem_request=float(mem),
                duration=int(dur),
                arrival_time=timestep,
                sla_deadline=sla,
            )
            jobs.append(job)
            self._job_counter += 1
            self._total_generated += 1
        return jobs

    def is_exhausted(self) -> bool:
        return self._total_generated >= self.max_jobs
