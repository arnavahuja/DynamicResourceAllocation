from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from environment import config
from environment.job import Job
from environment.workload.base import WorkloadGenerator


class AlibabaWorkloadGenerator(WorkloadGenerator):
    """Parser for Alibaba Cluster Trace 2018 batch_task CSV files.

    Expected CSV columns:
        task_name: unique task identifier
        inst_num: number of instances
        job_name: parent job identifier
        task_type: type of task
        status: task status
        start_time: seconds since trace start
        end_time: seconds since trace start
        plan_cpu: CPU cores requested (raw count, e.g., 100 = 1 core)
        plan_mem: memory requested (normalized, percentage)

    CPU is normalized by dividing by a reference machine capacity (96 cores).
    Memory is already normalized in the trace (percentage, divided by 100).
    """

    def __init__(
        self,
        trace_path: str | Path,
        timestep_duration_s: int = 300,  # 5 minutes per timestep
        sla_multiplier: float = config.SLA_MULTIPLIER,
        cpu_normalize_factor: float = 9600.0,  # 96 cores * 100 (Alibaba uses 100x units)
        max_jobs: int | None = None,
    ):
        self.trace_path = Path(trace_path)
        self.timestep_duration_s = timestep_duration_s
        self.sla_multiplier = sla_multiplier
        self.cpu_normalize_factor = cpu_normalize_factor
        self.max_jobs = max_jobs

        self._jobs_by_timestep: dict[int, list[Job]] = {}
        self._max_timestep = 0
        self._current_timestep = 0
        self._loaded = False

    def _load(self) -> None:
        files = sorted(self.trace_path.parent.glob(self.trace_path.name)) if "*" in str(self.trace_path) else [self.trace_path]
        if not any(f.exists() for f in files):
            raise FileNotFoundError(f"Trace file not found: {self.trace_path}")

        frames = []
        for f in files:
            if f.exists():
                df = pd.read_csv(f)
                frames.append(df)
        data = pd.concat(frames, ignore_index=True)

        # Normalize column names
        data.columns = [c.strip().lower() for c in data.columns]

        required = ["start_time", "plan_cpu", "plan_mem"]
        for r in required:
            if r not in data.columns:
                raise ValueError(f"Missing column '{r}'. Available: {list(data.columns)}")

        data = data.dropna(subset=["start_time", "plan_cpu", "plan_mem"])
        data = data[data["start_time"] >= 0]

        # Normalize CPU and memory to [0, 1]
        data["cpu"] = (data["plan_cpu"] / self.cpu_normalize_factor).clip(0.01, 1.0)
        data["mem"] = (data["plan_mem"] / 100.0).clip(0.01, 1.0)

        # Compute duration
        if "end_time" in data.columns:
            data["end_time"] = data["end_time"].fillna(data["start_time"] + self.timestep_duration_s * 4)
            data["duration_s"] = (data["end_time"] - data["start_time"]).clip(lower=self.timestep_duration_s)
            data["duration"] = np.maximum(1, (data["duration_s"] / self.timestep_duration_s).astype(int))
        else:
            data["duration"] = 4

        # Convert to timesteps
        min_ts = data["start_time"].min()
        data["timestep"] = ((data["start_time"] - min_ts) / self.timestep_duration_s).astype(int)

        if self.max_jobs is not None:
            data = data.head(self.max_jobs)

        self._jobs_by_timestep = {}
        for idx, row in data.iterrows():
            t = int(row["timestep"])
            dur = int(row["duration"])
            sla = max(dur, int(dur * self.sla_multiplier))
            job = Job(
                job_id=int(idx),
                cpu_request=float(row["cpu"]),
                mem_request=float(row["mem"]),
                duration=dur,
                arrival_time=t,
                sla_deadline=sla,
            )
            self._jobs_by_timestep.setdefault(t, []).append(job)

        self._max_timestep = max(self._jobs_by_timestep.keys()) if self._jobs_by_timestep else 0
        self._loaded = True

    def reset(self, seed: int | None = None) -> None:
        # `seed` ignored — trace replay is deterministic.
        if not self._loaded:
            self._load()
        self._current_timestep = 0

    def get_next_jobs(self, timestep: int) -> list[Job]:
        if not self._loaded:
            self._load()
        self._current_timestep = timestep
        raw_jobs = self._jobs_by_timestep.get(timestep, [])
        for job in raw_jobs:
            job.arrival_time = timestep
        return raw_jobs

    def is_exhausted(self) -> bool:
        return self._loaded and self._current_timestep > self._max_timestep
