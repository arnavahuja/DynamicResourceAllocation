from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from environment.job import Job
from environment.workload.base import WorkloadGenerator


class GoogleV2WorkloadGenerator(WorkloadGenerator):
    """Parser for Google Cluster Data v2 (2011) task_events CSV files.

    Expected CSV columns (standard v2 schema):
        0: timestamp (microseconds)
        1: missing_info
        2: job_id
        3: task_index
        4: machine_id
        5: event_type  (0=SUBMIT, 1=SCHEDULE, 2=EVICT, ...)
        6: user
        7: scheduling_class
        8: priority
        9: cpu_request  (normalized to machine capacity, [0,1])
       10: memory_request (normalized to machine capacity, [0,1])
       11: disk_space_request
       12: different_machines_restriction

    We filter for SUBMIT events (event_type=0) and extract resource requests.
    Job duration is estimated from the difference between SUBMIT and a terminal
    event if available, otherwise a default is used.
    """

    # Column indices in the v2 task_events CSV (no header)
    COL_TIMESTAMP = 0
    COL_JOB_ID = 2
    COL_TASK_INDEX = 3
    COL_EVENT_TYPE = 5
    COL_CPU_REQUEST = 9
    COL_MEM_REQUEST = 10

    EVENT_SUBMIT = 0
    EVENT_FINISH = 4
    EVENT_FAIL = 5
    EVENT_KILL = 6

    def __init__(
        self,
        trace_dir: str | Path,
        timestep_duration_us: int = 300_000_000,  # 5 minutes per timestep
        sla_multiplier: float = 2.0,
        default_duration: int = 4,
        max_jobs: int | None = None,
    ):
        """
        Args:
            trace_dir: Directory containing task_events CSV part files.
            timestep_duration_us: Microseconds per simulation timestep.
            sla_multiplier: SLA deadline = duration * sla_multiplier.
            default_duration: Default job duration if no finish event found.
            max_jobs: Optional cap on total jobs to load.
        """
        self.trace_dir = Path(trace_dir)
        self.timestep_duration_us = timestep_duration_us
        self.sla_multiplier = sla_multiplier
        self.default_duration = default_duration
        self.max_jobs = max_jobs

        self._jobs_by_timestep: dict[int, list[Job]] = {}
        self._max_timestep = 0
        self._pointer = 0
        self._loaded = False

    def _load(self) -> None:
        """Load and parse task_events CSV files."""
        csv_files = sorted(self.trace_dir.glob("task_events*.csv*"))
        if not csv_files:
            raise FileNotFoundError(
                f"No task_events CSV files found in {self.trace_dir}"
            )

        frames = []
        for f in csv_files:
            df = pd.read_csv(
                f,
                header=None,
                usecols=[
                    self.COL_TIMESTAMP,
                    self.COL_JOB_ID,
                    self.COL_TASK_INDEX,
                    self.COL_EVENT_TYPE,
                    self.COL_CPU_REQUEST,
                    self.COL_MEM_REQUEST,
                ],
                names=["timestamp", "job_id", "task_index", "event_type", "cpu", "mem"],
            )
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)

        # Build duration lookup from finish/fail/kill events
        terminal_mask = data["event_type"].isin(
            [self.EVENT_FINISH, self.EVENT_FAIL, self.EVENT_KILL]
        )
        submit_mask = data["event_type"] == self.EVENT_SUBMIT
        submits = data[submit_mask].copy()

        terminal = data[terminal_mask].copy()
        terminal = terminal.groupby(["job_id", "task_index"])["timestamp"].min().reset_index()
        terminal.rename(columns={"timestamp": "end_timestamp"}, inplace=True)

        submits = submits.merge(terminal, on=["job_id", "task_index"], how="left")

        # Drop rows with missing resource requests
        submits = submits.dropna(subset=["cpu", "mem"])
        submits["cpu"] = submits["cpu"].clip(0.01, 1.0)
        submits["mem"] = submits["mem"].clip(0.01, 1.0)

        # Compute duration in timesteps
        has_end = submits["end_timestamp"].notna()
        submits.loc[has_end, "duration_us"] = (
            submits.loc[has_end, "end_timestamp"] - submits.loc[has_end, "timestamp"]
        )
        submits["duration"] = self.default_duration
        submits.loc[has_end, "duration"] = np.maximum(
            1, (submits.loc[has_end, "duration_us"] / self.timestep_duration_us).astype(int)
        )

        # Convert arrival timestamps to timesteps
        min_ts = submits["timestamp"].min()
        submits["timestep"] = ((submits["timestamp"] - min_ts) / self.timestep_duration_us).astype(int)

        if self.max_jobs is not None:
            submits = submits.head(self.max_jobs)

        # Build per-timestep job lists
        self._jobs_by_timestep = {}
        for idx, row in submits.iterrows():
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

    def reset(self) -> None:
        if not self._loaded:
            self._load()
        self._pointer = 0

    def get_next_jobs(self, timestep: int) -> list[Job]:
        if not self._loaded:
            self._load()
        # Update arrival_time to match the simulation timestep
        raw_jobs = self._jobs_by_timestep.get(timestep, [])
        for job in raw_jobs:
            job.arrival_time = timestep
        return raw_jobs

    def is_exhausted(self) -> bool:
        return self._loaded and self._pointer > self._max_timestep
