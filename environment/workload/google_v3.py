from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from environment import config
from environment.job import Job
from environment.workload.base import WorkloadGenerator


class GoogleV3WorkloadGenerator(WorkloadGenerator):
    """Parser for Google Cluster Data v3 (2019) instance_events JSON/CSV files.

    Expected columns (v3 schema):
        time: microseconds
        type: event type (0=SUBMIT, ...)
        collection_id: job/collection identifier
        instance_index: task index within collection
        resource_request.cpus: CPU request (normalized)
        resource_request.memory: Memory request (normalized)

    We filter for SUBMIT events and normalize resources.
    """

    EVENT_SUBMIT = 0
    EVENT_FINISH = 4
    EVENT_FAIL = 5
    EVENT_KILL = 6

    def __init__(
        self,
        trace_dir: str | Path,
        timestep_duration_us: int = 300_000_000,
        sla_multiplier: float = config.SLA_MULTIPLIER,
        default_duration: int = 4,
        max_jobs: int | None = None,
    ):
        self.trace_dir = Path(trace_dir)
        self.timestep_duration_us = timestep_duration_us
        self.sla_multiplier = sla_multiplier
        self.default_duration = default_duration
        self.max_jobs = max_jobs

        self._jobs_by_timestep: dict[int, list[Job]] = {}
        self._max_timestep = 0
        self._current_timestep = 0
        self._loaded = False

    def _load(self) -> None:
        csv_files = sorted(self.trace_dir.glob("instance_events*.csv*"))
        json_files = sorted(self.trace_dir.glob("instance_events*.json*"))
        files = csv_files or json_files
        if not files:
            raise FileNotFoundError(
                f"No instance_events files found in {self.trace_dir}"
            )

        frames = []
        for f in files:
            if f.suffix in (".json", ".gz") and "json" in f.name:
                df = pd.read_json(f, lines=True)
            else:
                df = pd.read_csv(f)
            frames.append(df)
        data = pd.concat(frames, ignore_index=True)

        # Normalize column names (handle nested resource_request.cpus etc.)
        col_map = {}
        for col in data.columns:
            lower = col.lower().replace(" ", "_")
            if "cpu" in lower:
                col_map[col] = "cpu"
            elif "memory" in lower or "mem" in lower:
                col_map[col] = "mem"
            elif lower == "time":
                col_map[col] = "timestamp"
            elif lower == "type":
                col_map[col] = "event_type"
            elif "collection" in lower:
                col_map[col] = "collection_id"
            elif "instance" in lower and "index" in lower:
                col_map[col] = "instance_index"
        data = data.rename(columns=col_map)

        required = ["timestamp", "event_type", "cpu", "mem"]
        for r in required:
            if r not in data.columns:
                raise ValueError(f"Missing required column '{r}' in trace data. Available: {list(data.columns)}")

        submit_mask = data["event_type"] == self.EVENT_SUBMIT
        terminal_mask = data["event_type"].isin([self.EVENT_FINISH, self.EVENT_FAIL, self.EVENT_KILL])

        submits = data[submit_mask].copy()
        submits = submits.dropna(subset=["cpu", "mem"])
        submits["cpu"] = submits["cpu"].clip(0.01, 1.0)
        submits["mem"] = submits["mem"].clip(0.01, 1.0)

        # Duration estimation from terminal events
        id_cols = ["collection_id", "instance_index"] if "collection_id" in data.columns and "instance_index" in data.columns else []
        if id_cols:
            terminal = data[terminal_mask].groupby(id_cols)["timestamp"].min().reset_index()
            terminal.rename(columns={"timestamp": "end_timestamp"}, inplace=True)
            submits = submits.merge(terminal, on=id_cols, how="left")
            has_end = submits["end_timestamp"].notna()
            submits["duration"] = self.default_duration
            submits.loc[has_end, "duration"] = np.maximum(
                1,
                ((submits.loc[has_end, "end_timestamp"] - submits.loc[has_end, "timestamp"]) / self.timestep_duration_us).astype(int),
            )
        else:
            submits["duration"] = self.default_duration

        min_ts = submits["timestamp"].min()
        submits["timestep"] = ((submits["timestamp"] - min_ts) / self.timestep_duration_us).astype(int)

        if self.max_jobs is not None:
            submits = submits.head(self.max_jobs)

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
