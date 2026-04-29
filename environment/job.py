from dataclasses import dataclass


@dataclass
class Job:
    job_id: int
    cpu_request: float   # fraction of server capacity [0, 1]
    mem_request: float   # fraction of server capacity [0, 1]
    duration: int        # timesteps to complete
    arrival_time: int    # timestep when job entered the queue
    sla_deadline: int    # max allowed latency (arrival to completion)

    @property
    def is_valid(self) -> bool:
        return (
            0 < self.cpu_request <= 1.0
            and 0 < self.mem_request <= 1.0
            and self.duration > 0
            and self.sla_deadline > 0
        )
