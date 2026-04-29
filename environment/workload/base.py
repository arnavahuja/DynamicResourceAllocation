from __future__ import annotations

from abc import ABC, abstractmethod

from environment.job import Job


class WorkloadGenerator(ABC):
    """Abstract base class for workload generators / trace parsers."""

    @abstractmethod
    def reset(self) -> None:
        """Reset generator state (e.g., rewind trace pointer)."""

    @abstractmethod
    def get_next_jobs(self, timestep: int) -> list[Job]:
        """Return jobs arriving at the given timestep."""

    @abstractmethod
    def is_exhausted(self) -> bool:
        """Return True if no more jobs will arrive."""
