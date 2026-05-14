from __future__ import annotations

import warnings
from typing import Any

from environment import config


class WandBLogger:
    """Thin wrapper around `wandb` that gracefully no-ops when:
      - the `wandb` package isn't installed, or
      - `WANDB_API_KEY` isn't set.

    Use as `with WandBLogger(...) as log: log.log({...})`.
    """

    def __init__(
        self,
        run_name: str,
        agent_name: str,
        cfg: dict[str, Any] | None = None,
        project: str | None = None,
        entity: str | None = None,
    ):
        self.run_name = run_name
        self.agent_name = agent_name
        self.cfg = cfg or {}
        self.project = project or config.WANDB_PROJECT
        self.entity = entity or config.WANDB_ENTITY or None
        self._wandb = None
        self._run = None
        self._enabled = False

    def __enter__(self) -> "WandBLogger":
        if not config.WANDB_API_KEY:
            warnings.warn("WANDB_API_KEY not set — wandb logging disabled.")
            return self
        try:
            import wandb  # type: ignore
        except ImportError:
            warnings.warn("wandb not installed — skipping wandb logging.")
            return self
        self._wandb = wandb
        self._run = wandb.init(
            project=self.project,
            entity=self.entity,
            name=self.run_name,
            config={"agent": self.agent_name, **self.cfg},
            reinit=True,
        )
        self._enabled = True
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._enabled and self._wandb is not None:
            self._wandb.finish()

    def log(self, metrics: dict[str, Any], step: int | None = None) -> None:
        if not self._enabled:
            return
        if step is not None:
            self._wandb.log(metrics, step=step)
        else:
            self._wandb.log(metrics)

    def log_table(self, name: str, columns: list[str], rows: list[list]) -> None:
        if not self._enabled:
            return
        table = self._wandb.Table(columns=columns, data=rows)
        self._wandb.log({name: table})
