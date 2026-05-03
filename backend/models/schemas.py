from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

AgentName = Literal["dqn", "ppo", "agentic", "round_robin", "sjf", "ffd"]
ClusterType = Literal["homogeneous", "heterogeneous"]
TraceFamily = Literal["alibaba", "google_v2", "google_v3", "google_v2_sampled"]
RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class TrainRequest(BaseModel):
    agent: AgentName = "dqn"
    n_servers: int = Field(10, ge=1, le=200)
    episodes: int = Field(200, ge=1, le=100_000)
    episode_length: int = Field(500, ge=10, le=10_000)
    total_steps: int = Field(50_000, ge=100, le=10_000_000,
                             description="PPO only — ignored otherwise")
    use_real_traces: bool = False
    trace_family: TraceFamily = "alibaba"
    cluster_type: ClusterType = "homogeneous"
    alpha: float = Field(1.0, ge=0.0, le=100.0,
                         description="Power weight in reward")
    beta: float = Field(50.0, ge=0.0, le=1000.0,
                        description="SLA weight in reward")
    seed: int = 0
    eval_episodes: int = Field(10, ge=0, le=500)
    # Train/test seed pool sizes for ML-style generalization eval.
    # n_train_seeds=1, n_test_seeds=0 → legacy single-trajectory training.
    # n_train_seeds=N, n_test_seeds=M → train uniformly across N seeds,
    # eval on M held-out seeds.
    n_train_seeds: int = Field(1, ge=1, le=10_000)
    n_test_seeds: int = Field(0, ge=0, le=1000)


class TrainResponse(BaseModel):
    run_id: str


class RunStatusResponse(BaseModel):
    run_id: str
    status: RunStatus
    progress: float = Field(0.0, ge=0.0, le=1.0)
    eta_seconds: float | None = None
    current_episode: int = 0
    total_episodes: int = 0
    last_reward: float | None = None
    error: str | None = None


class ExperimentSummary(BaseModel):
    run_id: str
    agent: str
    status: RunStatus
    created_at: str
    n_servers: int
    episodes: int
    mean_reward_last10: float | None = None
    mean_power: float | None = None
    sla_violation_rate: float | None = None


class EpisodePoint(BaseModel):
    episode: int
    reward: float
    power: float
    sla_violations: int
    steps: int


class ExperimentResults(BaseModel):
    summary: ExperimentSummary
    config: dict
    episodes: list[EpisodePoint]
    eval: dict | None = None


class SimulateRequest(BaseModel):
    agent: AgentName = "round_robin"
    checkpoint_path: str | None = None
    n_servers: int = 10
    episode_length: int = 200
    seed: int = 0


class SimulateStep(BaseModel):
    step: int
    action: int
    reward: float
    power: float
    sla_violations: int
    server_utilizations: list[float]
    queue_length: int


class SimulateResponse(BaseModel):
    total_reward: float
    total_power: float
    total_sla: int
    steps: list[SimulateStep]
