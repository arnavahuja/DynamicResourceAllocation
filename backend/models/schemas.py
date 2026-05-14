from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from environment import config as env_config

AgentName = Literal["dqn", "ppo", "agentic", "round_robin", "sjf", "ffd", "cmdp"]
ClusterType = Literal["homogeneous", "heterogeneous"]
TraceFamily = Literal["alibaba", "google_v2", "google_v3", "google_v2_sampled"]
RunStatus = Literal["pending", "running", "completed", "failed", "cancelled"]


class TrainRequest(BaseModel):
    agent: AgentName = "dqn"
    n_servers: int = Field(env_config.NUM_SERVERS, ge=1, le=200)
    episodes: int = Field(200, ge=1, le=100_000)
    episode_length: int = Field(env_config.EPISODE_LENGTH, ge=10, le=10_000)
    total_steps: int = Field(50_000, ge=100, le=10_000_000,
                             description="PPO only — ignored otherwise")
    use_real_traces: bool = False
    trace_family: TraceFamily = "alibaba"
    cluster_type: ClusterType = "homogeneous"
    alpha: float = Field(env_config.REWARD_ALPHA, ge=0.0, le=100.0,
                         description="Power weight in reward")
    beta: float = Field(env_config.REWARD_BETA, ge=0.0, le=1000.0,
                        description="SLA weight in reward")
    seed: int = 0
    eval_episodes: int = Field(10, ge=0, le=500)
    # Train/test seed pool sizes for ML-style generalization eval.
    # n_train_seeds=1, n_test_seeds=0 → legacy single-trajectory training.
    # n_train_seeds=N, n_test_seeds=M → train uniformly across N seeds,
    # eval on M held-out seeds.
    n_train_seeds: int = Field(1, ge=1, le=10_000)
    n_test_seeds: int = Field(0, ge=0, le=1000)
    sweep_id: str | None = None    # set when this run is part of a sweep
    sweep_param: str | None = None # parameter being swept (for the sweep page chart)
    sweep_value: float | None = None  # the parameter's value in this run


class TrainResponse(BaseModel):
    run_id: str


class OfflineTrainRequest(BaseModel):
    """Offline / CMDP training request — no env interaction during training."""
    n_servers: int = Field(env_config.NUM_SERVERS, ge=1, le=200)
    episode_length: int = Field(env_config.EPISODE_LENGTH, ge=10, le=10_000)
    cluster_type: ClusterType = "homogeneous"
    alpha: float = Field(env_config.REWARD_ALPHA, ge=0.0, le=100.0)
    beta: float = Field(env_config.REWARD_BETA, ge=0.0, le=1000.0)
    seed: int = 0
    # Offline-specific
    dataset_path: str = "data/processed/offline.parquet"
    auto_generate_dataset: bool = True
    dataset_episodes: int = Field(200, ge=1, le=10_000,
                                  description="Episodes to roll out for auto-generation")
    dataset_random_eps: float = Field(0.30, ge=0.0, le=1.0,
                                      description="Probability of taking a uniform-random action during behavior rollout (action-coverage knob)")
    behavior_weight_rr: float = Field(0.34, ge=0.0, le=1.0,
                                      description="Weight on RoundRobin in the behavior mix")
    behavior_weight_sjf: float = Field(0.33, ge=0.0, le=1.0,
                                       description="Weight on ShortestJobFirst in the behavior mix")
    behavior_weight_ffd: float = Field(0.33, ge=0.0, le=1.0,
                                       description="Weight on FirstFitDecreasing in the behavior mix")
    iterations: int = Field(20_000, ge=100, le=1_000_000)
    batch_size: int = Field(256, ge=16, le=4096)
    sla_budget: float = Field(0.05, ge=0.0, le=10.0,
                              description="Per-step expected SLA-violation budget (ε_sla)")
    cql_alpha: float = Field(1.0, ge=0.0, le=100.0,
                             description="Conservative-Q penalty weight on Q_c. 0 = vanilla FQI, higher = more pessimistic about OOD actions.")
    reward_kind: Literal["neg_power", "env"] = "neg_power"
    target_update_freq: int = Field(500, ge=1, le=100_000)
    dual_update_freq: int = Field(200, ge=1, le=100_000)
    lambda_init: float = Field(0.5, ge=0.0, le=100.0,
                               description="Initial λ. Warm-starting at 0.5 makes the policy conservative early; λ relaxes if there's slack.")
    lambda_lr: float = Field(0.05, ge=0.0, le=10.0,
                             description="Step size for the dual ascent on (rate − ε).")
    dual_signal: Literal["q_c", "empirical"] = Field(
        "empirical",
        description="'empirical' uses a short on-policy rollout to estimate SLA rate (bypasses CQL inflation of Q_c). 'q_c' uses the original Q_c-based dual signal.",
    )
    dual_eval_seeds: int = Field(3, ge=1, le=20,
                                 description="# rollout seeds per dual update when dual_signal='empirical'.")
    dual_eval_steps: int = Field(200, ge=20, le=10_000,
                                 description="Max steps per dual rollout episode.")
    n_test_seeds: int = Field(50, ge=0, le=1000)
    sweep_id: str | None = None
    sweep_param: str | None = None
    sweep_value: float | None = None


class SweepRequest(BaseModel):
    mode: Literal["online", "offline"]
    sweep_param: str
    sweep_values: list[float] = Field(..., min_length=1, max_length=50)
    base_online: TrainRequest | None = None
    base_offline: OfflineTrainRequest | None = None


class SweepResponse(BaseModel):
    sweep_id: str
    run_ids: list[str]
    sweep_param: str
    sweep_values: list[float]


class ConfigDefaults(BaseModel):
    """Runtime defaults the frontend should hydrate its form with — single
    source of truth lives in `environment/config.py`."""
    n_servers: int
    episode_length: int
    alpha: float
    beta: float
    sla_multiplier: float
    synthetic_arrival_rate: float
    real_trace_max_jobs: int
    trace_sampled_max_jobs: int
    test_seed_offset: int


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
    n_servers: int | None = None
    episodes: int | None = None
    cluster_type: ClusterType = "homogeneous"
    mean_reward_last10: float | None = None
    mean_reward_eval: float | None = None
    mean_power: float | None = None
    mean_active_power: float | None = None
    mean_asleep_servers: float | None = None
    sla_violation_rate: float | None = None
    optimal_reward: float | None = None
    gap_pct: float | None = None
    sweep_id: str | None = None
    sweep_param: str | None = None
    sweep_value: float | None = None


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
