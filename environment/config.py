"""Single source of truth for all environment + training hyperparameters.

Values are loaded from environment variables (which python-dotenv populates from
.env at process start). Hardcoded constants here are *defaults only* — they
should never be referenced directly elsewhere; always go through this module.
"""
from __future__ import annotations

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    _ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
    if _ENV_PATH.exists():
        load_dotenv(_ENV_PATH)
except ImportError:
    pass


def _f(key: str, default: float) -> float:
    return float(os.getenv(key, default))


def _i(key: str, default: int) -> int:
    return int(os.getenv(key, default))


def _s(key: str, default: str) -> str:
    return os.getenv(key, default)


def _b(key: str, default: bool) -> bool:
    return os.getenv(key, str(default)).lower() in ("1", "true", "yes", "y")


# ── Cluster ─────────────────────────────────────────────────────
NUM_SERVERS: int = _i("N_SERVERS", 10)
SERVER_CPU_CAPACITY: float = _f("SERVER_CPU_CAPACITY", 1.0)
SERVER_MEM_CAPACITY: float = _f("SERVER_MEM_CAPACITY", 1.0)

# ── Power model: P(u) = P_idle + (P_max - P_idle) * u^alpha ─────
P_IDLE: float = _f("P_IDLE", 100.0)
P_MAX: float = _f("P_MAX", 300.0)
POWER_ALPHA: float = _f("POWER_ALPHA", 1.4)

# ── SLA / reward ────────────────────────────────────────────────
SLA_LATENCY_DEADLINE: int = _i("SLA_LATENCY_DEADLINE", 10)
REWARD_ALPHA: float = _f("ALPHA", 1.0)
REWARD_BETA: float = _f("BETA", 50.0)

# ── Episode ─────────────────────────────────────────────────────
JOB_QUEUE_SIZE: int = _i("MAX_QUEUE_SIZE", 5)
EPISODE_LENGTH: int = _i("EPISODE_LENGTH", 1000)
INVALID_ACTION_PENALTY: float = _f("INVALID_ACTION_PENALTY", -10.0)

# ── Training ────────────────────────────────────────────────────
GAMMA: float = _f("GAMMA", 0.99)
LEARNING_RATE: float = _f("LEARNING_RATE", 3e-4)
BATCH_SIZE: int = _i("BATCH_SIZE", 64)
REPLAY_BUFFER_SIZE: int = _i("REPLAY_BUFFER_SIZE", 100_000)
TARGET_UPDATE_FREQ: int = _i("TARGET_UPDATE_FREQ", 1000)
EPSILON_START: float = _f("EPSILON_START", 1.0)
EPSILON_END: float = _f("EPSILON_END", 0.05)
EPSILON_DECAY: float = _f("EPSILON_DECAY", 0.999)
PPO_CLIP: float = _f("PPO_CLIP", 0.2)
PPO_EPOCHS: int = _i("PPO_EPOCHS", 10)
MAX_EPISODES: int = _i("MAX_EPISODES", 5000)

# ── WandB ───────────────────────────────────────────────────────
WANDB_API_KEY: str = _s("WANDB_API_KEY", "")
WANDB_PROJECT: str = _s("WANDB_PROJECT", "cloud-rl-scheduler")
WANDB_ENTITY: str = _s("WANDB_ENTITY", "")

# ── Backend ─────────────────────────────────────────────────────
BACKEND_HOST: str = _s("BACKEND_HOST", "0.0.0.0")
BACKEND_PORT: int = _i("BACKEND_PORT", 8000)
DATABASE_URL: str = _s("DATABASE_URL", "sqlite:///./experiments.db")

# ── GCP ─────────────────────────────────────────────────────────
GCP_PROJECT_ID: str = _s("GCP_PROJECT_ID", "")
GCP_REGION: str = _s("GCP_REGION", "us-central1")
GCP_BUCKET_NAME: str = _s("GCP_BUCKET_NAME", "")
GOOGLE_APPLICATION_CREDENTIALS: str = _s("GOOGLE_APPLICATION_CREDENTIALS", "")
