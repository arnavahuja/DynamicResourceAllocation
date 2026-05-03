from __future__ import annotations

from fastapi import APIRouter

from backend.models.schemas import ConfigDefaults
from environment import config as env_config

router = APIRouter(prefix="/config", tags=["config"])


@router.get("/defaults", response_model=ConfigDefaults)
def get_defaults() -> ConfigDefaults:
    """Single source of truth for form defaults — frontend hydrates from this
    so values can't drift between Python and JS."""
    return ConfigDefaults(
        n_servers=env_config.NUM_SERVERS,
        episode_length=env_config.EPISODE_LENGTH,
        alpha=env_config.REWARD_ALPHA,
        beta=env_config.REWARD_BETA,
        sla_multiplier=env_config.SLA_MULTIPLIER,
        synthetic_arrival_rate=env_config.SYNTHETIC_ARRIVAL_RATE,
        real_trace_max_jobs=env_config.REAL_TRACE_MAX_JOBS,
        trace_sampled_max_jobs=env_config.TRACE_SAMPLED_MAX_JOBS,
        test_seed_offset=env_config.TEST_SEED_OFFSET,
    )
