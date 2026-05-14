import numpy as np

from environment import config


def compute_power(
    utilization: float,
    p_idle: float = config.P_IDLE,
    p_max: float = config.P_MAX,
    alpha: float = config.POWER_ALPHA,
) -> float:
    """Cubic power model: P(u) = P_idle + (P_max - P_idle) * u^alpha."""
    u = np.clip(utilization, 0.0, 1.0)
    return p_idle + (p_max - p_idle) * (u ** alpha)


def compute_cluster_power(
    server_utilizations: list[float],
    p_idle: float = config.P_IDLE,
    p_max: float = config.P_MAX,
    alpha: float = config.POWER_ALPHA,
) -> float:
    """Total power consumption across all servers."""
    return sum(
        compute_power(u, p_idle, p_max, alpha) for u in server_utilizations
    )
