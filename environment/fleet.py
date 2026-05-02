"""Server fleet generators.

Two cluster types:
- homogeneous: every server uses the env-default power params (P_IDLE, P_MAX,
  POWER_ALPHA from .env).
- heterogeneous: servers are drawn from three efficiency tiers (low / mid /
  high P_max) so RL can exploit server-identity information that load-balancing
  heuristics ignore. Generation is deterministic in (num_servers, seed) so the
  *same* heterogeneous cluster is built whenever those two match — that's what
  makes RR vs DQN comparisons fair.
"""
from __future__ import annotations

import numpy as np

from environment import config
from environment.server import Server

ClusterType = str  # "homogeneous" | "heterogeneous"

# Three tiers used by the heterogeneous generator. Tiers vary in BOTH
# power efficiency (p_max) and capacity (cpu): efficient servers are also
# smaller, power-hungry servers are also bigger. This creates a real
# tradeoff — heuristics that ignore server identity (RR) waste energy by
# routing to big servers; DQN can learn to consolidate small jobs on
# efficient ones and reserve big servers for jobs that need them.
_HETERO_TIERS = [
    {"p_max_factor": 0.3, "cpu_factor": 0.5, "label": "efficient"},
    {"p_max_factor": 1.0, "cpu_factor": 1.0, "label": "standard"},
    {"p_max_factor": 3.0, "cpu_factor": 2.0, "label": "power_hungry"},
]


def build_fleet(
    num_servers: int,
    cluster_type: ClusterType = "homogeneous",
    seed: int = 0,
) -> list[Server]:
    if cluster_type == "homogeneous":
        return [Server(server_id=i) for i in range(num_servers)]
    if cluster_type == "heterogeneous":
        rng = np.random.default_rng(seed)
        # Cycle tiers so small clusters still see all three; shuffle order
        # within the cycle so server_id doesn't trivially encode tier.
        tier_assignments = [i % len(_HETERO_TIERS) for i in range(num_servers)]
        rng.shuffle(tier_assignments)
        servers = []
        for i, tier_idx in enumerate(tier_assignments):
            tier = _HETERO_TIERS[tier_idx]
            servers.append(
                Server(
                    server_id=i,
                    cpu_capacity=config.SERVER_CPU_CAPACITY * tier["cpu_factor"],
                    mem_capacity=config.SERVER_MEM_CAPACITY * tier["cpu_factor"],
                    p_idle=config.P_IDLE,
                    p_max=config.P_MAX * tier["p_max_factor"],
                    power_alpha=config.POWER_ALPHA,
                )
            )
        return servers
    raise ValueError(f"Unknown cluster_type: {cluster_type!r}")


def fleet_summary(servers: list[Server]) -> list[dict]:
    """Compact per-server snapshot — useful for surfacing in the API/UI."""
    return [
        {
            "server_id": s.server_id,
            "p_idle": s.p_idle,
            "p_max": s.p_max,
            "power_alpha": s.power_alpha,
        }
        for s in servers
    ]
