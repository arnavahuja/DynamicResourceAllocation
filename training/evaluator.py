from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from agents.base_agent import BaseAgent, compute_action_mask
from environment.cluster_env import CloudClusterEnv


@dataclass
class EvalResult:
    agent_name: str
    n_episodes: int
    mean_reward: float
    std_reward: float
    mean_power: float
    std_power: float
    sla_violation_rate: float       # violations / jobs_completed
    mean_jobs_completed: float
    mean_steps: float
    per_episode_rewards: list[float] = field(default_factory=list)
    per_episode_power: list[float] = field(default_factory=list)
    per_episode_sla: list[int] = field(default_factory=list)
    per_episode_utilization: list[list[float]] = field(default_factory=list)

    def as_row(self) -> dict:
        return {
            "agent": self.agent_name,
            "mean_reward": round(self.mean_reward, 2),
            "std_reward": round(self.std_reward, 2),
            "mean_power": round(self.mean_power, 1),
            "sla_violation_rate": round(self.sla_violation_rate, 4),
            "mean_jobs_completed": round(self.mean_jobs_completed, 1),
        }


def evaluate(
    agent: BaseAgent,
    env: CloudClusterEnv,
    n_episodes: int = 100,
    agent_name: str | None = None,
    workload_seeds: list[int] | None = None,
) -> EvalResult:
    """Run `n_episodes` greedy rollouts and return aggregate stats.

    If `workload_seeds` is provided, episode i uses workload_seeds[i] (and
    n_episodes is overridden by the list length). This is what enables the
    held-out generalization test — pass test seeds the agent never trained on.
    """
    rewards: list[float] = []
    powers: list[float] = []
    slas: list[int] = []
    jobs_done: list[int] = []
    steps: list[int] = []
    utilizations: list[list[float]] = []

    if workload_seeds is not None:
        n_episodes = len(workload_seeds)

    for ep_idx in range(n_episodes):
        if workload_seeds is not None:
            obs, _ = env.reset(options={"workload_seed": int(workload_seeds[ep_idx])})
        else:
            obs, _ = env.reset()
        mask = compute_action_mask(env)
        ep_r = 0.0
        ep_p = 0.0
        ep_s = 0
        ep_steps = 0
        cpu_trace: list[float] = []
        info: dict = {}

        while True:
            action = agent.select_action(obs, mask, greedy=True)
            obs, reward, terminated, truncated, info = env.step(action)
            mask = compute_action_mask(env)
            ep_r += reward
            ep_p += info.get("step_power", 0.0)
            ep_s = info.get("sla_violations", ep_s)
            cpu_trace.append(info.get("mean_cpu_utilization", 0.0))
            ep_steps += 1
            if terminated or truncated:
                break

        rewards.append(ep_r)
        powers.append(ep_p)
        slas.append(ep_s)
        jobs_done.append(info.get("jobs_completed", 0))
        steps.append(ep_steps)
        utilizations.append(cpu_trace)

    total_jobs = sum(jobs_done)
    sla_rate = sum(slas) / total_jobs if total_jobs > 0 else 0.0

    return EvalResult(
        agent_name=agent_name or type(agent).__name__,
        n_episodes=n_episodes,
        mean_reward=float(np.mean(rewards)),
        std_reward=float(np.std(rewards)),
        mean_power=float(np.mean(powers)),
        std_power=float(np.std(powers)),
        sla_violation_rate=sla_rate,
        mean_jobs_completed=float(np.mean(jobs_done)),
        mean_steps=float(np.mean(steps)),
        per_episode_rewards=rewards,
        per_episode_power=powers,
        per_episode_sla=slas,
        per_episode_utilization=utilizations,
    )


def comparison_table(results: list[EvalResult]) -> str:
    """Format a list of EvalResults as a fixed-width text table."""
    header = (
        f"{'Agent':<28}{'mean_R':>12}{'std_R':>10}"
        f"{'mean_power':>14}{'sla_rate':>12}{'mean_jobs':>12}"
    )
    lines = [header, "-" * len(header)]
    for r in results:
        lines.append(
            f"{r.agent_name:<28}"
            f"{r.mean_reward:>12.2f}"
            f"{r.std_reward:>10.2f}"
            f"{r.mean_power:>14.1f}"
            f"{r.sla_violation_rate:>12.4f}"
            f"{r.mean_jobs_completed:>12.1f}"
        )
    return "\n".join(lines)
