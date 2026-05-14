from __future__ import annotations

import numpy as np
import torch
from fastapi import APIRouter, HTTPException

from agents.agentic import SupervisorAgent
from agents.base_agent import compute_action_mask
from agents.dqn_agent import DQNAgent
from agents.heuristics import (
    FirstFitDecreasingAgent,
    RoundRobinAgent,
    ShortestJobFirstAgent,
)
from agents.ppo_agent import PPOAgent
from backend.models.schemas import (
    SimulateRequest,
    SimulateResponse,
    SimulateStep,
)
from environment.cluster_env import CloudClusterEnv
from environment.workload.synthetic import SyntheticWorkloadGenerator

router = APIRouter(tags=["simulator"])


def _build_agent(name: str, state_dim: int, n_actions: int, n_servers: int, queue_size: int):
    name = name.lower()
    if name == "dqn":
        return DQNAgent(state_dim=state_dim, n_actions=n_actions)
    if name == "ppo":
        return PPOAgent(state_dim=state_dim, n_actions=n_actions)
    if name == "agentic":
        return SupervisorAgent(state_dim=state_dim, n_actions=n_actions)
    if name == "round_robin":
        return RoundRobinAgent(n_servers=n_servers, queue_size=queue_size)
    if name == "sjf":
        return ShortestJobFirstAgent(n_servers=n_servers, queue_size=queue_size)
    if name == "ffd":
        return FirstFitDecreasingAgent(n_servers=n_servers, queue_size=queue_size)
    raise ValueError(name)


@router.post("/simulate", response_model=SimulateResponse)
def simulate(req: SimulateRequest) -> SimulateResponse:
    np.random.seed(req.seed)
    torch.manual_seed(req.seed)

    env = CloudClusterEnv(
        num_servers=req.n_servers,
        workload_generator=SyntheticWorkloadGenerator(seed=req.seed),
        episode_length=req.episode_length,
    )
    state_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    try:
        agent = _build_agent(
            req.agent, state_dim, n_actions,
            n_servers=env.num_servers, queue_size=env.job_queue_size,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"unknown agent: {e}")

    if req.checkpoint_path:
        try:
            agent.load(req.checkpoint_path)
            if isinstance(agent, DQNAgent):
                agent.eps = 0.0
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="checkpoint not found")

    obs, _ = env.reset()
    mask = compute_action_mask(env)
    steps: list[SimulateStep] = []
    total_reward = 0.0
    total_power = 0.0
    total_sla = 0

    while True:
        action = agent.select_action(obs, mask, greedy=True)
        next_obs, reward, terminated, truncated, info = env.step(action)
        mask = compute_action_mask(env)
        utils = [s.cpu_utilization for s in env.servers]
        steps.append(
            SimulateStep(
                step=info["timestep"],
                action=int(action),
                reward=float(reward),
                power=float(info.get("step_power", 0.0)),
                sla_violations=int(info.get("step_sla_violations", 0)),
                server_utilizations=[float(u) for u in utils],
                queue_length=int(info.get("queue_length", 0)),
            )
        )
        total_reward += reward
        total_power += info.get("step_power", 0.0)
        total_sla = info.get("sla_violations", total_sla)
        obs = next_obs
        if terminated or truncated:
            break

    return SimulateResponse(
        total_reward=total_reward,
        total_power=total_power,
        total_sla=total_sla,
        steps=steps,
    )
