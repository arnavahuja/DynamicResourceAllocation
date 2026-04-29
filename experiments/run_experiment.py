"""CLI entry point for training + evaluation experiments.

Examples:
    # Train one agent
    python -m experiments.run_experiment train --agent dqn --episodes 1000

    # Quick PPO run
    python -m experiments.run_experiment train --agent ppo --total-steps 50000

    # Evaluate all baselines + a trained DQN checkpoint side-by-side
    python -m experiments.run_experiment compare \\
        --dqn-checkpoint checkpoints/dqn_n10_s0/best.pt
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agents.agentic import SupervisorAgent
from agents.base_agent import BaseAgent
from agents.dqn_agent import DQNAgent
from agents.offline import CMDPAgent, load_dataset, train_offline
from agents.heuristics import (
    FirstFitDecreasingAgent,
    RoundRobinAgent,
    ShortestJobFirstAgent,
)
from agents.ppo_agent import PPOAgent
from environment import config
from environment.cluster_env import CloudClusterEnv
from environment.workload.synthetic import SyntheticWorkloadGenerator
from training.evaluator import comparison_table, evaluate
from training.trainer import Trainer
from training.wandb_logger import WandBLogger


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(n_servers: int, episode_length: int, seed: int) -> CloudClusterEnv:
    return CloudClusterEnv(
        num_servers=n_servers,
        workload_generator=SyntheticWorkloadGenerator(seed=seed),
        episode_length=episode_length,
    )


def build_agent(name: str, state_dim: int, n_actions: int) -> BaseAgent:
    name = name.lower()
    if name == "dqn":
        return DQNAgent(state_dim=state_dim, n_actions=n_actions)
    if name == "ppo":
        return PPOAgent(state_dim=state_dim, n_actions=n_actions)
    if name == "agentic":
        return SupervisorAgent(state_dim=state_dim, n_actions=n_actions)
    if name == "round_robin":
        return RoundRobinAgent(n_actions=n_actions)
    if name == "sjf":
        return ShortestJobFirstAgent(n_servers=n_actions)
    if name == "ffd":
        return FirstFitDecreasingAgent(n_servers=n_actions)
    raise ValueError(f"Unknown agent: {name}")


# ───────────────────────── train ─────────────────────────

def cmd_train(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    env = make_env(args.n_servers, args.episode_length, args.seed)
    state_dim = int(np.prod(env.observation_space.shape))
    n_actions = int(env.action_space.n)
    agent = build_agent(args.agent, state_dim, n_actions)

    run_name = args.run_name or f"{args.agent}_n{args.n_servers}_s{args.seed}"
    cfg_dump = {
        "agent": args.agent,
        "n_servers": args.n_servers,
        "episode_length": args.episode_length,
        "seed": args.seed,
        "episodes": args.episodes,
        "total_steps": args.total_steps,
    }

    with WandBLogger(run_name=run_name, agent_name=args.agent, cfg=cfg_dump) as wb:
        if isinstance(agent, SupervisorAgent):
            ep_stats = agent.train_loop(env=env, episodes=args.episodes)
            for i, s in enumerate(ep_stats):
                wb.log(
                    {
                        "train/episode_reward": s["reward"],
                        "train/power_consumption": s["power"],
                        "train/sla_violations": s["sla_violations"],
                    },
                    step=i,
                )
            ckpt_dir = Path("checkpoints") / run_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(ckpt_dir / "last.pt"))
            agent.save(str(ckpt_dir / "best.pt"))
            rewards = [s["reward"] for s in ep_stats]
        elif isinstance(agent, PPOAgent):
            ep_stats = agent.train_loop(env=env, total_steps=args.total_steps)
            for i, s in enumerate(ep_stats):
                wb.log(
                    {
                        "train/episode_reward": s["reward"],
                        "train/power_consumption": s["power"],
                        "train/sla_violations": s["sla_violations"],
                    },
                    step=i,
                )
            ckpt_dir = Path("checkpoints") / run_name
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            agent.save(str(ckpt_dir / "last.pt"))
            agent.save(str(ckpt_dir / "best.pt"))
            rewards = [s["reward"] for s in ep_stats]
        else:
            trainer = Trainer(env=env, agent=agent, run_name=run_name, wandb_logger=wb)
            result = trainer.train(args.episodes)
            rewards = [e.reward for e in result.episodes]

    if rewards:
        print(
            f"\nDone. first10_mean={np.mean(rewards[:10]):.2f} "
            f"last10_mean={np.mean(rewards[-10:]):.2f} max={max(rewards):.2f}"
        )

    if args.eval_episodes > 0:
        eval_env = make_env(args.n_servers, args.episode_length, args.seed + 1000)
        result = evaluate(agent, eval_env, n_episodes=args.eval_episodes)
        print("\n" + comparison_table([result]))
    return 0


# ───────────────────────── offline ─────────────────────────

def cmd_offline(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    print(f"Loading offline dataset {args.dataset} ...")
    ds = load_dataset(args.dataset, reward_kind=args.reward_kind)
    print(
        f"  loaded {ds.n:,} transitions "
        f"obs_dim={ds.obs_dim} n_actions={ds.n_actions}"
    )

    if ds.n_actions != args.n_servers:
        print(
            f"[!] dataset has n_actions={ds.n_actions} but --n-servers={args.n_servers}; "
            f"using dataset value to keep network shapes consistent."
        )
    n_actions = ds.n_actions

    agent = CMDPAgent(
        state_dim=ds.obs_dim,
        n_actions=n_actions,
        sla_budget=args.sla_budget,
    )
    print(f"  device={agent.device}  λ_init={agent.lam:.3f}  ε_sla={agent.sla_budget}")

    train_offline(
        agent=agent,
        dataset=ds,
        iterations=args.iterations,
        batch_size=args.batch_size,
        target_update_freq=args.target_update_freq,
        dual_update_freq=args.dual_update_freq,
        log_every=args.log_every,
        seed=args.seed,
    )

    run_name = args.run_name or f"cmdp_offline_s{args.seed}"
    ckpt_dir = Path("checkpoints") / run_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    agent.save(str(ckpt_dir / "best.pt"))
    agent.save(str(ckpt_dir / "last.pt"))
    print(f"✔ saved checkpoint to {ckpt_dir}/best.pt")

    if args.eval_episodes > 0:
        eval_env = make_env(n_actions, args.episode_length, args.seed + 1000)
        result = evaluate(agent, eval_env, n_episodes=args.eval_episodes, agent_name="CMDP-offline")
        print("\n" + comparison_table([result]))
    return 0


# ───────────────────────── compare ─────────────────────────

def cmd_compare(args: argparse.Namespace) -> int:
    set_seed(args.seed)
    eval_env = make_env(args.n_servers, args.episode_length, args.seed + 1000)
    state_dim = int(np.prod(eval_env.observation_space.shape))
    n_actions = int(eval_env.action_space.n)

    agents: list[tuple[str, BaseAgent]] = [
        ("RoundRobin", RoundRobinAgent(n_actions=n_actions)),
        ("ShortestJobFirst", ShortestJobFirstAgent(n_servers=n_actions)),
        ("FirstFitDecreasing", FirstFitDecreasingAgent(n_servers=n_actions)),
    ]

    if args.dqn_checkpoint:
        dqn = DQNAgent(state_dim=state_dim, n_actions=n_actions)
        dqn.load(args.dqn_checkpoint)
        dqn.eps = 0.0
        agents.append(("DQN", dqn))
    if args.ppo_checkpoint:
        ppo = PPOAgent(state_dim=state_dim, n_actions=n_actions)
        ppo.load(args.ppo_checkpoint)
        agents.append(("PPO", ppo))
    if args.agentic_checkpoint:
        sup = SupervisorAgent(state_dim=state_dim, n_actions=n_actions)
        sup.load(args.agentic_checkpoint)
        agents.append(("Agentic", sup))
    if args.cmdp_checkpoint:
        cmdp = CMDPAgent(state_dim=state_dim, n_actions=n_actions)
        cmdp.load(args.cmdp_checkpoint)
        agents.append(("CMDP-offline", cmdp))

    results = []
    for name, ag in agents:
        print(f"Evaluating {name}...")
        results.append(evaluate(ag, eval_env, n_episodes=args.eval_episodes, agent_name=name))

    print("\n" + comparison_table(results))
    return 0


# ───────────────────────── main ─────────────────────────

def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--n-servers", type=int, default=config.NUM_SERVERS)
    p.add_argument("--episode-length", type=int, default=config.EPISODE_LENGTH)
    p.add_argument("--seed", type=int, default=0)

    sub = p.add_subparsers(dest="cmd", required=True)

    pt = sub.add_parser("train")
    pt.add_argument("--agent", default="dqn",
                    choices=["dqn", "ppo", "agentic", "round_robin", "sjf", "ffd"])
    pt.add_argument("--episodes", type=int, default=config.MAX_EPISODES,
                    help="Episodes (DQN / heuristics).")
    pt.add_argument("--total-steps", type=int, default=200_000,
                    help="Total env steps (PPO).")
    pt.add_argument("--run-name", default=None)
    pt.add_argument("--eval-episodes", type=int, default=20)
    pt.set_defaults(func=cmd_train)

    pc = sub.add_parser("compare")
    pc.add_argument("--dqn-checkpoint", default=None)
    pc.add_argument("--ppo-checkpoint", default=None)
    pc.add_argument("--agentic-checkpoint", default=None,
                    help="Path to supervisor checkpoint (sub-agents auto-loaded)")
    pc.add_argument("--cmdp-checkpoint", default=None,
                    help="Path to offline CMDP agent checkpoint")

    po = sub.add_parser("offline", help="Train CMDP from a logged-transitions Parquet")
    po.add_argument("--dataset", required=True, help="Path to offline.parquet")
    po.add_argument("--iterations", type=int, default=20_000)
    po.add_argument("--batch-size", type=int, default=256)
    po.add_argument("--target-update-freq", type=int, default=500)
    po.add_argument("--dual-update-freq", type=int, default=200)
    po.add_argument("--log-every", type=int, default=1000)
    po.add_argument("--sla-budget", type=float, default=0.05,
                    help="ε_sla — per-step expected SLA violations the policy must stay under")
    po.add_argument("--reward-kind", default="neg_power",
                    choices=["neg_power", "env"])
    po.add_argument("--run-name", default=None)
    po.add_argument("--eval-episodes", type=int, default=10)
    po.set_defaults(func=cmd_offline)
    pc.add_argument("--eval-episodes", type=int, default=20)
    pc.set_defaults(func=cmd_compare)

    args = p.parse_args()
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
