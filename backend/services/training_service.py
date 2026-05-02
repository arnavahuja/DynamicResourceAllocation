"""Async training-job manager.

Each `POST /train` spawns a worker thread that runs the training loop and
streams metrics into the MetricsBus. Status (progress / current episode)
is tracked in an in-memory dict; final episode rows are also persisted to
SQLite via `backend.models.db`.
"""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from agents.agentic import SupervisorAgent
from agents.dqn_agent import DQNAgent
from agents.heuristics import (
    FirstFitDecreasingAgent,
    RoundRobinAgent,
    ShortestJobFirstAgent,
)
from agents.ppo_agent import PPOAgent
from backend.core.websocket_manager import bus
from backend.models import db
from backend.services import gcp_service
from backend.models.schemas import TrainRequest
from environment import config as env_config
from environment.cluster_env import CloudClusterEnv
from environment.workload.synthetic import SyntheticWorkloadGenerator
from training.evaluator import evaluate
from training.trainer import EpisodeStats, Trainer

_executor = ThreadPoolExecutor(max_workers=4)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()


def _new_run_id(
    agent: str, cluster_type: str, n_servers: int, seed: int,
    n_train_seeds: int, n_test_seeds: int,
) -> str:
    """Format: <agent>_<homo|het>_n<N>_s<seed>_tr<TR>te<TE>_<uuid>.

    Example: dqn_het_n10_s0_tr20te10_3a83b3da. The tr/te tags make the
    train/test protocol glanceable.
    """
    cluster_tag = "het" if cluster_type == "heterogeneous" else "homo"
    return (
        f"{agent}_{cluster_tag}_n{n_servers}_s{seed}_"
        f"tr{n_train_seeds}te{n_test_seeds}_{uuid.uuid4().hex[:8]}"
    )


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


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
    raise ValueError(f"Unknown agent: {name}")


def get_status(run_id: str) -> dict[str, Any] | None:
    with _jobs_lock:
        job = _jobs.get(run_id)
        if job is None:
            return None
        return dict(job)  # shallow copy


def cancel(run_id: str) -> bool:
    with _jobs_lock:
        job = _jobs.get(run_id)
        if not job or job["status"] not in ("running", "pending"):
            return False
        job["cancel_requested"] = True
        return True


def list_active() -> list[str]:
    with _jobs_lock:
        return list(_jobs.keys())


def start_training(req: TrainRequest) -> str:
    run_id = _new_run_id(
        req.agent, req.cluster_type, req.n_servers, req.seed,
        req.n_train_seeds, req.n_test_seeds,
    )
    created_at = datetime.now(timezone.utc).isoformat()

    # Snapshot the cluster's actual idle/ceiling power so optimal-reward
    # is correct for heterogeneous runs (where per-server p_max varies).
    from environment.fleet import build_fleet
    _preview_servers = build_fleet(
        num_servers=req.n_servers,
        cluster_type=req.cluster_type,
        seed=0,
    )
    sum_p_idle = sum(s.p_idle for s in _preview_servers)
    sum_p_max = sum(s.p_max for s in _preview_servers)

    config_dump = {
        **req.model_dump(),
        "p_idle": env_config.P_IDLE,
        "p_max": env_config.P_MAX,
        "sum_p_idle": sum_p_idle,
        "sum_p_max": sum_p_max,
        "invalid_action_penalty": env_config.INVALID_ACTION_PENALTY,
    }

    db.insert_experiment(
        run_id=run_id,
        agent=req.agent,
        status="pending",
        created_at=created_at,
        config=config_dump,
    )

    with _jobs_lock:
        _jobs[run_id] = {
            "run_id": run_id,
            "status": "pending",
            "progress": 0.0,
            "eta_seconds": None,
            "current_episode": 0,
            "total_episodes": req.episodes,
            "last_reward": None,
            "started_at": None,
            "cancel_requested": False,
            "error": None,
            "agent": req.agent,
        }

    _executor.submit(_run_job, run_id, req)
    return run_id


def _on_episode(run_id: str, total: int, started_at: float):
    """Build a per-episode hook closure used by both DQN/heuristic Trainer
    and the Supervisor's own loop."""

    def hook(stats: EpisodeStats) -> None:
        with _jobs_lock:
            job = _jobs.get(run_id)
            if job is None:
                return
            ep_idx = stats.episode + 1
            job["current_episode"] = ep_idx
            job["progress"] = min(1.0, ep_idx / max(1, total))
            job["last_reward"] = stats.reward
            elapsed = time.time() - started_at
            rate = ep_idx / elapsed if elapsed > 0 else 0.0
            job["eta_seconds"] = (
                (total - ep_idx) / rate if rate > 0 else None
            )

        db.insert_episode(
            run_id=run_id,
            episode=stats.episode,
            reward=stats.reward,
            power=stats.power,
            sla_violations=stats.sla_violations,
            steps=stats.steps,
        )
        bus.publish_threadsafe(
            run_id,
            {
                "event": "episode",
                "episode": stats.episode,
                "reward": stats.reward,
                "power": stats.power,
                "sla_violations": stats.sla_violations,
                "steps": stats.steps,
                "epsilon": stats.epsilon,
                "loss_mean": stats.loss_mean,
            },
        )

    return hook


def _run_job(run_id: str, req: TrainRequest) -> None:
    started_at = time.time()
    try:
        with _jobs_lock:
            _jobs[run_id]["status"] = "running"
            _jobs[run_id]["started_at"] = started_at
        db.update_experiment_status(run_id, "running")

        _seed_all(req.seed)

        # Train/test seed pools.
        # Train seeds: [seed, seed+1, ..., seed + n_train_seeds - 1]
        # Test seeds:  [seed + 1_000_000, ..., seed + 1_000_000 + n_test_seeds - 1]
        # The 1M offset guarantees no overlap regardless of n_train_seeds.
        train_seeds = [req.seed + i for i in range(req.n_train_seeds)]
        test_seeds = [req.seed + 1_000_000 + i for i in range(req.n_test_seeds)]

        env = CloudClusterEnv(
            num_servers=req.n_servers,
            workload_generator=SyntheticWorkloadGenerator(seed=req.seed),
            episode_length=req.episode_length,
            reward_alpha=req.alpha,
            reward_beta=req.beta,
            cluster_type=req.cluster_type,
            cluster_seed=0,  # Fixed across all runs of same N → fair comparison.
        )
        state_dim = int(np.prod(env.observation_space.shape))
        n_actions = int(env.action_space.n)
        agent = _build_agent(
            req.agent, state_dim, n_actions,
            n_servers=env.num_servers, queue_size=env.job_queue_size,
        )

        episode_hook = _on_episode(run_id, req.episodes, started_at)

        if isinstance(agent, SupervisorAgent):
            ep_stats = agent.train_loop(
                env=env, episodes=req.episodes,
                on_episode_end=episode_hook, log_every=10,
            )
        elif isinstance(agent, PPOAgent):
            ep_stats = agent.train_loop(
                env=env, total_steps=req.total_steps,
                on_episode_end=episode_hook,
                train_seeds=train_seeds,
            )
        else:
            trainer = Trainer(env=env, agent=agent, run_name=run_id)
            result = trainer.train(
                req.episodes,
                on_episode_end=episode_hook,
                train_seeds=train_seeds,
            )
            ep_stats = [
                {
                    "episode": e.episode,
                    "reward": e.reward,
                    "power": e.power,
                    "sla_violations": e.sla_violations,
                    "steps": e.steps,
                }
                for e in result.episodes
            ]

        # save final checkpoint
        ckpt_dir = Path("checkpoints") / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            agent.save(str(ckpt_dir / "last.pt"))
        except Exception:
            pass

        # Best-effort upload to GCS — silently no-op if not configured.
        try:
            gcp_service.upload_run_checkpoints(run_id, ckpt_dir)
        except Exception:
            pass

        # Eval — generalization test on held-out test seeds.
        # If n_test_seeds > 0: eval on those (the proper ML-style protocol).
        # Else: legacy fallback to eval_episodes runs on the training seed.
        eval_data = None
        if req.n_test_seeds > 0 or req.eval_episodes > 0:
            eval_env = CloudClusterEnv(
                num_servers=req.n_servers,
                workload_generator=SyntheticWorkloadGenerator(seed=req.seed),
                episode_length=req.episode_length,
                reward_alpha=req.alpha,
                reward_beta=req.beta,
                cluster_type=req.cluster_type,
                cluster_seed=0,
            )
            if req.n_test_seeds > 0:
                ev = evaluate(agent, eval_env, workload_seeds=test_seeds)
            else:
                ev = evaluate(agent, eval_env, n_episodes=req.eval_episodes)
            eval_data = ev.as_row()

        rewards = [s["reward"] for s in ep_stats] if ep_stats else []
        summary = {
            "n_episodes": len(ep_stats),
            "mean_reward_last10": float(np.mean(rewards[-10:])) if rewards else None,
            "max_reward": float(max(rewards)) if rewards else None,
        }

        with _jobs_lock:
            _jobs[run_id]["status"] = "completed"
            _jobs[run_id]["progress"] = 1.0
        db.update_experiment_status(
            run_id, "completed", summary=summary, eval_data=eval_data
        )
        bus.publish_threadsafe(run_id, {"event": "completed", **summary})

    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        with _jobs_lock:
            _jobs[run_id]["status"] = "failed"
            _jobs[run_id]["error"] = err
        db.update_experiment_status(run_id, "failed", error=err)
        bus.publish_threadsafe(run_id, {"event": "failed", "error": str(e)})
    finally:
        bus.close_run(run_id)
