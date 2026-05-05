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
from backend.models.schemas import OfflineTrainRequest, TrainRequest
from environment import config as env_config
from environment.cluster_env import CloudClusterEnv
from environment.workload.alibaba import AlibabaWorkloadGenerator
from environment.workload.google_v2 import GoogleV2WorkloadGenerator
from environment.workload.google_v3 import GoogleV3WorkloadGenerator
from environment.workload.synthetic import SyntheticWorkloadGenerator
from environment.workload.trace_sampled import TraceSampledWorkloadGenerator
from training.evaluator import evaluate
from training.trainer import EpisodeStats, Trainer

_executor = ThreadPoolExecutor(max_workers=4)
_jobs: dict[str, dict[str, Any]] = {}
_jobs_lock = threading.Lock()

# Shared log directory across Trainer / PPOAgent / SupervisorAgent so every
# agent's per-run log lands in the same place regardless of agent family.
LOG_DIR = "logs"


_WORKLOAD_TAGS = {
    "synthetic": "synth",
    "google_v2": "gv2",
    "google_v2_sampled": "gv2s",
    "alibaba": "ali",
    "google_v3": "gv3",
    "offline": "offline",  # CMDP runs from a static parquet
}


def _new_run_id(
    agent: str, cluster_type: str, n_servers: int, seed: int,
    n_train_seeds: int, n_test_seeds: int,
    workload: str | None = None,
) -> str:
    """Format: <agent>_<workload>_<homo|het>_n<N>_s<seed>_tr<TR>te<TE>_<uuid>.

    Example: dqn_synth_het_n10_s0_tr20te10_3a83b3da. The workload tag is
    only added when explicitly provided — runs created before this field
    existed remain valid; their IDs simply lack the tag, which is fine
    because nothing parses run_ids semantically.
    """
    cluster_tag = "het" if cluster_type == "heterogeneous" else "homo"
    workload_tag = _WORKLOAD_TAGS.get(workload or "", "") if workload else ""
    parts = [agent]
    if workload_tag:
        parts.append(workload_tag)
    parts.append(cluster_tag)
    parts.extend([
        f"n{n_servers}", f"s{seed}",
        f"tr{n_train_seeds}te{n_test_seeds}", uuid.uuid4().hex[:8],
    ])
    return "_".join(parts)


def _seed_all(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


def _build_workload(req: TrainRequest):
    """Pick a workload generator based on request flags.

    Real-trace generators ignore `seed` (replay is deterministic). Returns
    a generator instance; the caller is responsible for env wiring.

    Trace paths are anchored to the repo root so uvicorn's CWD doesn't
    matter.
    """
    if not req.use_real_traces:
        return SyntheticWorkloadGenerator(seed=req.seed)

    repo_root = Path(__file__).resolve().parents[2]
    fam = req.trace_family
    # Real-trace cap lives in env_config (REAL_TRACE_MAX_JOBS) — these traces
    # are from huge production clusters and >cap arrivals saturate a 50-server
    # study cluster, making every agent look identical at ~99% SLA violations.
    MAX_JOBS = env_config.REAL_TRACE_MAX_JOBS
    if fam == "alibaba":
        return AlibabaWorkloadGenerator(
            trace_path=str(repo_root / "data/raw/alibaba/batch_task*.csv"),
            max_jobs=MAX_JOBS,
        )
    if fam == "google_v2":
        return GoogleV2WorkloadGenerator(
            trace_dir=str(repo_root / "data/raw/google_v2"),
            max_jobs=MAX_JOBS,
        )
    if fam == "google_v3":
        return GoogleV3WorkloadGenerator(
            trace_dir=str(repo_root / "data/raw/google_v3"),
            max_jobs=MAX_JOBS,
        )
    if fam == "google_v2_sampled":
        # Trace-distribution job specs + synthetic Poisson timing → avoids
        # the saturation that pure replay causes on small study clusters.
        return TraceSampledWorkloadGenerator(
            trace_dir=str(repo_root / "data/raw/google_v2"),
            max_jobs=env_config.TRACE_SAMPLED_MAX_JOBS,
            seed=req.seed,
            source="google_v2",
        )
    raise ValueError(f"Unknown trace_family: {fam}")


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


# ── Cleanup helpers used by the experiments router on delete ──────────────

_REPO_ROOT = Path(__file__).resolve().parents[2]
_CHECKPOINTS_DIR = _REPO_ROOT / "checkpoints"
_LOGS_DIR_PATH = _REPO_ROOT / "logs"


def cleanup_artifacts(run_id: str) -> dict:
    """Remove on-disk artefacts produced by a run. Best-effort.

    Cleans:
        checkpoints/<run_id>/         entire dir (covers PPO/DQN last.pt,
                                      supervisor + its _sla / _power files,
                                      CMDP Q_r/Q_c checkpoints)
        logs/<run_id>*                <run_id>.log, <run_id>.log.* (rotated),
                                      and any <run_id>/ sub-directory
    Datasets (data/processed/*.parquet) are NOT deleted — they are shared
    across CMDP runs, not per-run.
    """
    import shutil

    removed: dict[str, Any] = {"checkpoints": False, "logs": []}

    ckpt = _CHECKPOINTS_DIR / run_id
    if ckpt.exists() and ckpt.is_dir():
        try:
            shutil.rmtree(ckpt)
            removed["checkpoints"] = True
        except OSError:
            pass

    if _LOGS_DIR_PATH.exists():
        for entry in _LOGS_DIR_PATH.glob(f"{run_id}*"):
            try:
                if entry.is_dir():
                    shutil.rmtree(entry)
                else:
                    entry.unlink()
                removed["logs"].append(entry.name)
            except OSError:
                pass
    return removed


def purge_job_state(run_id: str) -> None:
    """Forget a run's in-memory tracker + WebSocket bus subscribers.

    Called after the DB row and disk artefacts are gone. If a worker
    thread is still running for this run_id, it'll find its DB row
    missing on the next status update and exit gracefully (we already
    requested cancellation upstream).
    """
    with _jobs_lock:
        _jobs.pop(run_id, None)
    try:
        bus.close_run(run_id)
    except Exception:
        pass


def start_training(req: TrainRequest) -> str:
    workload = req.trace_family if req.use_real_traces else "synthetic"
    run_id = _new_run_id(
        req.agent, req.cluster_type, req.n_servers, req.seed,
        req.n_train_seeds, req.n_test_seeds,
        workload=workload,
    )
    created_at = datetime.now(timezone.utc).isoformat()

    # Snapshot the cluster's actual idle/ceiling power so optimal-reward
    # is correct for heterogeneous runs (where per-server p_max varies).
    from environment.fleet import build_fleet
    _preview_servers = build_fleet(
        num_servers=req.n_servers,
        cluster_type=req.cluster_type,
        seed=env_config.FLEET_CLUSTER_SEED,
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
        payload = {
            "event": "episode",
            "episode": stats.episode,
            "reward": stats.reward,
            "power": stats.power,
            "sla_violations": stats.sla_violations,
            "steps": stats.steps,
        }
        # Only include agent-specific fields when populated — PPO and the
        # supervisor's REINFORCE head don't have an ε; sending None forces
        # the frontend to render NaNs in the live charts.
        if stats.epsilon is not None:
            payload["epsilon"] = stats.epsilon
        if stats.loss_mean is not None:
            payload["loss_mean"] = stats.loss_mean
        bus.publish_threadsafe(run_id, payload)

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
        # Test seeds:  [seed + TEST_SEED_OFFSET, ..., seed + TEST_SEED_OFFSET + n_test_seeds - 1]
        # The offset (default 1M, see env_config) guarantees no overlap regardless of n_train_seeds.
        #
        # Pure-replay trace families (alibaba / google_v2 / google_v3) ignore
        # seeds — replay is deterministic, so seed pools are meaningless and
        # we collapse to a single trajectory. The trace-sampled family is
        # synthetic Poisson timing on top of trace-derived templates and DOES
        # honor seeds — treat it like the synthetic generator.
        PURE_REPLAY_FAMILIES = {"alibaba", "google_v2", "google_v3"}
        is_pure_replay = req.use_real_traces and req.trace_family in PURE_REPLAY_FAMILIES
        if is_pure_replay:
            train_seeds: list[int] = []
            test_seeds: list[int] = []
        else:
            train_seeds = [req.seed + i for i in range(req.n_train_seeds)]
            test_seeds = [req.seed + env_config.TEST_SEED_OFFSET + i for i in range(req.n_test_seeds)]

        env = CloudClusterEnv(
            num_servers=req.n_servers,
            workload_generator=_build_workload(req),
            episode_length=req.episode_length,
            reward_alpha=req.alpha,
            reward_beta=req.beta,
            cluster_type=req.cluster_type,
            cluster_seed=env_config.FLEET_CLUSTER_SEED,  # Fixed across all runs of same N → fair comparison.
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
                run_name=run_id,
                log_dir=LOG_DIR,
                train_seeds=train_seeds,
            )
        elif isinstance(agent, PPOAgent):
            ep_stats = agent.train_loop(
                env=env, total_steps=req.total_steps,
                on_episode_end=episode_hook,
                train_seeds=train_seeds,
                run_name=run_id,
                log_dir=LOG_DIR,
            )
        else:
            trainer = Trainer(env=env, agent=agent, run_name=run_id, log_dir=LOG_DIR)
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
        # If test_seeds is non-empty: eval on those (proper ML-style protocol).
        # Else if eval_episodes > 0: legacy fallback on the training seed.
        # Else: skip — calling evaluate(n_episodes=0) silently returns NaN/0
        # metrics that look like a real result of zero, masking the misconfig.
        eval_data = None
        run_eval = bool(test_seeds) or req.eval_episodes > 0
        if run_eval:
            eval_env = CloudClusterEnv(
                num_servers=req.n_servers,
                workload_generator=_build_workload(req),
                episode_length=req.episode_length,
                reward_alpha=req.alpha,
                reward_beta=req.beta,
                cluster_type=req.cluster_type,
                cluster_seed=env_config.FLEET_CLUSTER_SEED,
            )
            if test_seeds:
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


# ─────────────────────────────────────────────────────────────────────────
# Offline / CMDP training path
# ─────────────────────────────────────────────────────────────────────────


def _ensure_offline_dataset(
    path: Path, n_servers: int, episode_length: int,
    cluster_type: str, episodes: int, seed: int,
    random_eps: float = 0.10,
    behavior_weights: tuple[float, float, float] = (0.34, 0.33, 0.33),
) -> None:
    """If the parquet file is missing, roll out the heuristic-mix behavior
    policy through the synthetic env and write transitions to disk."""
    if path.exists():
        return
    from agents.base_agent import compute_action_mask
    from agents.heuristics import (
        FirstFitDecreasingAgent,
        RoundRobinAgent,
        ShortestJobFirstAgent,
    )
    import pandas as pd

    rng = np.random.default_rng(seed)
    workload = SyntheticWorkloadGenerator(seed=seed)
    env = CloudClusterEnv(
        num_servers=n_servers,
        workload_generator=workload,
        episode_length=episode_length,
        cluster_type=cluster_type,
        cluster_seed=env_config.FLEET_CLUSTER_SEED,
    )
    n_actions = int(env.action_space.n)
    queue_size = env.job_queue_size
    policies = [
        RoundRobinAgent(n_servers=n_servers, queue_size=queue_size),
        ShortestJobFirstAgent(n_servers=n_servers, queue_size=queue_size),
        FirstFitDecreasingAgent(n_servers=n_servers, queue_size=queue_size),
    ]
    w = np.asarray(behavior_weights, dtype=np.float64)
    if w.sum() <= 0:
        w = np.array([1.0, 1.0, 1.0])
    weights = w / w.sum()
    max_cluster_power = env_config.P_MAX * n_servers

    rows = []
    for _ in range(episodes):
        obs, _ = env.reset()
        mask = compute_action_mask(env)
        for pol in policies:
            pol.on_episode_end()
        while True:
            if rng.random() < random_eps:
                legal = np.flatnonzero(mask)
                action = int(rng.choice(legal)) if len(legal) else 0
            else:
                idx = int(rng.choice(len(policies), p=weights))
                action = policies[idx].select_action(obs, mask, greedy=False)
            next_obs, reward, terminated, truncated, info = env.step(action)
            next_mask = compute_action_mask(env)
            done = terminated or truncated
            rows.append({
                "state": obs.astype(np.float32).tolist(),
                "action": int(action),
                "action_mask": mask.astype(bool).tolist(),
                "next_state": next_obs.astype(np.float32).tolist(),
                "next_action_mask": next_mask.astype(bool).tolist(),
                "reward": float(reward),
                "cost": float(info.get("step_sla_violations", 0)),
                "power": float(info.get("step_power", 0.0)),
                "power_norm": float(info.get("step_power", 0.0) / max_cluster_power),
                "done": bool(done),
            })
            obs = next_obs
            mask = next_mask
            if done:
                break
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)


def start_offline_training(req: OfflineTrainRequest) -> str:
    run_id = _new_run_id(
        "cmdp", req.cluster_type, req.n_servers, req.seed,
        n_train_seeds=0, n_test_seeds=req.n_test_seeds,
        workload="offline",
    )
    created_at = datetime.now(timezone.utc).isoformat()

    from environment.fleet import build_fleet
    _preview_servers = build_fleet(
        num_servers=req.n_servers,
        cluster_type=req.cluster_type,
        seed=env_config.FLEET_CLUSTER_SEED,
    )
    sum_p_idle = sum(s.p_idle for s in _preview_servers)
    sum_p_max = sum(s.p_max for s in _preview_servers)

    config_dump = {
        **req.model_dump(),
        "agent": "cmdp",
        "p_idle": env_config.P_IDLE,
        "p_max": env_config.P_MAX,
        "sum_p_idle": sum_p_idle,
        "sum_p_max": sum_p_max,
        "invalid_action_penalty": env_config.INVALID_ACTION_PENALTY,
    }
    db.insert_experiment(
        run_id=run_id, agent="cmdp", status="pending",
        created_at=created_at, config=config_dump,
    )
    with _jobs_lock:
        _jobs[run_id] = {
            "run_id": run_id,
            "status": "pending",
            "progress": 0.0,
            "eta_seconds": None,
            "current_episode": 0,
            "total_episodes": req.iterations,
            "last_reward": None,
            "started_at": None,
            "cancel_requested": False,
            "error": None,
            "agent": "cmdp",
        }
    _executor.submit(_run_offline_job, run_id, req)
    return run_id


def _run_offline_job(run_id: str, req: OfflineTrainRequest) -> None:
    started_at = time.time()
    log_fh = None
    try:
        Path(LOG_DIR).mkdir(parents=True, exist_ok=True)
        log_fh = open(Path(LOG_DIR) / f"{run_id}.log", "a", buffering=1)

        def _log(msg: str) -> None:
            print(msg)
            if log_fh is not None and not log_fh.closed:
                log_fh.write(msg + "\n")

        _log(f"[cmdp] run_id={run_id}")
        _log(f"[cmdp] config: iterations={req.iterations} batch_size={req.batch_size} "
             f"sla_budget={req.sla_budget} cql_alpha={req.cql_alpha} "
             f"λ_init={req.lambda_init} λ_lr={req.lambda_lr} "
             f"dual_signal={req.dual_signal} dual_eval_seeds={req.dual_eval_seeds} "
             f"reward_kind={req.reward_kind} dataset={req.dataset_path} "
             f"n_servers={req.n_servers} cluster={req.cluster_type}")

        with _jobs_lock:
            _jobs[run_id]["status"] = "running"
            _jobs[run_id]["started_at"] = started_at
        db.update_experiment_status(run_id, "running")

        _seed_all(req.seed)

        repo_root = Path(__file__).resolve().parents[2]
        ds_path = (repo_root / req.dataset_path).resolve() if not Path(req.dataset_path).is_absolute() else Path(req.dataset_path)

        if req.auto_generate_dataset:
            existed = ds_path.exists()
            _log(f"[cmdp] dataset {'found' if existed else 'missing'} at {ds_path}")
            bus.publish_threadsafe(run_id, {
                "event": "log",
                "message": f"Ensuring offline dataset at {ds_path}…",
            })
            if not existed:
                _log(f"[cmdp] generating dataset: episodes={req.dataset_episodes} "
                     f"random_eps={req.dataset_random_eps} "
                     f"weights=(rr={req.behavior_weight_rr}, "
                     f"sjf={req.behavior_weight_sjf}, "
                     f"ffd={req.behavior_weight_ffd})")
            _ensure_offline_dataset(
                ds_path, req.n_servers, req.episode_length,
                req.cluster_type, req.dataset_episodes, req.seed,
                random_eps=req.dataset_random_eps,
                behavior_weights=(
                    req.behavior_weight_rr,
                    req.behavior_weight_sjf,
                    req.behavior_weight_ffd,
                ),
            )
            if not existed:
                _log(f"[cmdp] dataset generated at {ds_path} "
                     f"size={ds_path.stat().st_size/1e6:.2f} MB")

        from agents.offline.cmdp_agent import CMDPAgent
        from agents.offline.offline_trainer import load_dataset, _sample_batch

        dataset = load_dataset(str(ds_path), reward_kind=req.reward_kind)
        _log(f"[cmdp] dataset loaded: n={dataset.n} obs_dim={dataset.obs_dim} "
             f"n_actions={dataset.n_actions}")

        # Unit-fix: cap step cost at 1.0 so Q_c learns a violation indicator
        # rather than an unbounded count.
        dataset.cost = np.minimum(dataset.cost, 1.0).astype(np.float32)

        # Scale-fix: Q_c is the *discounted return* of cost, with horizon
        # 1/(1-γ) ≈ 100 at γ=0.99. Without rescaling, Q_c ∈ [0, ~100] while
        # ε is a per-step rate (≪1) → the constraint is unreachable at the
        # native scale and λ rises monotonically forever. Multiplying cost
        # by (1−γ) collapses the geometric series to an *average*, so
        # Q_c ∈ [0, 1] and ε becomes the per-step violation rate budget,
        # directly comparable to the eval-time SLA rate.
        dataset.cost = ((1.0 - env_config.GAMMA) * dataset.cost).astype(np.float32)

        agent = CMDPAgent(
            state_dim=dataset.obs_dim,
            n_actions=dataset.n_actions,
            sla_budget=req.sla_budget,
            cql_alpha=req.cql_alpha,
            lambda_init=req.lambda_init,
            lambda_lr=req.lambda_lr,
        )

        # Build a small env for empirical-rate dual rollouts. Reused across
        # FQI iterations — env reset is cheap, env construction is not.
        dual_env = None
        if req.dual_signal == "empirical":
            dual_env = CloudClusterEnv(
                num_servers=req.n_servers,
                workload_generator=SyntheticWorkloadGenerator(seed=req.seed),
                episode_length=min(req.episode_length, req.dual_eval_steps),
                reward_alpha=req.alpha,
                reward_beta=req.beta,
                cluster_type=req.cluster_type,
                cluster_seed=env_config.FLEET_CLUSTER_SEED,
            )

        def _empirical_sla_rate() -> float:
            """Run greedy Lagrangian policy on a few dual-eval seeds, return mean step-SLA rate."""
            from agents.base_agent import compute_action_mask
            total_viol = 0
            total_steps = 0
            # Use a separate seed pool so dual rollouts don't overlap with
            # the held-out test seeds used for final eval.
            base = req.seed + env_config.TEST_SEED_OFFSET + 10_000
            for k in range(req.dual_eval_seeds):
                obs, _ = dual_env.reset(seed=base + k)
                mask = compute_action_mask(dual_env)
                steps = 0
                while steps < req.dual_eval_steps:
                    a = agent.select_action(obs, mask, greedy=True)
                    obs, _, term, trunc, info = dual_env.step(a)
                    mask = compute_action_mask(dual_env)
                    total_viol += int(info.get("step_sla_violations", 0))
                    total_steps += 1
                    steps += 1
                    if term or trunc:
                        break
            return (total_viol / total_steps) if total_steps > 0 else 0.0

        rng = np.random.default_rng(req.seed)
        log_every = max(50, req.iterations // 200)
        for it in range(1, req.iterations + 1):
            with _jobs_lock:
                if _jobs[run_id].get("cancel_requested"):
                    raise RuntimeError("cancelled by user")
            batch = _sample_batch(dataset, req.batch_size, agent.device, rng)
            losses = agent.fqi_step(batch)
            violation = None
            emp_rate = None
            if it % req.dual_update_freq == 0:
                if req.dual_signal == "empirical" and dual_env is not None:
                    emp_rate = _empirical_sla_rate()
                    violation = agent.dual_step_empirical(emp_rate)
                else:
                    violation = agent.dual_step(batch)
            if it % req.target_update_freq == 0:
                agent.hard_update_targets()

            if it % log_every == 0 or it == 1:
                with _jobs_lock:
                    job = _jobs.get(run_id)
                    if job is not None:
                        job["current_episode"] = it
                        job["progress"] = min(1.0, it / max(1, req.iterations))
                        elapsed = time.time() - started_at
                        rate = it / elapsed if elapsed > 0 else 0.0
                        job["eta_seconds"] = (
                            (req.iterations - it) / rate if rate > 0 else None
                        )
                db.insert_cmdp_iteration(
                    run_id=run_id, iteration=it,
                    loss_r=float(losses["loss_r"]),
                    loss_c=float(losses["loss_c"]),
                    lambda_val=float(agent.lam),
                    violation=(float(violation) if violation is not None else None),
                )
                bus.publish_threadsafe(run_id, {
                    "event": "iteration",
                    "iteration": it,
                    "loss_r": losses["loss_r"],
                    "loss_c": losses["loss_c"],
                    "lambda": agent.lam,
                    "constraint_violation": violation,
                })
                _log(
                    f"[cmdp] it {it:6d}/{req.iterations} "
                    f"loss_r={losses['loss_r']:.4f} loss_c={losses['loss_c']:.4f} "
                    f"λ={agent.lam:.3f}"
                    + (f" viol={violation:+.4f}" if violation is not None else "")
                    + (f" rate={emp_rate*100:.2f}%" if emp_rate is not None else "")
                )

        ckpt_dir = Path("checkpoints") / run_id
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        try:
            agent.save(str(ckpt_dir / "last.pt"))
        except Exception:
            pass
        try:
            gcp_service.upload_run_checkpoints(run_id, ckpt_dir)
        except Exception:
            pass

        _log(f"[cmdp] FQI complete. λ_final={agent.lam:.3f}")

        # Eval against the same env shape on a held-out seed pool.
        eval_data = None
        if req.n_test_seeds > 0:
            _log(f"[cmdp] eval on {req.n_test_seeds} held-out seeds…")
            eval_env = CloudClusterEnv(
                num_servers=req.n_servers,
                workload_generator=SyntheticWorkloadGenerator(seed=req.seed),
                episode_length=req.episode_length,
                reward_alpha=req.alpha,
                reward_beta=req.beta,
                cluster_type=req.cluster_type,
                cluster_seed=env_config.FLEET_CLUSTER_SEED,
            )
            test_seeds = [
                req.seed + env_config.TEST_SEED_OFFSET + i
                for i in range(req.n_test_seeds)
            ]
            ev = evaluate(agent, eval_env, workload_seeds=test_seeds)
            eval_data = ev.as_row()
            _log(
                f"[cmdp] eval done: mean_R={ev.mean_reward:.2f} "
                f"std_R={ev.std_reward:.2f} mean_power={ev.mean_power:.1f} "
                f"sla_rate={ev.sla_violation_rate*100:.2f}% "
                f"jobs/ep={ev.mean_jobs_completed:.1f}"
            )
            # Persist eval episodes so the Results page has *something* to plot.
            for i, (r, p, s) in enumerate(zip(
                ev.per_episode_rewards, ev.per_episode_power, ev.per_episode_sla
            )):
                db.insert_episode(
                    run_id=run_id, episode=i, reward=float(r), power=float(p),
                    sla_violations=int(s), steps=int(ev.mean_steps),
                )

        # Diagnostic: mean Q_c(s, π(s)) over a sample of dataset states.
        # If the constraint is tight, this should sit just above sla_budget.
        # If it's well below, the constraint was inactive (the policy never
        # had to pay for SLA). If it's well above, the policy is violating
        # and λ either hasn't ramped up or the offline data lacks the
        # cheap-SLA actions the policy would need to switch to.
        import torch as _torch
        with _torch.no_grad():
            n_diag = min(4096, dataset.n)
            idx = np.random.default_rng(req.seed).integers(0, dataset.n, size=n_diag)
            s_diag = _torch.from_numpy(dataset.state[idx]).to(agent.device)
            mask_diag = _torch.from_numpy(dataset.action_mask[idx]).to(agent.device)
            q_lag = agent.q_r_online(s_diag) - agent.lam * agent.q_c_online(s_diag)
            q_lag = q_lag.masked_fill(~mask_diag, float("-inf"))
            a_star = q_lag.argmax(dim=1, keepdim=True)
            mean_qc_pi = float(
                agent.q_c_online(s_diag).gather(1, a_star).squeeze(1).mean().item()
            )

        summary = {
            "n_iterations": req.iterations,
            "lambda_final": agent.lam,
            "sla_budget": req.sla_budget,
            "mean_qc_pi": mean_qc_pi,
            "constraint_slack": req.sla_budget - mean_qc_pi,
        }
        _log(
            f"[cmdp] diagnostics: mean_qc_pi={mean_qc_pi:.4f} "
            f"slack={req.sla_budget - mean_qc_pi:+.4f} (ε={req.sla_budget})"
        )
        _log(f"[cmdp] DONE run_id={run_id} elapsed={time.time() - started_at:.1f}s")
        with _jobs_lock:
            _jobs[run_id]["status"] = "completed"
            _jobs[run_id]["progress"] = 1.0
        db.update_experiment_status(
            run_id, "completed", summary=summary, eval_data=eval_data,
        )
        bus.publish_threadsafe(run_id, {"event": "completed", **summary})

    except Exception as e:
        err = f"{type(e).__name__}: {e}\n{traceback.format_exc()}"
        if log_fh is not None and not log_fh.closed:
            try:
                log_fh.write(f"[cmdp] FAILED: {err}\n")
            except Exception:
                pass
        with _jobs_lock:
            _jobs[run_id]["status"] = "failed"
            _jobs[run_id]["error"] = err
        db.update_experiment_status(run_id, "failed", error=err)
        bus.publish_threadsafe(run_id, {"event": "failed", "error": str(e)})
    finally:
        bus.close_run(run_id)
        if log_fh is not None and not log_fh.closed:
            log_fh.close()
