from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from agents.base_agent import BaseAgent, compute_action_mask
from environment.cluster_env import CloudClusterEnv


@dataclass
class EpisodeStats:
    episode: int
    reward: float
    power: float            # cumulative power-time over episode (W·timestep)
    sla_violations: int
    jobs_completed: int
    steps: int
    epsilon: float | None = None
    loss_mean: float | None = None
    duration_s: float = 0.0


@dataclass
class ExperimentResult:
    run_name: str
    agent_name: str
    episodes: list[EpisodeStats] = field(default_factory=list)
    best_reward: float = float("-inf")
    best_episode: int = -1


class Trainer:
    """Generic episode-based trainer for any BaseAgent.

    Off-policy agents should call `agent.on_step(...)` with full transition
    info; on-policy agents are expected to override or wrap this loop.
    """

    def __init__(
        self,
        env: CloudClusterEnv,
        agent: BaseAgent,
        run_name: str = "run",
        checkpoint_dir: str | Path = "checkpoints",
        save_every: int = 500,
        log_every: int = 10,
        wandb_logger=None,
        log_dir: str | Path = "logs",
    ):
        self.env = env
        self.agent = agent
        self.run_name = run_name
        self.checkpoint_dir = Path(checkpoint_dir) / run_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.save_every = save_every
        self.log_every = log_every
        self.wandb = wandb_logger
        # Mirror every printed log line to logs/<run_name>.log so the user can
        # post the file back for diagnosis instead of copy-pasting from the
        # terminal. Open in append mode in case of resumed runs.
        Path(log_dir).mkdir(parents=True, exist_ok=True)
        self.log_path = Path(log_dir) / f"{run_name}.log"
        self._log_fh = open(self.log_path, "a", buffering=1)  # line-buffered

    def train(self, episodes: int, on_episode_end=None, train_seeds: list[int] | None = None) -> ExperimentResult:
        try:
            return self._train_impl(episodes, on_episode_end, train_seeds)
        finally:
            # Always close the log handle, even if training crashes — otherwise
            # the partial log on disk is missing whatever the final exception
            # would have written through line-buffered flushes.
            if self._log_fh and not self._log_fh.closed:
                self._log_fh.close()

    def _train_impl(self, episodes: int, on_episode_end, train_seeds) -> ExperimentResult:
        result = ExperimentResult(
            run_name=self.run_name,
            agent_name=type(self.agent).__name__,
        )

        # If no pool supplied, fall back to whatever seed the workload was
        # built with (single-trajectory training — Path B legacy behavior).
        rng = np.random.default_rng(0)

        for ep in range(episodes):
            t_start = time.time()
            if train_seeds:
                workload_seed = int(rng.choice(train_seeds))
                obs, _info = self.env.reset(options={"workload_seed": workload_seed})
            else:
                obs, _info = self.env.reset()
            mask = compute_action_mask(self.env)
            ep_reward = 0.0
            ep_power = 0.0
            ep_sla = 0
            losses: list[float] = []
            last_eps: float | None = None
            steps = 0
            action_hist = np.zeros(int(self.env.action_space.n), dtype=np.int64)

            while True:
                action = self.agent.select_action(obs, mask, greedy=False)
                action_hist[action] += 1
                next_obs, reward, terminated, truncated, info = self.env.step(action)
                next_mask = compute_action_mask(self.env)
                done = terminated or truncated

                step_metrics = self.agent.on_step(
                    state=obs,
                    action=action,
                    reward=reward,
                    next_state=next_obs,
                    done=done,
                    next_action_mask=next_mask,
                ) or {}
                if step_metrics.get("loss") is not None:
                    losses.append(step_metrics["loss"])
                last_eps = step_metrics.get("epsilon", last_eps)

                ep_reward += reward
                ep_power += info.get("step_power", 0.0)
                ep_sla = info.get("sla_violations", ep_sla)
                steps += 1

                obs = next_obs
                mask = next_mask
                if done:
                    break

            self.agent.on_episode_end()

            stats = EpisodeStats(
                episode=ep,
                reward=ep_reward,
                power=ep_power,
                sla_violations=ep_sla,
                jobs_completed=info.get("jobs_completed", 0),
                steps=steps,
                epsilon=last_eps,
                loss_mean=float(np.mean(losses)) if losses else None,
                duration_s=time.time() - t_start,
            )
            result.episodes.append(stats)

            if ep_reward > result.best_reward:
                result.best_reward = ep_reward
                result.best_episode = ep
                self.agent.save(str(self.checkpoint_dir / "best.pt"))

            if (ep + 1) % self.save_every == 0:
                self.agent.save(str(self.checkpoint_dir / "last.pt"))

            if (ep + 1) % self.log_every == 0 or ep == 0:
                eps_str = f" eps={last_eps:.3f}" if last_eps is not None else ""
                loss_str = f" loss={stats.loss_mean:.4f}" if stats.loss_mean else ""
                # Marginal histograms over servers, queue-position, and wait —
                # the joint K*N+1 vector is too wide to print at log_every cadence.
                N = self.env.num_servers
                K = self.env.job_queue_size
                wait_count = int(action_hist[N * K])
                joint = action_hist[: N * K].reshape(K, N)
                server_marg = joint.sum(axis=0).tolist()
                queue_marg = joint.sum(axis=1).tolist()
                hist_str = (
                    f" wait={wait_count}"
                    f" srv={server_marg}"
                    f" q={queue_marg}"
                )
                line = (
                    f"[{self.run_name}] ep {ep+1:5d}/{episodes} "
                    f"R={ep_reward:9.2f} power={ep_power:8.0f} "
                    f"sla={ep_sla:4d} steps={steps:4d}{eps_str}{loss_str}{hist_str}"
                )
                print(line)
                self._log_fh.write(line + "\n")

            if self.wandb is not None:
                self.wandb.log(
                    {
                        "train/episode_reward": ep_reward,
                        "train/power_consumption": ep_power,
                        "train/sla_violations": ep_sla,
                        "train/steps": steps,
                        **({"train/epsilon": last_eps} if last_eps is not None else {}),
                        **({"train/loss": stats.loss_mean} if stats.loss_mean else {}),
                    },
                    step=ep,
                )

            if on_episode_end is not None:
                on_episode_end(stats)

        # final checkpoint
        self.agent.save(str(self.checkpoint_dir / "last.pt"))
        return result
