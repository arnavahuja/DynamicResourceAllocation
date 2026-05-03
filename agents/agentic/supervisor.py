from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli

from agents.agentic.power_agent import PowerAgent
from agents.agentic.sla_agent import SLAAgent
from agents.base_agent import BaseAgent, compute_action_mask
from environment import config
from environment.cluster_env import CloudClusterEnv


# Supervisor input vector — kept small and interpretable per spec:
#   [sla_budget_remaining, current_power_level, time_in_episode]
SUPERVISOR_STATE_DIM = 3


class SupervisorNet(nn.Module):
    """Tiny MLP that outputs the probability of following the SLA agent."""

    def __init__(self, hidden: int = 32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(SUPERVISOR_STATE_DIM, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(x)).squeeze(-1)


def _supervisor_state(
    env: CloudClusterEnv,
    step_in_episode: int,
    sla_budget: float,
    last_power_norm: float,
) -> np.ndarray:
    return np.array(
        [
            sla_budget,
            last_power_norm,
            step_in_episode / max(1, env.episode_length),
        ],
        dtype=np.float32,
    )


class SupervisorAgent(BaseAgent):
    """Agentic-RL composite: SLA sub-agent + Power sub-agent + a learned
    supervisor that picks which sub-agent to follow at each step.

    Supervisor net: small MLP → P(follow SLA agent). Trained with REINFORCE
    on the combined env reward (-α·power - β·sla). Sub-agents trained as
    DQNs on their own decomposed reward signals.
    """

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        sla_budget_per_episode: int = 50,
        supervisor_lr: float = 1e-3,
        entropy_coef: float = 0.01,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.sla_budget_per_episode = sla_budget_per_episode
        self.entropy_coef = entropy_coef

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.sla_agent = SLAAgent(state_dim=state_dim, n_actions=n_actions, device=str(self.device))
        self.power_agent = PowerAgent(state_dim=state_dim, n_actions=n_actions, device=str(self.device))
        self.supervisor = SupervisorNet().to(self.device)
        self.sup_optim = torch.optim.Adam(self.supervisor.parameters(), lr=supervisor_lr)

        # buffers used during a rollout to compute the REINFORCE update
        self._ep_log_probs: list[torch.Tensor] = []
        self._ep_entropies: list[torch.Tensor] = []
        self._ep_rewards: list[float] = []

    # ---------- BaseAgent (inference-only path) ----------

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        """Inference: ask both sub-agents, follow whichever the supervisor's
        deterministic preference points to. Used by `evaluate(...)`.

        Note: supervisor_state during inference uses neutral defaults
        because the evaluator doesn't track per-step running totals.
        """
        sla_a, _ = self.sla_agent.recommend(state, action_mask)
        pwr_a, _ = self.power_agent.recommend(state, action_mask)
        sup_in = torch.tensor(
            [[1.0, 0.5, 0.5]], dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            w = self.supervisor(sup_in).item()
        return sla_a if w > 0.5 else pwr_a

    # ---------- training loop ----------

    def train_loop(
        self,
        env: CloudClusterEnv,
        episodes: int,
        on_episode_end: Callable | None = None,
        log_every: int = 10,
        run_name: str | None = None,
        log_dir: str = "logs",
        train_seeds: list[int] | None = None,
    ) -> list[dict]:
        from training.trainer import EpisodeStats  # local import avoids cycle

        max_cluster_power = config.P_MAX * env.num_servers
        ep_stats: list[dict] = []

        log_fh = None
        if run_name is not None:
            Path(log_dir).mkdir(parents=True, exist_ok=True)
            log_fh = open(Path(log_dir) / f"{run_name}.log", "a", buffering=1)

        def _log(msg: str) -> None:
            print(msg)
            if log_fh is not None:
                log_fh.write(msg + "\n")

        seed_rng = np.random.default_rng(0)

        for ep in range(episodes):
            if train_seeds:
                workload_seed = int(seed_rng.choice(train_seeds))
                obs, _info = env.reset(options={"workload_seed": workload_seed})
            else:
                obs, _info = env.reset()
            mask = compute_action_mask(env)
            sla_budget_remaining = float(self.sla_budget_per_episode)
            last_power_norm = 0.0

            log_probs: list[torch.Tensor] = []
            entropies: list[torch.Tensor] = []
            rewards: list[float] = []

            ep_reward = 0.0
            ep_power = 0.0
            ep_sla = 0
            steps = 0
            sla_loss_acc = 0.0
            pwr_loss_acc = 0.0
            updates = 0

            while True:
                # 1) sub-agents propose actions
                sla_a, _sla_conf = self.sla_agent.recommend(obs, mask)
                pwr_a, _pwr_conf = self.power_agent.recommend(obs, mask)

                # ε-greedy exploration: occasionally let sub-agents explore
                # using their own ε (already decayed inside DQNAgent).
                if np.random.rand() < self.sla_agent.eps:
                    legal = np.flatnonzero(mask)
                    sla_a = int(np.random.choice(legal)) if len(legal) else 0
                if np.random.rand() < self.power_agent.eps:
                    legal = np.flatnonzero(mask)
                    pwr_a = int(np.random.choice(legal)) if len(legal) else 0

                # 2) supervisor samples which sub-agent to follow
                sup_state = _supervisor_state(
                    env,
                    step_in_episode=steps,
                    sla_budget=sla_budget_remaining / max(1, self.sla_budget_per_episode),
                    last_power_norm=last_power_norm,
                )
                sup_in = torch.from_numpy(sup_state).unsqueeze(0).to(self.device)
                w = self.supervisor(sup_in)            # P(follow SLA)
                w = w.clamp(1e-4, 1 - 1e-4)
                dist = Bernoulli(probs=w)
                follow_sla = dist.sample()             # 1.0 → SLA, 0.0 → Power
                action = sla_a if follow_sla.item() > 0.5 else pwr_a

                log_probs.append(dist.log_prob(follow_sla))
                entropies.append(dist.entropy())

                # 3) env step
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_mask = compute_action_mask(env)

                # 4) decompose reward for sub-agents
                step_power = info.get("step_power", 0.0)
                step_sla = info.get("step_sla_violations", 0)
                power_norm = step_power / max_cluster_power
                last_power_norm = power_norm
                sla_budget_remaining = max(0.0, sla_budget_remaining - step_sla)

                r_sla = -float(step_sla)
                r_power = -float(power_norm)

                # 5) train sub-agents with their own reward
                m1 = self.sla_agent.on_step(
                    state=obs, action=sla_a, reward=r_sla,
                    next_state=next_obs, done=done, next_action_mask=next_mask,
                )
                m2 = self.power_agent.on_step(
                    state=obs, action=pwr_a, reward=r_power,
                    next_state=next_obs, done=done, next_action_mask=next_mask,
                )
                if m1.get("loss") is not None:
                    sla_loss_acc += m1["loss"]
                    updates += 1
                if m2.get("loss") is not None:
                    pwr_loss_acc += m2["loss"]

                rewards.append(float(reward))
                ep_reward += reward
                ep_power += step_power
                ep_sla = info.get("sla_violations", ep_sla)
                steps += 1

                obs = next_obs
                mask = next_mask
                if done:
                    break

            # 6) supervisor REINFORCE update on full-episode return
            returns = []
            G = 0.0
            for r in reversed(rewards):
                G = r + config.GAMMA * G
                returns.insert(0, G)
            ret_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
            ret_t = (ret_t - ret_t.mean()) / (ret_t.std() + 1e-8)
            log_probs_t = torch.cat(log_probs)
            entropy_t = torch.cat(entropies).mean()
            policy_loss = -(log_probs_t * ret_t).mean() - self.entropy_coef * entropy_t

            self.sup_optim.zero_grad()
            policy_loss.backward()
            nn.utils.clip_grad_norm_(self.supervisor.parameters(), max_norm=1.0)
            self.sup_optim.step()

            stats = EpisodeStats(
                episode=ep,
                reward=ep_reward,
                power=ep_power,
                sla_violations=ep_sla,
                jobs_completed=info.get("jobs_completed", 0),
                steps=steps,
                epsilon=self.sla_agent.eps,
                loss_mean=(sla_loss_acc / max(1, updates)) if updates else None,
            )
            ep_stats.append(stats.__dict__)

            if (ep + 1) % log_every == 0 or ep == 0:
                _log(
                    f"[supervisor] ep {ep+1:5d}/{episodes} "
                    f"R={ep_reward:9.2f} power={ep_power:8.0f} sla={ep_sla:4d} "
                    f"sup_loss={float(policy_loss.item()):.4f} "
                    f"sla_loss={(sla_loss_acc/max(1,updates)):.4f} "
                    f"pwr_loss={(pwr_loss_acc/max(1,updates)):.4f} "
                    f"eps={self.sla_agent.eps:.3f}"
                )
            if on_episode_end is not None:
                on_episode_end(stats)

        if log_fh is not None:
            log_fh.close()
        return ep_stats

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "supervisor": self.supervisor.state_dict(),
                "sup_optim": self.sup_optim.state_dict(),
            },
            path,
        )
        # sub-agents save next to the supervisor
        self.sla_agent.save(str(p.with_name(p.stem + "_sla.pt")))
        self.power_agent.save(str(p.with_name(p.stem + "_power.pt")))

    def load(self, path: str) -> None:
        p = Path(path)
        ckpt = torch.load(path, map_location=self.device)
        self.supervisor.load_state_dict(ckpt["supervisor"])
        self.sup_optim.load_state_dict(ckpt["sup_optim"])
        self.sla_agent.load(str(p.with_name(p.stem + "_sla.pt")))
        self.power_agent.load(str(p.with_name(p.stem + "_power.pt")))
        # turn off exploration for inference
        self.sla_agent.eps = 0.0
        self.power_agent.eps = 0.0
