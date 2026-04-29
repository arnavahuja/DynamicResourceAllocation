from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from agents.base_agent import BaseAgent, compute_action_mask
from environment import config
from environment.cluster_env import CloudClusterEnv


class ActorCritic(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.policy_head = nn.Linear(hidden, n_actions)
        self.value_head = nn.Linear(hidden, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        return self.policy_head(h), self.value_head(h).squeeze(-1)


@dataclass
class _Rollout:
    states: list[np.ndarray]
    actions: list[int]
    log_probs: list[float]
    values: list[float]
    rewards: list[float]
    dones: list[bool]
    masks: list[np.ndarray]


class PPOAgent(BaseAgent):
    """PPO with clipped surrogate, GAE-λ, masked categorical policy, and
    entropy bonus. Implements its own outer training loop because PPO is
    on-policy (the generic Trainer's per-step replay model doesn't apply)."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = config.GAMMA,
        lr: float = config.LEARNING_RATE,
        clip: float = config.PPO_CLIP,
        epochs: int = config.PPO_EPOCHS,
        batch_size: int = config.BATCH_SIZE,
        gae_lambda: float = 0.95,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        rollout_steps: int = 2048,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.clip = clip
        self.epochs = epochs
        self.batch_size = batch_size
        self.gae_lambda = gae_lambda
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.rollout_steps = rollout_steps

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.net = ActorCritic(state_dim, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr)

    # ---------- BaseAgent ----------

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            logits, _ = self.net(s)
            mask_t = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
            logits = logits.masked_fill(~mask_t, float("-inf"))
            if greedy:
                return int(torch.argmax(logits, dim=-1).item())
            probs = F.softmax(logits, dim=-1)
            dist = Categorical(probs=probs)
            return int(dist.sample().item())

    # ---------- core PPO loop ----------

    def _action_and_value(
        self, state: np.ndarray, action_mask: np.ndarray
    ) -> tuple[int, float, float]:
        s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
        m = torch.from_numpy(action_mask).unsqueeze(0).to(self.device)
        logits, value = self.net(s)
        logits = logits.masked_fill(~m, float("-inf"))
        dist = Categorical(logits=logits)
        a = dist.sample()
        return int(a.item()), float(dist.log_prob(a).item()), float(value.item())

    def _compute_gae(
        self,
        rewards: list[float],
        values: list[float],
        dones: list[bool],
        last_value: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        T = len(rewards)
        adv = np.zeros(T, dtype=np.float32)
        gae = 0.0
        next_value = last_value
        for t in reversed(range(T)):
            non_terminal = 1.0 - float(dones[t])
            delta = rewards[t] + self.gamma * next_value * non_terminal - values[t]
            gae = delta + self.gamma * self.gae_lambda * non_terminal * gae
            adv[t] = gae
            next_value = values[t]
        returns = adv + np.array(values, dtype=np.float32)
        return adv, returns

    def _ppo_update(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        old_log_probs: np.ndarray,
        advantages: np.ndarray,
        returns: np.ndarray,
        masks: np.ndarray,
    ) -> dict[str, float]:
        s = torch.from_numpy(states).to(self.device)
        a = torch.from_numpy(actions).long().to(self.device)
        old_lp = torch.from_numpy(old_log_probs).to(self.device)
        adv = torch.from_numpy(advantages).to(self.device)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        ret = torch.from_numpy(returns).to(self.device)
        m = torch.from_numpy(masks).to(self.device)

        n = s.shape[0]
        idx = np.arange(n)
        losses = {"policy": 0.0, "value": 0.0, "entropy": 0.0}
        n_batches = 0

        for _ in range(self.epochs):
            np.random.shuffle(idx)
            for start in range(0, n, self.batch_size):
                b = idx[start : start + self.batch_size]
                bi = torch.from_numpy(b).long().to(self.device)
                logits, values = self.net(s[bi])
                logits = logits.masked_fill(~m[bi], float("-inf"))
                dist = Categorical(logits=logits)
                lp = dist.log_prob(a[bi])
                entropy = dist.entropy().mean()

                ratio = torch.exp(lp - old_lp[bi])
                surr1 = ratio * adv[bi]
                surr2 = torch.clamp(ratio, 1.0 - self.clip, 1.0 + self.clip) * adv[bi]
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = F.mse_loss(values, ret[bi])
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=0.5)
                self.optimizer.step()

                losses["policy"] += float(policy_loss.item())
                losses["value"] += float(value_loss.item())
                losses["entropy"] += float(entropy.item())
                n_batches += 1

        if n_batches:
            for k in losses:
                losses[k] /= n_batches
        return losses

    def train_loop(
        self,
        env: CloudClusterEnv,
        total_steps: int,
        on_episode_end: Callable | None = None,
    ) -> list[dict]:
        """PPO outer loop. Collects rollouts of `rollout_steps` and updates.

        Returns a list of per-episode summary dicts.
        """
        from training.trainer import EpisodeStats  # avoid circular import

        ep_stats: list[dict] = []
        steps_done = 0
        obs, _ = env.reset()
        mask = compute_action_mask(env)
        ep_reward = 0.0
        ep_power = 0.0
        ep_sla = 0
        ep_steps = 0
        ep_idx = 0

        while steps_done < total_steps:
            roll = _Rollout([], [], [], [], [], [], [])
            for _ in range(self.rollout_steps):
                action, log_prob, value = self._action_and_value(obs, mask)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                next_mask = compute_action_mask(env)

                roll.states.append(obs)
                roll.actions.append(action)
                roll.log_probs.append(log_prob)
                roll.values.append(value)
                roll.rewards.append(float(reward))
                roll.dones.append(done)
                roll.masks.append(mask)

                ep_reward += reward
                ep_power += info.get("step_power", 0.0)
                ep_sla = info.get("sla_violations", ep_sla)
                ep_steps += 1
                steps_done += 1

                obs = next_obs
                mask = next_mask
                if done:
                    stats = EpisodeStats(
                        episode=ep_idx,
                        reward=ep_reward,
                        power=ep_power,
                        sla_violations=ep_sla,
                        jobs_completed=info.get("jobs_completed", 0),
                        steps=ep_steps,
                    )
                    ep_stats.append(stats.__dict__)
                    if on_episode_end is not None:
                        on_episode_end(stats)
                    ep_idx += 1
                    obs, _ = env.reset()
                    mask = compute_action_mask(env)
                    ep_reward = 0.0
                    ep_power = 0.0
                    ep_sla = 0
                    ep_steps = 0
                if steps_done >= total_steps:
                    break

            # bootstrap value for GAE
            with torch.no_grad():
                s_last = torch.from_numpy(obs.astype(np.float32)).unsqueeze(0).to(self.device)
                _, last_value = self.net(s_last)
                last_value = float(last_value.item())

            adv, ret = self._compute_gae(
                roll.rewards, roll.values, roll.dones, last_value
            )
            losses = self._ppo_update(
                states=np.stack(roll.states).astype(np.float32),
                actions=np.array(roll.actions, dtype=np.int64),
                old_log_probs=np.array(roll.log_probs, dtype=np.float32),
                advantages=adv,
                returns=ret,
                masks=np.stack(roll.masks),
            )
            print(
                f"[ppo] steps={steps_done:7d} "
                f"pol={losses['policy']:.4f} val={losses['value']:.4f} "
                f"ent={losses['entropy']:.4f} eps_collected={ep_idx}"
            )

        return ep_stats

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        torch.save(
            {"net": self.net.state_dict(), "optimizer": self.optimizer.state_dict()},
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.net.load_state_dict(ckpt["net"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
