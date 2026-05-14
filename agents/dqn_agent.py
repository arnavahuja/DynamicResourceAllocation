from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from environment import config
from training.replay_buffer import ReplayBuffer


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, n_actions: int, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, n_actions),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class DQNAgent(BaseAgent):
    """Double DQN with experience replay, target network, ε-greedy, and
    invalid-action masking. Hyperparameters come from environment.config."""

    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = config.GAMMA,
        lr: float = config.LEARNING_RATE,
        batch_size: int = config.BATCH_SIZE,
        buffer_size: int = config.REPLAY_BUFFER_SIZE,
        target_update_freq: int = config.TARGET_UPDATE_FREQ,
        eps_start: float = config.EPSILON_START,
        eps_end: float = config.EPSILON_END,
        eps_decay: float = config.EPSILON_DECAY,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update_freq = target_update_freq
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.online = QNetwork(state_dim, n_actions).to(self.device)
        self.target = QNetwork(state_dim, n_actions).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        for p in self.target.parameters():
            p.requires_grad = False

        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=lr)
        self.buffer = ReplayBuffer(buffer_size)
        self._train_steps = 0

    # ---------- BaseAgent ----------

    def select_action(
        self,
        state: np.ndarray,
        action_mask: np.ndarray,
        greedy: bool = False,
    ) -> int:
        legal = np.flatnonzero(action_mask)
        if len(legal) == 0:
            return 0  # should not happen — caller guarantees at least one True
        if not greedy and np.random.rand() < self.eps:
            return int(np.random.choice(legal))
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            q = self.online(s).squeeze(0).cpu().numpy()
        q_masked = np.where(action_mask, q, -np.inf)
        return int(np.argmax(q_masked))

    def on_step(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> dict:
        self.buffer.push(state, action, reward, next_state, done, next_action_mask)
        loss_val = self._learn()
        self._train_steps += 1
        if self._train_steps % self.target_update_freq == 0:
            self.target.load_state_dict(self.online.state_dict())
        # decay ε per step
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return {"loss": loss_val, "epsilon": self.eps}

    # ---------- internals ----------

    def _learn(self) -> float | None:
        if len(self.buffer) < self.batch_size:
            return None
        batch = self.buffer.sample(self.batch_size, self.device)
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"]
        s_next = batch["next_state"]
        done = batch["done"]
        next_mask = batch["next_action_mask"]

        # Q(s, a)
        q_sa = self.online(s).gather(1, a.unsqueeze(1)).squeeze(1)

        # Double DQN: action via online net, value via target net (masked)
        with torch.no_grad():
            q_next_online = self.online(s_next)
            q_next_online = q_next_online.masked_fill(~next_mask, float("-inf"))
            a_star = q_next_online.argmax(dim=1, keepdim=True)
            q_next_target = self.target(s_next).gather(1, a_star).squeeze(1)
            target = r + self.gamma * (1.0 - done) * q_next_target

        loss = F.smooth_l1_loss(q_sa, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), max_norm=10.0)
        self.optimizer.step()
        return float(loss.item())

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        torch.save(
            {
                "online": self.online.state_dict(),
                "target": self.target.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "eps": self.eps,
                "train_steps": self._train_steps,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.online.load_state_dict(ckpt["online"])
        self.target.load_state_dict(ckpt["target"])
        self.optimizer.load_state_dict(ckpt["optimizer"])
        self.eps = ckpt.get("eps", self.eps_end)
        self._train_steps = ckpt.get("train_steps", 0)
