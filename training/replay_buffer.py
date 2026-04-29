from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray  # bool[n_actions], True = legal


class ReplayBuffer:
    """Circular replay buffer for off-policy RL agents (DQN)."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self._buf: deque[Transition] = deque(maxlen=capacity)

    def push(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
        next_action_mask: np.ndarray,
    ) -> None:
        self._buf.append(
            Transition(
                state=state.astype(np.float32),
                action=int(action),
                reward=float(reward),
                next_state=next_state.astype(np.float32),
                done=bool(done),
                next_action_mask=next_action_mask.astype(bool),
            )
        )

    def __len__(self) -> int:
        return len(self._buf)

    def sample(self, batch_size: int, device: torch.device) -> dict[str, torch.Tensor]:
        idx = np.random.choice(len(self._buf), size=batch_size, replace=False)
        batch = [self._buf[i] for i in idx]
        return {
            "state": torch.from_numpy(np.stack([t.state for t in batch])).to(device),
            "action": torch.tensor([t.action for t in batch], dtype=torch.long, device=device),
            "reward": torch.tensor([t.reward for t in batch], dtype=torch.float32, device=device),
            "next_state": torch.from_numpy(np.stack([t.next_state for t in batch])).to(device),
            "done": torch.tensor([t.done for t in batch], dtype=torch.float32, device=device),
            "next_action_mask": torch.from_numpy(
                np.stack([t.next_action_mask for t in batch])
            ).to(device),
        }
