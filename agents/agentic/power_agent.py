from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from agents.dqn_agent import DQNAgent


class PowerAgent(DQNAgent):
    """DQN sub-agent specialised for power minimisation.

    Identical structure to SLAAgent; differs only in the training reward
    fed by the supervisor (R_power = -normalized_power_this_step).
    """

    def recommend(
        self, state: np.ndarray, action_mask: np.ndarray
    ) -> tuple[int, float]:
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            q = self.online(s).squeeze(0)
            mask_t = torch.from_numpy(action_mask).to(self.device)
            q_masked = q.masked_fill(~mask_t, float("-inf"))
            probs = F.softmax(q_masked, dim=-1)
            action = int(torch.argmax(q_masked).item())
            confidence = float(probs[action].item())
        return action, confidence
