"""Constrained MDP agent (Lagrangian relaxation) for offline learning.

Two Q-networks:
    Q_r(s,a)  — expected return of negative power (objective to maximise)
    Q_c(s,a)  — expected return of cost (SLA violations to constrain)

Policy is greedy w.r.t. the Lagrangian Q:
    Q_L(s,a) = Q_r(s,a) - λ · Q_c(s,a)

λ ≥ 0 is the dual variable, updated by gradient ascent on
    L(λ) = E[Q_c(s, π(s))] - ε_sla
so λ rises whenever the current policy violates the constraint and falls
when it has slack. Both Q-networks are trained with FQI / Double-DQN-style
Bellman targets on the *static* offline dataset — no environment access.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from agents.base_agent import BaseAgent
from agents.dqn_agent import QNetwork
from environment import config


class CMDPAgent(BaseAgent):
    def __init__(
        self,
        state_dim: int,
        n_actions: int,
        gamma: float = config.GAMMA,
        lr: float = config.LEARNING_RATE,
        sla_budget: float = 0.05,
        lambda_init: float = 1.0,
        lambda_lr: float = 1e-2,
        lambda_max: float = 1000.0,
        cql_alpha: float = 1.0,
        device: str | None = None,
    ):
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.gamma = gamma
        self.sla_budget = sla_budget          # ε_sla — per-step expected violations
        self.lambda_lr = lambda_lr
        self.lambda_max = lambda_max
        self.cql_alpha = cql_alpha            # weight on the conservative Q_c penalty

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self.q_r_online = QNetwork(state_dim, n_actions).to(self.device)
        self.q_r_target = QNetwork(state_dim, n_actions).to(self.device)
        self.q_c_online = QNetwork(state_dim, n_actions).to(self.device)
        self.q_c_target = QNetwork(state_dim, n_actions).to(self.device)
        for target, online in [
            (self.q_r_target, self.q_r_online),
            (self.q_c_target, self.q_c_online),
        ]:
            target.load_state_dict(online.state_dict())
            for p in target.parameters():
                p.requires_grad = False

        self.opt_r = torch.optim.Adam(self.q_r_online.parameters(), lr=lr)
        self.opt_c = torch.optim.Adam(self.q_c_online.parameters(), lr=lr)
        # Track λ as an unconstrained scalar; we project to ≥0 when reading.
        self.log_lambda = torch.tensor(np.log(lambda_init), device=self.device, requires_grad=False)

    # ---------- inference ----------

    @property
    def lam(self) -> float:
        return float(torch.exp(self.log_lambda).clamp(0, self.lambda_max).item())

    def _q_lagrangian(self, s: torch.Tensor) -> torch.Tensor:
        return self.q_r_online(s) - self.lam * self.q_c_online(s)

    def select_action(self, state: np.ndarray, action_mask: np.ndarray, greedy: bool = False) -> int:
        with torch.no_grad():
            s = torch.from_numpy(state.astype(np.float32)).unsqueeze(0).to(self.device)
            q = self._q_lagrangian(s).squeeze(0).cpu().numpy()
        q = np.where(action_mask, q, -np.inf)
        return int(np.argmax(q))

    # ---------- training updates ----------

    def fqi_step(self, batch: dict[str, torch.Tensor]) -> dict[str, float]:
        """One FQI gradient step on Q_r and Q_c."""
        s = batch["state"]
        a = batch["action"]
        r = batch["reward"]            # objective: -power (already shaped)
        c = batch["cost"]              # constraint: SLA violations (≥0)
        s_next = batch["next_state"]
        done = batch["done"]
        next_mask = batch["next_action_mask"]

        # --- Q_r update (Double DQN target) ---
        q_r_sa = self.q_r_online(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Pick action under current Lagrangian policy at s'
            q_next_lagrangian = (
                self.q_r_online(s_next) - self.lam * self.q_c_online(s_next)
            )
            q_next_lagrangian = q_next_lagrangian.masked_fill(~next_mask, float("-inf"))
            a_star = q_next_lagrangian.argmax(dim=1, keepdim=True)
            q_r_next = self.q_r_target(s_next).gather(1, a_star).squeeze(1)
            target_r = r + self.gamma * (1.0 - done) * q_r_next
        loss_r = F.smooth_l1_loss(q_r_sa, target_r)
        self.opt_r.zero_grad()
        loss_r.backward()
        nn.utils.clip_grad_norm_(self.q_r_online.parameters(), 10.0)
        self.opt_r.step()

        # --- Q_c update (same a_star) ---
        q_c_all = self.q_c_online(s)                                  # (B, n_actions)
        q_c_sa = q_c_all.gather(1, a.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            q_c_next = self.q_c_target(s_next).gather(1, a_star).squeeze(1)
            target_c = c + self.gamma * (1.0 - done) * q_c_next
        bellman_c = F.smooth_l1_loss(q_c_sa, target_c)

        # Conservative (CQL-style) penalty on Q_c. We want OOD actions to
        # look MORE costly so the Lagrangian-greedy policy avoids them.
        # The naive form  E[Q_c(a_data) − logsumexp_a Q_c(a)]  is ≤ 0 and
        # UNBOUNDED BELOW: minimising it lets the network drive logsumexp
        # to +∞, blowing up loss_c (we observed loss_c → −10¹² in practice).
        #
        # Correct pessimistic form: apply standard CQL to −Q_c.
        #   cql_c = E_s[ logsumexp_a(−Q_c(s, a)) + Q_c(s, a_data) ]
        # By log-sum-exp inequality this is ≥ 0 (bounded below). Gradient:
        #   ▼  Q_c on the LOW-Q_c (would-be-policy) action  → push UP
        #   ▲  Q_c(s, a_data)                                → push DOWN
        # which is exactly the pessimism direction we want — Q_c becomes
        # an upper bound on true cost on the policy's chosen actions.
        # Illegal actions: set −Q_c to −inf so they don't enter logsumexp.
        action_mask = batch.get("action_mask", batch["next_action_mask"])
        neg_q_c_for_lse = (-q_c_all).masked_fill(~action_mask, float("-inf"))
        logsumexp_neg_c = torch.logsumexp(neg_q_c_for_lse, dim=1)     # (B,)
        cql_c = (logsumexp_neg_c + q_c_sa).mean()                     # ≥ 0
        loss_c = bellman_c + self.cql_alpha * cql_c

        self.opt_c.zero_grad()
        loss_c.backward()
        nn.utils.clip_grad_norm_(self.q_c_online.parameters(), 10.0)
        self.opt_c.step()

        return {
            "loss_r": float(loss_r.item()),
            "loss_c": float(loss_c.item()),
            "loss_c_bellman": float(bellman_c.item()),
            "loss_c_cql": float(cql_c.item()),
        }

    def dual_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Update λ: gradient ascent on E[Q_c(s, π(s))] - sla_budget."""
        s = batch["state"]
        mask = batch["next_action_mask"]   # any boolean mask is fine here; we'll
                                           # use action_mask if available.
        if "action_mask" in batch:
            mask = batch["action_mask"]
        with torch.no_grad():
            q_lag = self.q_r_online(s) - self.lam * self.q_c_online(s)
            q_lag = q_lag.masked_fill(~mask, float("-inf"))
            a_star = q_lag.argmax(dim=1, keepdim=True)
            q_c_pi = self.q_c_online(s).gather(1, a_star).squeeze(1)
        constraint_violation = float(q_c_pi.mean().item()) - self.sla_budget
        # log_lambda += lr · violation; clamp >= log(1e-6) to avoid underflow
        new = self.log_lambda + self.lambda_lr * constraint_violation
        self.log_lambda = torch.clamp(new, min=np.log(1e-6), max=np.log(self.lambda_max))
        return constraint_violation

    def dual_step_empirical(self, sla_rate: float) -> float:
        """Dual ascent using an on-policy *empirical* SLA rate.

        Bypasses CQL's inflation of Q_c on OOD actions: the CQL-c penalty
        deliberately makes Q_c much larger than the true cost, so the
        Q_c-based dual signal saturates and λ never converges sensibly.
        Feeding the empirically observed step-SLA rate from a short
        rollout is the ground truth the user actually cares about.
        """
        constraint_violation = float(sla_rate) - self.sla_budget
        new = self.log_lambda + self.lambda_lr * constraint_violation
        self.log_lambda = torch.clamp(new, min=np.log(1e-6), max=np.log(self.lambda_max))
        return constraint_violation

    def hard_update_targets(self) -> None:
        self.q_r_target.load_state_dict(self.q_r_online.state_dict())
        self.q_c_target.load_state_dict(self.q_c_online.state_dict())

    # ---------- persistence ----------

    def save(self, path: str) -> None:
        torch.save(
            {
                "q_r": self.q_r_online.state_dict(),
                "q_c": self.q_c_online.state_dict(),
                "q_r_target": self.q_r_target.state_dict(),
                "q_c_target": self.q_c_target.state_dict(),
                "log_lambda": self.log_lambda.detach().cpu(),
                "sla_budget": self.sla_budget,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.q_r_online.load_state_dict(ckpt["q_r"])
        self.q_c_online.load_state_dict(ckpt["q_c"])
        self.q_r_target.load_state_dict(ckpt["q_r_target"])
        self.q_c_target.load_state_dict(ckpt["q_c_target"])
        self.log_lambda = ckpt["log_lambda"].to(self.device)
        self.sla_budget = ckpt.get("sla_budget", self.sla_budget)
