# Dynamic Resource Allocation in Cloud Computing via Deep RL

Columbia COMS Advanced Reinforcement Learning final project. A unified
benchmark for energy- and SLA-aware job-to-server scheduling in a
heterogeneous cloud cluster, comparing classical heuristics, online deep
RL (Double DQN, PPO, hierarchical "Agentic" RL), and an offline
Constrained MDP trained with Lagrangian relaxation + CQL pessimism.

## What's in here

```
agents/         RR / SJF / FFD heuristics, DQN, PPO, agentic supervisor, offline CMDP
environment/    Cluster MDP — fleet, server (with sleep/wake), workload generators
training/       Training driver and evaluator (50-train / 50-test seed protocol)
backend/        FastAPI service: launches runs, streams progress, persists to SQLite
frontend/       React + Vite dashboard for configuring, monitoring, and comparing runs
data/           Workload generators + Google v2 trace ingestion
scripts/        One-shot utilities (offline dataset generation, sweeps, etc.)
tests/          Unit/integration tests for env and agents
gcp/, docker/   Optional cloud + container deployment artefacts
```

## Highlights

- **Heterogeneous cluster model** with per-tier power curves
  `P(u) = P_idle + (P_max − P_idle) · u^α` and a non-preemptive job model.
- **Sleep/wake action space extension** with a wake-up delay and a
  toggle penalty that prevents flicker policies.
- **Masked Categorical PPO** and masked-target Double DQN over a
  `K·N + 1` (baseline) or `K·N + N + 1` (sleep-aware) discrete action
  space.
- **Hierarchical "Agentic" RL**: a REINFORCE supervisor delegating to
  power- and SLA-specialised DQN sub-agents.
- **Offline CMDP**: Fitted-Q-Iteration on two Q-heads with CQL
  pessimism on the cost head and dual ascent on `log λ`.
- **Strict generalisation protocol**: 50 train seeds + 50 held-out test
  seeds (offset by 10⁶), fleet seed fixed across runs for fair
  comparison.
- **Workloads**: synthetic Poisson, and `google_v2_sampled` (Google v2
  demand/duration marginals + synthetic Poisson timing).

## Quick start

```bash
# 1. Install
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. (optional) tune defaults
cp .env.example .env

# 3. Backend
uvicorn backend.main:app --reload \
  --reload-dir backend --reload-dir agents \
  --reload-dir environment --reload-dir training \
  --reload-include "*.py" \
  --reload-exclude "checkpoints/*" --reload-exclude "logs/*" \
  --reload-exclude "*.db*" --reload-exclude "*.pt"

# 4. Frontend
cd frontend && npm install && npm run dev
```

Open `http://localhost:5173`, pick an agent on the **Train** page,
launch a run, and watch live curves on the **Monitor** page.

### CLI alternative

```bash
curl -X POST http://localhost:8000/api/train \
  -H 'Content-Type: application/json' \
  -d '{
    "agent": "ppo",
    "n_servers": 50,
    "cluster_type": "heterogeneous",
    "episodes": 2000,
    "episode_length": 1000,
    "total_steps": 200000,
    "alpha": 1.0,
    "beta": 50.0,
    "seed": 0,
    "n_train_seeds": 50,
    "n_test_seeds": 50,
    "use_real_traces": false,
    "trace_family": "google_v2_sampled"
  }'
```

## Reproducing the headline numbers

1. Reset state: `rm -f experiments.db && rm -rf checkpoints/ logs/`
2. Run heuristics (RR, SJF, FFD) with `episodes=1, n_train_seeds=1,
   n_test_seeds=50` — these don't train, just eval on the test pool.
3. Run DQN (2,000 episodes), PPO (200,000 steps), Agentic (2,000
   episodes), each with `n_train_seeds=n_test_seeds=50`.
4. Flip `use_real_traces=true, trace_family=google_v2_sampled` and
   repeat to get the real-trace comparison.
5. Pull `/api/experiments` for the final 12-row comparison table.

Full plan is in `execution_plan_latest.md`. The Google v2 trace shards
are downloaded into `data/raw/google_v2/` via `gsutil` (see the
execution plan for the exact commands).

## Configuration

All hyperparameters live in `environment/config.py`, which loads from a
`.env` file at process start. Key knobs:

| Group   | Parameter                  | Default     |
|---------|----------------------------|-------------|
| Cluster | `N_SERVERS`                | 15          |
|         | `P_IDLE`, `P_MAX`          | 100, 300 W  |
|         | power exponent `POWER_ALPHA` | 1.4       |
| SLA     | `SLA_LATENCY_DEADLINE`     | 30 steps    |
| Reward  | `ALPHA` (power) / `BETA` (SLA) | 5.0 / 60.0 |
|         | `TOGGLE_PENALTY`           | 0.2         |
| Sleep   | `SLEEP_STANDBY_FACTOR`     | 0.05        |
|         | `SERVER_WAKEUP_DELAY`      | 1 step      |
| Seeds   | `FLEET_CLUSTER_SEED`       | 0 (fixed)   |
|         | `TEST_SEED_OFFSET`         | 1,000,000   |

The frontend hydrates its defaults from `/api/config/defaults` on
mount, so retuning the env doesn't require a frontend edit.

## Repository structure (detail)

- `environment/cluster_env.py` — Gymnasium-style MDP wrapping the
  fleet, queue, reward, and SLA bookkeeping.
- `environment/server.py` — three-state server lifecycle
  (active / waking / asleep) with utilisation-driven power model.
- `agents/heuristics/` — Round-Robin, SJF, FFD as masked policies.
- `agents/dqn_agent.py` — Double DQN with replay, target net, and
  feasibility masking on both action selection and bootstrap target.
- `agents/ppo_agent.py` — Actor-critic PPO with a masked Categorical
  head, GAE-λ, value clipping.
- `agents/agentic/supervisor.py` — REINFORCE gate over pre-trained
  power- and SLA-specialised sub-agents.
- `agents/offline/cmdp_agent.py` — Two-headed FQI (Q_r, Q_c) with CQL
  pessimism, Lagrangian shaping, log-λ dual ascent.
- `training/evaluator.py` — Held-out test-pool evaluation utilities.

## Project artefacts

- `Final_Dynamic_Resource_Allocation_RL_Presentation.pptx` — final
  presentation.
- `Midterm_Dynamic_Resource_Allocation_RL_Presentation.pdf` — midterm.
- `Project Proposal.pdf` — original proposal.
- `Final_Report.tex` — final write-up (NeurIPS 2024 format).
- `execution_plan_latest.md` — full sweep plan and per-agent training
  budgets.

## Headline findings (preview)

- **PPO** with action masking is the strongest learned agent on both
  synthetic and Google v2 sampled workloads; modestly outperforms FFD
  on out-of-sample SLA rate.
- **DQN** policy-collapses to a near-degenerate mode on the masked
  combinatorial action space (documented as a negative result).
- **Sleep/wake extension** with `(α,β,τ) = (5, 60, 0.2)` yields a
  material drop in mean cluster power under PPO; without the toggle
  penalty, the policy flickers.
- **Offline CMDP** fails to reduce SLA below the behaviour-policy
  floor — dataset feasibility floor + CQL inflation of `Q_c` + dual
  saturation compound.

See `Final_Report` for the full analysis.

## License

Course project. Released for academic use.
