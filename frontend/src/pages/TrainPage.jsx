import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import Button from "../components/UI/Button.jsx";
import Badge from "../components/UI/Badge.jsx";

const AGENTS = [
  { id: "dqn", label: "DQN" },
  { id: "ppo", label: "PPO" },
  { id: "agentic", label: "Agentic RL" },
  { id: "round_robin", label: "Round Robin" },
  { id: "sjf", label: "SJF" },
  { id: "ffd", label: "FFD" },
];

// Module-level cache: persists while the SPA is mounted (i.e. while you
// navigate between tabs) but is wiped on full page reload, since the JS
// module is re-evaluated. The numeric/env-tunable fields (nServers, epLen,
// alpha, beta) are hydrated from /api/config/defaults so they always match
// environment/config.py. The fallbacks below are only used until that fetch
// resolves on first mount.
const FORM_DEFAULTS = {
  agent: "dqn",
  nServers: 10,
  episodes: 200,
  epLen: 1000,
  alpha: 1.0,
  beta: 50.0,
  seed: 0,
  evalEps: 10,
  useTraces: false,
  traceFamily: "alibaba",
  clusterType: "homogeneous",
  nTrainSeeds: 1,
  nTestSeeds: 0,
};
const formCache = { ...FORM_DEFAULTS };
let _defaultsHydrated = false;

export default function TrainPage() {
  const nav = useNavigate();
  const qc = useQueryClient();

  const [agent, setAgent] = useState(() => formCache.agent);
  const [nServers, setNServers] = useState(() => formCache.nServers);
  const [episodes, setEpisodes] = useState(() => formCache.episodes);
  const [epLen, setEpLen] = useState(() => formCache.epLen);
  const [alpha, setAlpha] = useState(() => formCache.alpha);
  const [beta, setBeta] = useState(() => formCache.beta);
  const [seed, setSeed] = useState(() => formCache.seed);
  const [evalEps, setEvalEps] = useState(() => formCache.evalEps);
  const [useTraces, setUseTraces] = useState(() => formCache.useTraces);
  const [traceFamily, setTraceFamily] = useState(() => formCache.traceFamily);
  const [clusterType, setClusterType] = useState(() => formCache.clusterType);
  const [nTrainSeeds, setNTrainSeeds] = useState(() => formCache.nTrainSeeds);
  const [nTestSeeds, setNTestSeeds] = useState(() => formCache.nTestSeeds);

  // Hydrate from backend on first mount — keeps form defaults in lockstep
  // with environment/config.py so retuning the reward weights or default
  // cluster size doesn't require a frontend edit.
  useEffect(() => {
    if (_defaultsHydrated) return;
    api.configDefaults()
      .then((d) => {
        _defaultsHydrated = true;
        if (d.n_servers != null) { formCache.nServers = d.n_servers; setNServers(d.n_servers); }
        if (d.episode_length != null) { formCache.epLen = d.episode_length; setEpLen(d.episode_length); }
        if (d.alpha != null) { formCache.alpha = d.alpha; setAlpha(d.alpha); }
        if (d.beta != null) { formCache.beta = d.beta; setBeta(d.beta); }
      })
      .catch(() => { /* fall back to hardcoded FORM_DEFAULTS */ });
  }, []);

  // Mirror every state change back to the module-level cache so the next
  // mount of TrainPage (after a tab switch) reads the up-to-date values.
  useEffect(() => {
    Object.assign(formCache, {
      agent, nServers, episodes, epLen, alpha, beta, seed, evalEps, useTraces, traceFamily, clusterType,
      nTrainSeeds, nTestSeeds,
    });
  }, [agent, nServers, episodes, epLen, alpha, beta, seed, evalEps, useTraces, traceFamily, clusterType, nTrainSeeds, nTestSeeds]);

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
    refetchInterval: 5000,
  });

  const launch = useMutation({
    mutationFn: api.startTraining,
    onSuccess: ({ run_id }) => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      nav(`/monitor?run=${run_id}`);
    },
  });

  return (
    <div>
      <Card title="Configure Training Run" sub="Pick an agent and tweak the cluster — then launch.">
        <div className="grid cols-2">
          <div>
            <label>Agent</label>
            <div className="seg">
              {AGENTS.map((a) => (
                <button
                  key={a.id}
                  className={agent === a.id ? "active" : ""}
                  onClick={() => setAgent(a.id)}
                >
                  {a.label}
                </button>
              ))}
            </div>
          </div>
          <div>
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
          </div>
        </div>

        <div className="grid cols-3" style={{ marginTop: 16 }}>
          <div>
            <label>Servers (N): {nServers}</label>
            <input type="range" min={2} max={100} value={nServers} onChange={(e) => setNServers(+e.target.value)} />
          </div>
          <div>
            <label>Episodes</label>
            <input type="number" value={episodes} onChange={(e) => setEpisodes(+e.target.value)} />
          </div>
          <div>
            <label>Episode length</label>
            <input type="number" value={epLen} onChange={(e) => setEpLen(+e.target.value)} />
          </div>
        </div>

        <div className="grid cols-3" style={{ marginTop: 16 }}>
          <div>
            <label>α (power weight): {alpha.toFixed(2)}</label>
            <input type="range" min={0} max={5} step={0.05} value={alpha} onChange={(e) => setAlpha(+e.target.value)} />
          </div>
          <div>
            <label>β (SLA weight): {beta.toFixed(1)}</label>
            <input type="range" min={0} max={200} step={1} value={beta} onChange={(e) => setBeta(+e.target.value)} />
          </div>
          <div>
            <label>Eval episodes</label>
            <input type="number" value={evalEps} onChange={(e) => setEvalEps(+e.target.value)} />
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          <label>Cluster type</label>
          <div className="seg">
            <button
              className={clusterType === "homogeneous" ? "active" : ""}
              onClick={() => setClusterType("homogeneous")}
            >
              Homogeneous
            </button>
            <button
              className={clusterType === "heterogeneous" ? "active" : ""}
              onClick={() => setClusterType("heterogeneous")}
            >
              Heterogeneous
            </button>
          </div>
          <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
            {clusterType === "homogeneous"
              ? "All servers identical (uses .env P_IDLE / P_MAX)."
              : "Mixed efficient / standard / power-hungry tiers. Same fleet across all agents at the same N for fair comparison."}
          </div>
        </div>

        <div className="grid cols-2" style={{ marginTop: 16 }}>
          <div>
            <label>Train workload seeds</label>
            <input
              type="number"
              min={1}
              value={nTrainSeeds}
              onChange={(e) => setNTrainSeeds(+e.target.value)}
            />
            <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
              How many distinct workloads to sample from during training.
              1 = single trajectory (overfit risk). 20-50 = ML-style training set.
            </div>
          </div>
          <div>
            <label>Test (held-out) seeds</label>
            <input
              type="number"
              min={0}
              value={nTestSeeds}
              onChange={(e) => setNTestSeeds(+e.target.value)}
            />
            <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
              Workloads the agent never trains on, used only for final eval.
              0 = falls back to legacy "eval episodes" on the train seed.
            </div>
          </div>
        </div>

        <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 0 }}>
            <input type="checkbox" checked={useTraces} onChange={(e) => setUseTraces(e.target.checked)} />
            Use real cluster traces
          </label>
          {useTraces && (
            <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 0 }}>
              Trace:
              <select value={traceFamily} onChange={(e) => setTraceFamily(e.target.value)}>
                <option value="alibaba">Alibaba 2018</option>
                <option value="google_v2">Google v2 (2011)</option>
                <option value="google_v3">Google v3 (2019)</option>
                <option value="google_v2_sampled">Google v2 (sampled + Poisson)</option>
              </select>
            </label>
          )}
        </div>

        <div style={{ marginTop: 20, display: "flex", gap: 12, alignItems: "center" }}>
          <Button
            onClick={() =>
              launch.mutate({
                agent,
                n_servers: nServers,
                episodes,
                episode_length: epLen,
                total_steps: episodes * epLen,
                alpha,
                beta,
                seed,
                eval_episodes: evalEps,
                use_real_traces: useTraces,
                trace_family: traceFamily,
                cluster_type: clusterType,
                n_train_seeds: nTrainSeeds,
                n_test_seeds: nTestSeeds,
              })
            }
            disabled={launch.isPending}
          >
            {launch.isPending ? "Launching…" : "Launch Training Run"}
          </Button>
          {launch.isError && <span style={{ color: "var(--danger)" }}>Failed: {launch.error.message}</span>}
        </div>
      </Card>

      <Card title="Experiments" sub="Auto-refreshes every 5s">
        {experiments.length === 0 ? (
          <div className="empty">No runs yet. Configure one above and launch.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Agent</th>
                <th>Cluster</th>
                <th>Status</th>
                <th>Servers</th>
                <th>Episodes</th>
                <th>Last-10 R</th>
                <th>Optimal R</th>
                <th>Gap %</th>
                <th>SLA rate</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.run_id} onClick={() => nav(`/results?run=${e.run_id}`)}>
                  <td className="mono" style={{ fontSize: 11 }}>{e.run_id}</td>
                  <td>{e.agent}</td>
                  <td className="mono" style={{ fontSize: 11 }}>{e.cluster_type || "—"}</td>
                  <td><Badge status={e.status} /></td>
                  <td className="mono">{e.n_servers}</td>
                  <td className="mono">{e.episodes}</td>
                  <td className="mono">{fmt(e.mean_reward_last10)}</td>
                  <td className="mono">{fmt(e.optimal_reward)}</td>
                  <td className="mono">{e.gap_pct != null ? e.gap_pct.toFixed(1) + "%" : "—"}</td>
                  <td className="mono">{e.sla_violation_rate != null ? (e.sla_violation_rate * 100).toFixed(1) + "%" : "—"}</td>
                  <td onClick={(ev) => ev.stopPropagation()}>
                    {e.status === "running" && (
                      <a href={`/monitor?run=${e.run_id}`}>monitor →</a>
                    )}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}

function fmt(v) {
  if (v == null) return "—";
  return v.toFixed(2);
}
