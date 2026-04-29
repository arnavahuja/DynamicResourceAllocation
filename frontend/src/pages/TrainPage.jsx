import { useState } from "react";
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

export default function TrainPage() {
  const nav = useNavigate();
  const qc = useQueryClient();

  const [agent, setAgent] = useState("dqn");
  const [nServers, setNServers] = useState(10);
  const [episodes, setEpisodes] = useState(200);
  const [epLen, setEpLen] = useState(500);
  const [alpha, setAlpha] = useState(1.0);
  const [beta, setBeta] = useState(50.0);
  const [seed, setSeed] = useState(0);
  const [evalEps, setEvalEps] = useState(10);
  const [useTraces, setUseTraces] = useState(false);

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

        <div style={{ marginTop: 16, display: "flex", alignItems: "center", gap: 12 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 0 }}>
            <input type="checkbox" checked={useTraces} onChange={(e) => setUseTraces(e.target.checked)} />
            Use real cluster traces
          </label>
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
                <th>Status</th>
                <th>Servers</th>
                <th>Episodes</th>
                <th>Last-10 R</th>
                <th>SLA rate</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.run_id} onClick={() => nav(`/results?run=${e.run_id}`)}>
                  <td className="mono" style={{ fontSize: 11 }}>{e.run_id}</td>
                  <td>{e.agent}</td>
                  <td><Badge status={e.status} /></td>
                  <td className="mono">{e.n_servers}</td>
                  <td className="mono">{e.episodes}</td>
                  <td className="mono">{fmt(e.mean_reward_last10)}</td>
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
