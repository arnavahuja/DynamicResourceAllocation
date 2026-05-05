import { useEffect, useMemo, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import Button from "../components/UI/Button.jsx";

const ONLINE_AGENTS = ["dqn", "ppo", "agentic", "round_robin", "sjf", "ffd"];

// Default base configs — kept minimal, mirror the train pages.
const DEFAULT_ONLINE = {
  agent: "ppo",
  n_servers: 15,
  episodes: 500,
  episode_length: 1000,
  total_steps: 50000,
  alpha: 5.0,
  beta: 60.0,
  seed: 0,
  eval_episodes: 0,
  use_real_traces: false,
  trace_family: "google_v2_sampled",
  cluster_type: "heterogeneous",
  n_train_seeds: 50,
  n_test_seeds: 50,
};

const DEFAULT_OFFLINE = {
  n_servers: 15,
  episode_length: 1000,
  cluster_type: "heterogeneous",
  alpha: 5.0,
  beta: 60.0,
  seed: 0,
  dataset_path: "data/processed/offline.parquet",
  auto_generate_dataset: true,
  dataset_episodes: 200,
  dataset_random_eps: 0.05,
  iterations: 20000,
  batch_size: 256,
  sla_budget: 0.12,
  cql_alpha: 3.0,
  reward_kind: "neg_power",
  target_update_freq: 500,
  dual_update_freq: 200,
  n_test_seeds: 50,
  behavior_weight_rr: 0.0,
  behavior_weight_sjf: 0.7,
  behavior_weight_ffd: 0.3,
};

export default function SweepPage() {
  const qc = useQueryClient();
  const nav = useNavigate();

  const [mode, setMode] = useState("online");
  const [sweepParam, setSweepParam] = useState("alpha");
  const [valuesText, setValuesText] = useState("0.5, 1.0, 1.5, 2.0");
  const [baseOnline, setBaseOnline] = useState({ ...DEFAULT_ONLINE });
  const [baseOffline, setBaseOffline] = useState({ ...DEFAULT_OFFLINE });

  const { data: sweepable } = useQuery({
    queryKey: ["sweepable"],
    queryFn: api.sweepable,
  });
  const paramOptions = sweepable ? sweepable[mode] || [] : [];

  // Keep `sweepParam` valid when the mode changes.
  useEffect(() => {
    if (paramOptions.length && !paramOptions.includes(sweepParam)) {
      setSweepParam(paramOptions[0]);
    }
  }, [mode, paramOptions]);

  const launch = useMutation({
    mutationFn: api.startSweep,
    onSuccess: (resp) => {
      qc.invalidateQueries({ queryKey: ["sweeps"] });
      qc.invalidateQueries({ queryKey: ["experiments"] });
      nav(`/sweep/results?sweep=${resp.sweep_id}`);
    },
  });

  const parsedValues = useMemo(() => {
    return valuesText
      .split(/[,\s]+/)
      .map((s) => s.trim())
      .filter(Boolean)
      .map(Number)
      .filter((v) => !Number.isNaN(v));
  }, [valuesText]);

  const handleLaunch = () => {
    if (parsedValues.length === 0) return;
    const body = {
      mode,
      sweep_param: sweepParam,
      sweep_values: parsedValues,
    };
    if (mode === "online") body.base_online = baseOnline;
    else body.base_offline = baseOffline;
    launch.mutate(body);
  };

  return (
    <div>
      <Card
        title="Parameter Sweep"
        sub="Fire N runs in one click, varying a single parameter while everything else stays fixed. Works for both online RL and offline CMDP."
      >
        <div className="grid cols-3">
          <div>
            <label>Mode</label>
            <div className="seg">
              <button
                className={mode === "online" ? "active" : ""}
                onClick={() => setMode("online")}
              >
                Online RL
              </button>
              <button
                className={mode === "offline" ? "active" : ""}
                onClick={() => setMode("offline")}
              >
                Offline CMDP
              </button>
            </div>
          </div>
          <div>
            <label>Parameter to sweep</label>
            <select value={sweepParam} onChange={(e) => setSweepParam(e.target.value)}>
              {paramOptions.map((p) => (
                <option key={p} value={p}>{p}</option>
              ))}
            </select>
          </div>
          <div>
            <label>Values (comma- or space-separated)</label>
            <input
              type="text"
              value={valuesText}
              onChange={(e) => setValuesText(e.target.value)}
              style={{ fontFamily: "monospace", fontSize: 12 }}
            />
            <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
              Parsed: <span className="mono">[{parsedValues.join(", ")}]</span>{" "}
              ({parsedValues.length} run{parsedValues.length === 1 ? "" : "s"})
            </div>
          </div>
        </div>

        {mode === "online" ? (
          <BaseOnlineForm base={baseOnline} setBase={setBaseOnline} />
        ) : (
          <BaseOfflineForm base={baseOffline} setBase={setBaseOffline} />
        )}

        <div style={{ marginTop: 20, display: "flex", gap: 12, alignItems: "center" }}>
          <Button onClick={handleLaunch} disabled={launch.isPending || parsedValues.length === 0}>
            {launch.isPending ? "Launching…" : `Launch ${parsedValues.length} run${parsedValues.length === 1 ? "" : "s"}`}
          </Button>
          <Button variant="ghost" onClick={() => nav("/sweep/results")}>
            View past sweep results →
          </Button>
          {launch.isError && <span style={{ color: "var(--danger)" }}>Failed: {launch.error.message}</span>}
        </div>
      </Card>
    </div>
  );
}

// ── Base configuration sub-forms ─────────────────────────────────────────

function BaseOnlineForm({ base, setBase }) {
  const set = (k, v) => setBase((p) => ({ ...p, [k]: v }));
  return (
    <div style={{ marginTop: 16, padding: "10px 12px", border: "1px solid #e5e7eb", borderRadius: 6 }}>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
        Base configuration (the swept parameter is overridden per run)
      </div>
      <div className="grid cols-3">
        <div>
          <label>Agent</label>
          <select value={base.agent} onChange={(e) => set("agent", e.target.value)}>
            {ONLINE_AGENTS.map((a) => <option key={a} value={a}>{a}</option>)}
          </select>
        </div>
        <div>
          <label>Servers (N)</label>
          <input type="number" value={base.n_servers} onChange={(e) => set("n_servers", +e.target.value)} />
        </div>
        <div>
          <label>Episodes</label>
          <input type="number" value={base.episodes} onChange={(e) => set("episodes", +e.target.value)} />
        </div>
      </div>
      <div className="grid cols-3" style={{ marginTop: 12 }}>
        <div><label>α</label><input type="number" step="0.01" value={base.alpha} onChange={(e) => set("alpha", +e.target.value)} /></div>
        <div><label>β</label><input type="number" step="0.1" value={base.beta} onChange={(e) => set("beta", +e.target.value)} /></div>
        <div><label>Episode length</label><input type="number" value={base.episode_length} onChange={(e) => set("episode_length", +e.target.value)} /></div>
      </div>
      <div className="grid cols-3" style={{ marginTop: 12 }}>
        <div><label>Train seeds</label><input type="number" value={base.n_train_seeds} onChange={(e) => set("n_train_seeds", +e.target.value)} /></div>
        <div><label>Test seeds</label><input type="number" value={base.n_test_seeds} onChange={(e) => set("n_test_seeds", +e.target.value)} /></div>
        <div><label>Total steps (PPO)</label><input type="number" value={base.total_steps} onChange={(e) => set("total_steps", +e.target.value)} /></div>
      </div>
      <div style={{ marginTop: 12, display: "flex", alignItems: "center", gap: 16, flexWrap: "wrap" }}>
        <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 0 }}>
          <input type="checkbox" checked={base.use_real_traces} onChange={(e) => set("use_real_traces", e.target.checked)} />
          Use real cluster traces
        </label>
        {base.use_real_traces && (
          <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 0 }}>
            Trace:
            <select value={base.trace_family} onChange={(e) => set("trace_family", e.target.value)}>
              <option value="google_v2">Google v2 — pure replay</option>
              <option value="google_v2_sampled">Google v2 (sampled + Poisson)</option>
            </select>
          </label>
        )}
        <label style={{ display: "flex", alignItems: "center", gap: 8, marginBottom: 0 }}>
          Cluster:
          <select value={base.cluster_type} onChange={(e) => set("cluster_type", e.target.value)}>
            <option value="homogeneous">homogeneous</option>
            <option value="heterogeneous">heterogeneous</option>
          </select>
        </label>
      </div>
    </div>
  );
}

function BaseOfflineForm({ base, setBase }) {
  const set = (k, v) => setBase((p) => ({ ...p, [k]: v }));
  return (
    <div style={{ marginTop: 16, padding: "10px 12px", border: "1px solid #e5e7eb", borderRadius: 6 }}>
      <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 8 }}>
        Base configuration (the swept parameter is overridden per run)
      </div>
      <div className="grid cols-3">
        <div>
          <label>Dataset path</label>
          <input
            type="text"
            value={base.dataset_path}
            onChange={(e) => set("dataset_path", e.target.value)}
            style={{ fontFamily: "monospace", fontSize: 11 }}
          />
        </div>
        <div>
          <label>Servers (N)</label>
          <input type="number" value={base.n_servers} onChange={(e) => set("n_servers", +e.target.value)} />
        </div>
        <div>
          <label>Test seeds</label>
          <input type="number" value={base.n_test_seeds} onChange={(e) => set("n_test_seeds", +e.target.value)} />
        </div>
      </div>
      <div className="grid cols-3" style={{ marginTop: 12 }}>
        <div><label>FQI iterations</label><input type="number" value={base.iterations} onChange={(e) => set("iterations", +e.target.value)} /></div>
        <div><label>Batch size</label><input type="number" value={base.batch_size} onChange={(e) => set("batch_size", +e.target.value)} /></div>
        <div><label>SLA budget ε</label><input type="number" step="0.005" value={base.sla_budget} onChange={(e) => set("sla_budget", +e.target.value)} /></div>
      </div>
      <div className="grid cols-3" style={{ marginTop: 12 }}>
        <div><label>CQL α</label><input type="number" step="0.1" value={base.cql_alpha} onChange={(e) => set("cql_alpha", +e.target.value)} /></div>
        <div><label>Random injection</label><input type="number" step="0.05" value={base.dataset_random_eps} onChange={(e) => set("dataset_random_eps", +e.target.value)} /></div>
        <div><label>Behavior episodes</label><input type="number" value={base.dataset_episodes} onChange={(e) => set("dataset_episodes", +e.target.value)} /></div>
      </div>
      <div className="grid cols-3" style={{ marginTop: 12 }}>
        <div><label>w(RR)</label><input type="number" step="0.05" value={base.behavior_weight_rr} onChange={(e) => set("behavior_weight_rr", +e.target.value)} /></div>
        <div><label>w(SJF)</label><input type="number" step="0.05" value={base.behavior_weight_sjf} onChange={(e) => set("behavior_weight_sjf", +e.target.value)} /></div>
        <div><label>w(FFD)</label><input type="number" step="0.05" value={base.behavior_weight_ffd} onChange={(e) => set("behavior_weight_ffd", +e.target.value)} /></div>
      </div>
      <div className="grid cols-3" style={{ marginTop: 12 }}>
        <div><label>α (eval)</label><input type="number" step="0.05" value={base.alpha} onChange={(e) => set("alpha", +e.target.value)} /></div>
        <div><label>β (eval)</label><input type="number" step="0.1" value={base.beta} onChange={(e) => set("beta", +e.target.value)} /></div>
        <div>
          <label>Cluster type</label>
          <select value={base.cluster_type} onChange={(e) => set("cluster_type", e.target.value)}>
            <option value="homogeneous">homogeneous</option>
            <option value="heterogeneous">heterogeneous</option>
          </select>
        </div>
      </div>
    </div>
  );
}
