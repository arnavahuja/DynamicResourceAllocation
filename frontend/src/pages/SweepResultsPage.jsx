import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import Button from "../components/UI/Button.jsx";
import Badge from "../components/UI/Badge.jsx";
import MetricLine from "../components/Charts/MetricLine.jsx";

const ONLINE_METRICS = [
  { id: "mean_reward_eval", label: "Eval mean reward" },
  { id: "mean_reward_last10", label: "Train mean reward (last10)" },
  { id: "mean_power", label: "Mean power" },
  { id: "sla_violation_rate", label: "SLA rate", scale: 100 },
];
const OFFLINE_METRICS = [
  { id: "mean_reward_eval", label: "Eval mean reward" },
  { id: "mean_power", label: "Mean power" },
  { id: "sla_violation_rate", label: "SLA rate", scale: 100 },
  { id: "lambda_final", label: "λ final" },
  { id: "constraint_slack", label: "Constraint slack (ε−Q_c)" },
];

export default function SweepResultsPage() {
  const [params, setParams] = useSearchParams();
  const activeSweep = params.get("sweep") || "";
  const qc = useQueryClient();

  const [chartMetric, setChartMetric] = useState("mean_reward_eval");
  const [selectedRunIds, setSelectedRunIds] = useState([]);

  const deleteMutation = useMutation({
    mutationFn: async (ids) => {
      await Promise.all(ids.map((id) => api.deleteExperiment(id)));
      return ids;
    },
    onSuccess: (ids, _vars, _ctx) => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      qc.invalidateQueries({ queryKey: ["sweeps"] });
      qc.invalidateQueries({ queryKey: ["sweep", activeSweep] });
      setSelectedRunIds((prev) => prev.filter((x) => !ids.includes(x)));
    },
  });

  const handleDeleteSelected = () => {
    if (selectedRunIds.length === 0) return;
    const msg = `Delete ${selectedRunIds.length} run${selectedRunIds.length === 1 ? "" : "s"} from the database (and their checkpoints + logs)? This cannot be undone.`;
    if (window.confirm(msg)) deleteMutation.mutate(selectedRunIds);
  };

  const handleDeleteWholeSweep = (sweepRuns) => {
    if (!sweepRuns?.length) return;
    const msg = `Delete the entire sweep (${sweepRuns.length} runs) from the database (and their checkpoints + logs)? This cannot be undone.`;
    if (window.confirm(msg)) {
      deleteMutation.mutate(sweepRuns.map((r) => r.run_id), {
        onSuccess: () => {
          qc.invalidateQueries({ queryKey: ["experiments"] });
          qc.invalidateQueries({ queryKey: ["sweeps"] });
          setParams({});
        },
      });
    }
  };

  const { data: allSweeps = [] } = useQuery({
    queryKey: ["sweeps"],
    queryFn: api.listSweeps,
    refetchInterval: 5000,
  });

  const { data: sweepData } = useQuery({
    queryKey: ["sweep", activeSweep],
    queryFn: () => api.getSweep(activeSweep),
    enabled: !!activeSweep,
    refetchInterval: 3000,
  });

  const sweepRuns = sweepData?.runs || [];
  const sweepMode = sweepRuns.length && sweepRuns[0].agent === "cmdp"
    ? "offline" : "online";
  const activeMetricList = sweepMode === "offline" ? OFFLINE_METRICS : ONLINE_METRICS;
  const activeMetric = activeMetricList.find((m) => m.id === chartMetric)
    || activeMetricList[0];
  const chartScale = activeMetric?.scale || 1;

  const chartSeries = useMemo(() => {
    if (!sweepRuns.length) return [];
    const data = sweepRuns
      .filter((r) => r.status === "completed" && r[activeMetric.id] != null)
      .map((r) => ({
        episode: r.sweep_value,
        value: r[activeMetric.id] * chartScale,
      }));
    return [{ name: activeMetric.label, color: "#10B981", data }];
  }, [sweepRuns, activeMetric, chartScale]);

  const toggleRun = (id) =>
    setSelectedRunIds((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  const detailQs = useQueries({
    queries: selectedRunIds.map((id) => ({
      queryKey: ["details", id],
      queryFn: () => api.getExperiment(id),
    })),
  });
  const details = detailQs.filter((q) => q.data).map((q) => q.data);

  // Per-run training curves (only meaningful for online runs — CMDP has
  // no per-episode events during training, only eval episodes).
  const onlineDetails = details.filter((d) => d.summary.agent !== "cmdp");

  const rewardSeries = useMemo(
    () =>
      onlineDetails.map((d) => ({
        name: `${d.summary.agent}·${d.summary.run_id.slice(-6)}`,
        data: (d.episodes || []).map((e) => ({ episode: e.episode, value: e.reward })),
      })),
    [onlineDetails]
  );
  const powerSeries = useMemo(
    () =>
      onlineDetails.map((d) => ({
        name: `${d.summary.agent}·${d.summary.run_id.slice(-6)}`,
        data: (d.episodes || []).map((e) => ({ episode: e.episode, value: e.power })),
      })),
    [onlineDetails]
  );
  const slaSeries = useMemo(
    () =>
      onlineDetails.map((d) => ({
        name: `${d.summary.agent}·${d.summary.run_id.slice(-6)}`,
        data: (d.episodes || []).map((e) => ({ episode: e.episode, value: e.sla_violations })),
      })),
    [onlineDetails]
  );

  return (
    <div>
      <Card
        title="Sweeps"
        sub="Click a sweep to load its results below."
      >
        {allSweeps.length === 0 ? (
          <div className="empty">No sweeps yet. Launch one from the Sweep page.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Sweep</th>
                <th>Mode</th>
                <th>Param</th>
                <th>Runs</th>
                <th>Completed</th>
                <th>Created</th>
              </tr>
            </thead>
            <tbody>
              {allSweeps.map((s) => (
                <tr
                  key={s.sweep_id}
                  onClick={() => {
                    setParams({ sweep: s.sweep_id });
                    setSelectedRunIds([]);
                  }}
                  style={{
                    background: s.sweep_id === activeSweep
                      ? "rgba(16,185,129,0.08)"
                      : undefined,
                  }}
                >
                  <td className="mono" style={{ fontSize: 11 }}>{s.sweep_id}</td>
                  <td>{s.mode}</td>
                  <td className="mono">{s.sweep_param}</td>
                  <td className="mono">{s.n_runs}</td>
                  <td className="mono">{s.n_completed}/{s.n_runs}</td>
                  <td className="mono" style={{ fontSize: 11 }}>{s.created_at?.slice(0, 19)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {activeSweep && sweepData && (
        <>
          <Card
            title={`Sweep results — ${sweepData.sweep_id}`}
            sub={sweepRuns.length ? `Sweeping over ${sweepRuns[0].sweep_param} · ${sweepRuns.length} run${sweepRuns.length === 1 ? "" : "s"} · click rows to overlay training curves below` : ""}
            action={
              <div style={{ display: "flex", gap: 8 }}>
                <Button
                  variant="danger"
                  onClick={handleDeleteSelected}
                  disabled={selectedRunIds.length === 0 || deleteMutation.isPending}
                >
                  {deleteMutation.isPending
                    ? "Deleting…"
                    : `Delete ${selectedRunIds.length || ""} selected`}
                </Button>
                <Button
                  variant="ghost"
                  onClick={() => handleDeleteWholeSweep(sweepRuns)}
                  disabled={!sweepRuns.length || deleteMutation.isPending}
                >
                  Delete whole sweep
                </Button>
              </div>
            }
          >
            {sweepRuns.length === 0 ? (
              <div className="empty">No runs in this sweep yet.</div>
            ) : (
              <table>
                <thead>
                  <tr>
                    <th></th>
                    <th>Run</th>
                    <th>Agent</th>
                    <th>Status</th>
                    <th>{sweepRuns[0].sweep_param}</th>
                    <th>Eval mean R</th>
                    <th>Mean Power</th>
                    <th>SLA rate</th>
                    {sweepMode === "offline" && <th>λ final</th>}
                    {sweepMode === "offline" && <th>slack</th>}
                  </tr>
                </thead>
                <tbody>
                  {sweepRuns.map((r) => (
                    <tr key={r.run_id} onClick={() => toggleRun(r.run_id)}>
                      <td>
                        <input
                          type="checkbox"
                          checked={selectedRunIds.includes(r.run_id)}
                          onChange={() => toggleRun(r.run_id)}
                          onClick={(ev) => ev.stopPropagation()}
                        />
                      </td>
                      <td className="mono" style={{ fontSize: 11 }}>{r.run_id}</td>
                      <td>{r.agent}</td>
                      <td><Badge status={r.status} /></td>
                      <td className="mono">{fmt(r.sweep_value, 4)}</td>
                      <td className="mono">{fmt(r.mean_reward_eval)}</td>
                      <td className="mono">{r.mean_power != null ? r.mean_power.toFixed(0) : "—"}</td>
                      <td className="mono">{r.sla_violation_rate != null ? (r.sla_violation_rate * 100).toFixed(2) + "%" : "—"}</td>
                      {sweepMode === "offline" && <td className="mono">{fmt(r.lambda_final, 3)}</td>}
                      {sweepMode === "offline" && <td className="mono">{fmt(r.constraint_slack, 4)}</td>}
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </Card>

          <Card
            title="Sweep chart"
            sub={`X = ${sweepRuns[0]?.sweep_param || "param"}, Y = chosen metric. Only completed runs plotted.`}
            action={
              <select
                value={chartMetric}
                onChange={(e) => setChartMetric(e.target.value)}
              >
                {activeMetricList.map((m) => (
                  <option key={m.id} value={m.id}>{m.label}</option>
                ))}
              </select>
            }
          >
            <MetricLine
              series={chartSeries}
              yLabel={activeMetric.label}
              xKey="episode"
              yKey="value"
            />
          </Card>

          {onlineDetails.length > 0 && (
            <>
              <Card title="Reward per episode" sub="Selected runs overlaid">
                <MetricLine series={rewardSeries} yLabel="Reward" />
              </Card>
              <div className="grid cols-2">
                <Card title="Power per episode" sub="One line per selected run">
                  <MetricLine series={powerSeries} yLabel="Power (W·steps)" />
                </Card>
                <Card title="SLA violations per episode" sub="One line per selected run">
                  <MetricLine series={slaSeries} yLabel="Violations" />
                </Card>
              </div>
            </>
          )}

          {sweepMode === "offline" && selectedRunIds.length > 0 && (
            <Card title="CMDP runs" sub="For per-run FQI loss / λ / constraint-violation curves, open each run from the CMDP Results page.">
              <ul style={{ margin: 0, paddingLeft: 18, fontSize: 12 }}>
                {selectedRunIds.map((id) => (
                  <li key={id} className="mono" style={{ marginBottom: 4 }}>
                    <a href={`/cmdp/results?run=${id}`}>{id}</a>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function fmt(v, digits = 2) {
  if (v == null) return "—";
  return Number(v).toFixed(digits);
}
