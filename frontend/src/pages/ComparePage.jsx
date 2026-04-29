import { useMemo, useState } from "react";
import { useQueries, useQuery } from "@tanstack/react-query";

import { api } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import MetricBars, { colorForAgent } from "../components/Charts/MetricBars.jsx";
import MetricLine from "../components/Charts/MetricLine.jsx";
import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, CartesianGrid,
  ResponsiveContainer, Legend,
} from "recharts";

export default function ComparePage() {
  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
    refetchInterval: 30000,
  });

  const completed = experiments.filter((e) => e.status === "completed");

  // Group by agent type so the user can quickly toggle whole categories.
  const agents = useMemo(
    () => Array.from(new Set(completed.map((e) => e.agent))),
    [completed]
  );
  const [enabledAgents, setEnabledAgents] = useState(null);
  const active = enabledAgents ?? agents;

  const filtered = completed.filter((e) => active.includes(e.agent));

  // Sort by reward descending for nicer-looking bars (best on the left).
  const sorted = [...filtered].sort(
    (a, b) => (b.mean_reward_last10 ?? -Infinity) - (a.mean_reward_last10 ?? -Infinity)
  );

  // Decorate rows for charts.
  const rows = sorted.map((e) => ({
    label: `${e.agent}·${e.run_id.slice(-6)}`,
    agent: e.agent,
    run_id: e.run_id,
    mean_reward_last10: e.mean_reward_last10,
    mean_power: e.mean_power,
    sla_pct: e.sla_violation_rate != null ? e.sla_violation_rate * 100 : null,
    gap_pct: e.gap_pct,
    optimal_reward: e.optimal_reward,
  }));

  // Pareto: one point per run, axes = (mean_power, sla_pct). Group by agent
  // so the same agent shares a color across many runs.
  const paretoByAgent = useMemo(() => {
    const map = new Map();
    rows.forEach((r) => {
      if (r.mean_power == null || r.sla_pct == null) return;
      if (!map.has(r.agent)) map.set(r.agent, []);
      map.get(r.agent).push({ power: r.mean_power, sla: r.sla_pct, run: r.run_id });
    });
    return [...map.entries()].map(([agent, points]) => ({ agent, points }));
  }, [rows]);

  // Fetch full episode data for the top 8 runs to overlay learning curves.
  const topRuns = sorted.slice(0, 8);
  const detailQs = useQueries({
    queries: topRuns.map((r) => ({
      queryKey: ["details", r.run_id],
      queryFn: () => api.getExperiment(r.run_id),
    })),
  });
  const details = detailQs.filter((q) => q.data).map((q) => q.data);

  const overlaidRewardSeries = useMemo(
    () =>
      details.map((d) => ({
        name: `${d.summary.agent}·${d.summary.run_id.slice(-6)}`,
        data: (d.episodes || []).map((e) => ({ episode: e.episode, value: e.reward })),
      })),
    [details]
  );

  return (
    <div>
      <Card
        title="Cross-run comparison"
        sub="Aggregate metrics across every completed run. Toggle agents to filter."
      >
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
          {agents.map((a) => {
            const on = active.includes(a);
            return (
              <button
                key={a}
                className="btn ghost"
                onClick={() =>
                  setEnabledAgents(
                    on ? active.filter((x) => x !== a) : [...active, a]
                  )
                }
                style={{
                  borderColor: on ? colorForAgent(a) : "var(--border)",
                  color: on ? colorForAgent(a) : "var(--text-muted)",
                  fontWeight: on ? 600 : 400,
                  padding: "6px 12px",
                  fontSize: 12,
                }}
              >
                <span
                  style={{
                    display: "inline-block",
                    width: 8,
                    height: 8,
                    borderRadius: 4,
                    background: on ? colorForAgent(a) : "transparent",
                    border: `1px solid ${colorForAgent(a)}`,
                    marginRight: 6,
                    verticalAlign: "middle",
                  }}
                />
                {a}
              </button>
            );
          })}
        </div>
        {rows.length === 0 && (
          <div className="empty">No completed runs match the selected agents.</div>
        )}
      </Card>

      {rows.length > 0 && (
        <>
          <div className="grid cols-3">
            <Card title="Best reward (last-10 mean)" sub="Higher is better; closer to 0 is closer to optimal">
              <MetricBars
                data={rows}
                metric="mean_reward_last10"
                yLabel="Reward"
                formatY={(v) => v.toFixed(0)}
              />
            </Card>
            <Card title="Mean power (eval)" sub="Lower is better">
              <MetricBars
                data={rows.filter((r) => r.mean_power != null)}
                metric="mean_power"
                yLabel="Power (W·steps)"
                formatY={(v) => v.toFixed(0)}
              />
            </Card>
            <Card title="SLA violation rate (eval)" sub="Lower is better">
              <MetricBars
                data={rows.filter((r) => r.sla_pct != null)}
                metric="sla_pct"
                yLabel="%"
                formatY={(v) => v.toFixed(1) + "%"}
              />
            </Card>
          </div>

          <Card
            title="Gap from optimal"
            sub="(optimal − last10 reward) / |optimal| × 100. 0 = at the theoretical ceiling. Negative values = above the ceiling (impossible — usually a sign of too few episodes)."
          >
            <MetricBars
              data={rows.filter((r) => r.gap_pct != null)}
              metric="gap_pct"
              yLabel="%"
              formatY={(v) => v.toFixed(1) + "%"}
            />
          </Card>

          <Card
            title="Pareto frontier · power vs. SLA across all runs"
            sub="Each point = one completed run. Down-and-left wins. Color = agent type."
          >
            <ResponsiveContainer width="100%" height={340}>
              <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
                <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="power"
                  name="mean power"
                  tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }}
                  label={{ value: "Mean power (W·steps)", position: "insideBottom", offset: -4, style: { fontSize: 11, fill: "#6b7280" } }}
                />
                <YAxis
                  type="number"
                  dataKey="sla"
                  name="SLA %"
                  tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }}
                  label={{ value: "SLA rate (%)", angle: -90, position: "insideLeft", style: { fontSize: 11, fill: "#6b7280" } }}
                />
                <ZAxis range={[80, 80]} />
                <Tooltip
                  contentStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12 }}
                  cursor={{ strokeDasharray: "3 3" }}
                  formatter={(value, name) => {
                    if (name === "mean power") return value.toFixed(0);
                    if (name === "SLA %") return value.toFixed(2) + "%";
                    return value;
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                {paretoByAgent.map((g) => (
                  <Scatter key={g.agent} name={g.agent} data={g.points} fill={colorForAgent(g.agent)} />
                ))}
              </ScatterChart>
            </ResponsiveContainer>
          </Card>

          <Card
            title={`Reward curves overlay (top ${details.length} by reward)`}
            sub="Auto-loaded — fetches full episode history for the highest-scoring runs"
          >
            <MetricLine series={overlaidRewardSeries} yLabel="Reward" height={340} />
          </Card>
        </>
      )}
    </div>
  );
}
