import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery, useQueries } from "@tanstack/react-query";

import { api } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import Button from "../components/UI/Button.jsx";
import Badge from "../components/UI/Badge.jsx";
import RewardCurve from "../components/Charts/RewardCurve.jsx";
import PowerSlaScatter from "../components/Charts/PowerSlaScatter.jsx";
import MetricLine from "../components/Charts/MetricLine.jsx";
import MetricBars from "../components/Charts/MetricBars.jsx";

export default function ResultsPage() {
  const [params] = useSearchParams();
  const initial = params.get("run");

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
    refetchInterval: 30000,
  });

  const [selected, setSelected] = useState(() => (initial ? [initial] : []));
  const toggle = (id) =>
    setSelected((prev) =>
      prev.includes(id) ? prev.filter((x) => x !== id) : [...prev, id]
    );

  const detailQs = useQueries({
    queries: selected.map((id) => ({
      queryKey: ["details", id],
      queryFn: () => api.getExperiment(id),
    })),
  });
  const details = detailQs
    .filter((q) => q.data)
    .map((q) => q.data);

  const rewardSeries = useMemo(
    () =>
      details.map((d) => ({
        name: `${d.summary.agent} · ${d.summary.run_id.slice(-8)}`,
        data: (d.episodes || []).map((e) => ({ episode: e.episode, reward: e.reward })),
      })),
    [details]
  );

  const paretoGroups = useMemo(
    () =>
      details.map((d) => ({
        name: `${d.summary.agent} · ${d.summary.run_id.slice(-8)}`,
        points: (d.episodes || []).map((e) => ({ power: e.power, sla: e.sla_violations })),
      })),
    [details]
  );

  const powerSeries = useMemo(
    () =>
      details.map((d) => ({
        name: `${d.summary.agent} · ${d.summary.run_id.slice(-8)}`,
        data: (d.episodes || []).map((e) => ({ episode: e.episode, value: e.power })),
      })),
    [details]
  );

  const slaSeries = useMemo(
    () =>
      details.map((d) => ({
        name: `${d.summary.agent} · ${d.summary.run_id.slice(-8)}`,
        data: (d.episodes || []).map((e) => ({ episode: e.episode, value: e.sla_violations })),
      })),
    [details]
  );

  function downloadCsv() {
    const rows = [["run_id", "agent", "episode", "reward", "power", "sla_violations", "steps"]];
    details.forEach((d) =>
      (d.episodes || []).forEach((e) =>
        rows.push([d.summary.run_id, d.summary.agent, e.episode, e.reward, e.power, e.sla_violations, e.steps])
      )
    );
    const csv = rows.map((r) => r.join(",")).join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "experiments.csv";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  return (
    <div>
      <Card
        title="Experiments"
        sub="Click rows to overlay them on the comparison charts below"
        action={
          <Button variant="ghost" onClick={downloadCsv} disabled={details.length === 0}>
            Export CSV
          </Button>
        }
      >
        {experiments.length === 0 ? (
          <div className="empty">No experiments yet.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th></th>
                <th>Run</th>
                <th>Agent</th>
                <th>Status</th>
                <th>Servers</th>
                <th>Episodes</th>
                <th>Last-10 R</th>
                <th>Optimal R</th>
                <th>Gap %</th>
                <th>Mean Power</th>
                <th>SLA rate</th>
              </tr>
            </thead>
            <tbody>
              {experiments.map((e) => (
                <tr key={e.run_id} onClick={() => toggle(e.run_id)}>
                  <td>
                    <input type="checkbox" checked={selected.includes(e.run_id)} onChange={() => toggle(e.run_id)} onClick={(ev) => ev.stopPropagation()} />
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>{e.run_id}</td>
                  <td>{e.agent}</td>
                  <td><Badge status={e.status} /></td>
                  <td className="mono">{e.n_servers}</td>
                  <td className="mono">{e.episodes}</td>
                  <td className="mono">{fmt(e.mean_reward_last10)}</td>
                  <td className="mono">{fmt(e.optimal_reward)}</td>
                  <td className="mono">{e.gap_pct != null ? e.gap_pct.toFixed(1) + "%" : "—"}</td>
                  <td className="mono">{fmt(e.mean_power, 0)}</td>
                  <td className="mono">{e.sla_violation_rate != null ? (e.sla_violation_rate * 100).toFixed(1) + "%" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {details.length > 0 && (
        <>
          <Card title="Comparison Table" sub="Eval-time metrics. 'Optimal' is the theoretical reward ceiling for the run's α/P_idle/P_max/episode_length (zero SLA + idle-only power).">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Agent</th>
                  <th>mean R (eval)</th>
                  <th>std R</th>
                  <th>optimal R</th>
                  <th>gap %</th>
                  <th>mean power</th>
                  <th>SLA rate</th>
                  <th>jobs/ep</th>
                </tr>
              </thead>
              <tbody>
                {details.map((d) => {
                  const ev = d.eval || {};
                  const opt = d.optimal_reward;
                  const gap =
                    opt != null && ev.mean_reward != null && opt !== 0
                      ? ((opt - ev.mean_reward) / Math.abs(opt)) * 100
                      : null;
                  return (
                    <tr key={d.summary.run_id}>
                      <td className="mono" style={{ fontSize: 11 }}>{d.summary.run_id}</td>
                      <td>{d.summary.agent}</td>
                      <td className="mono">{fmt(ev.mean_reward)}</td>
                      <td className="mono">{fmt(ev.std_reward)}</td>
                      <td className="mono">{fmt(opt)}</td>
                      <td className="mono">{gap != null ? gap.toFixed(1) + "%" : "—"}</td>
                      <td className="mono">{fmt(ev.mean_power, 0)}</td>
                      <td className="mono">{ev.sla_violation_rate != null ? (ev.sla_violation_rate * 100).toFixed(2) + "%" : "—"}</td>
                      <td className="mono">{fmt(ev.mean_jobs_completed, 1)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>

          <Card title="Training Curves" sub={(() => {
              const opts = details.map((d) => d.optimal_reward).filter((v) => v != null);
              if (!opts.length) return undefined;
              const allEqual = opts.every((v) => Math.abs(v - opts[0]) < 1e-6);
              return allEqual
                ? `Dashed line = optimal reward ceiling (${opts[0].toFixed(2)})`
                : `Dashed line = optimal of best-case run (${Math.max(...opts).toFixed(2)}); selected runs have differing α/episode_length so other ceilings differ.`;
            })()}>
            <RewardCurve
              series={rewardSeries}
              height={320}
              optimalReward={(() => {
                const opts = details.map((d) => d.optimal_reward).filter((v) => v != null);
                return opts.length ? Math.max(...opts) : null;
              })()}
            />
          </Card>

          <div className="grid cols-2">
            <Card title="Power per episode" sub="One line per selected run">
              <MetricLine series={powerSeries} yLabel="Power (W·steps)" />
            </Card>
            <Card title="SLA violations per episode" sub="One line per selected run">
              <MetricLine series={slaSeries} yLabel="Violations" />
            </Card>
          </div>

          <Card title="Power vs. SLA (per episode)" sub="Each point = one training episode">
            <PowerSlaScatter groups={paretoGroups} height={320} />
          </Card>
        </>
      )}
    </div>
  );
}

function fmt(v, digits = 2) {
  if (v == null) return "—";
  return Number(v).toFixed(digits);
}
