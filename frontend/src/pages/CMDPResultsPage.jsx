import { useMemo, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useMutation, useQueries, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import Button from "../components/UI/Button.jsx";
import Badge from "../components/UI/Badge.jsx";
import MetricLine from "../components/Charts/MetricLine.jsx";

export default function CMDPResultsPage() {
  const [params] = useSearchParams();
  const initial = params.get("run");
  const qc = useQueryClient();

  const deleteMutation = useMutation({
    mutationFn: async (ids) => {
      await Promise.all(ids.map((id) => api.deleteExperiment(id)));
      return ids;
    },
    onSuccess: () => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      setSelected([]);
    },
  });

  const handleDelete = () => {
    if (selected.length === 0) return;
    const msg = `Delete ${selected.length} CMDP run${selected.length === 1 ? "" : "s"} from the database (and their checkpoints + logs)? This cannot be undone.`;
    if (window.confirm(msg)) deleteMutation.mutate(selected);
  };

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
    refetchInterval: 30000,
  });
  const cmdpRuns = experiments.filter((e) => e.agent === "cmdp");

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
  const iterQs = useQueries({
    queries: selected.map((id) => ({
      queryKey: ["cmdp_iters", id],
      queryFn: () => api.getCmdpIterations(id),
      refetchInterval: 5000,
    })),
  });

  const details = detailQs.filter((q) => q.data).map((q) => q.data);
  const itersByRun = useMemo(() => {
    const m = {};
    iterQs.forEach((q) => {
      if (q.data) m[q.data.run_id] = q.data.iterations || [];
    });
    return m;
  }, [iterQs]);

  const lossRSeries = useMemo(
    () =>
      selected.map((id) => ({
        name: `${id.slice(-8)} · loss_r`,
        data: (itersByRun[id] || []).map((p) => ({ episode: p.iteration, value: p.loss_r })),
      })).filter((s) => s.data.length > 0),
    [selected, itersByRun]
  );
  const lossCSeries = useMemo(
    () =>
      selected.map((id) => ({
        name: `${id.slice(-8)} · loss_c`,
        data: (itersByRun[id] || []).map((p) => ({ episode: p.iteration, value: p.loss_c })),
      })).filter((s) => s.data.length > 0),
    [selected, itersByRun]
  );
  const lambdaSeries = useMemo(
    () =>
      selected.map((id) => ({
        name: `${id.slice(-8)} · λ`,
        data: (itersByRun[id] || []).map((p) => ({ episode: p.iteration, value: p.lambda_val })),
      })).filter((s) => s.data.length > 0),
    [selected, itersByRun]
  );
  const violationSeries = useMemo(
    () =>
      selected.map((id) => ({
        name: `${id.slice(-8)} · violation`,
        data: (itersByRun[id] || [])
          .filter((p) => p.violation != null)
          .map((p) => ({ episode: p.iteration, value: p.violation })),
      })).filter((s) => s.data.length > 0),
    [selected, itersByRun]
  );

  return (
    <div>
      <Card
        title="CMDP Runs"
        sub="Click rows to overlay them on the FQI training charts below."
        action={
          <Button
            variant="danger"
            onClick={handleDelete}
            disabled={selected.length === 0 || deleteMutation.isPending}
          >
            {deleteMutation.isPending
              ? "Deleting…"
              : `Delete ${selected.length || ""} selected`}
          </Button>
        }
      >
        {cmdpRuns.length === 0 ? (
          <div className="empty">No CMDP runs yet. Launch one from the CMDP Training page.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th></th>
                <th>Run</th>
                <th>Status</th>
                <th>Servers</th>
                <th>Eval mean R</th>
                <th>Mean Power</th>
                <th>SLA rate</th>
              </tr>
            </thead>
            <tbody>
              {cmdpRuns.map((e) => (
                <tr key={e.run_id} onClick={() => toggle(e.run_id)}>
                  <td>
                    <input
                      type="checkbox"
                      checked={selected.includes(e.run_id)}
                      onChange={() => toggle(e.run_id)}
                      onClick={(ev) => ev.stopPropagation()}
                    />
                  </td>
                  <td className="mono" style={{ fontSize: 11 }}>{e.run_id}</td>
                  <td><Badge status={e.status} /></td>
                  <td className="mono">{e.n_servers}</td>
                  <td className="mono">{fmt(e.mean_reward_eval)}</td>
                  <td className="mono">{e.mean_power != null ? e.mean_power.toFixed(0) : "—"}</td>
                  <td className="mono">{e.sla_violation_rate != null ? (e.sla_violation_rate * 100).toFixed(1) + "%" : "—"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>

      {details.length > 0 && (
        <>
          <Card title="Eval Metrics (held-out test seeds)" sub="Computed after FQI by rolling out the greedy Lagrangian policy on each test seed.">
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>mean R</th>
                  <th>std R</th>
                  <th>mean power</th>
                  <th>SLA rate</th>
                  <th>jobs/ep</th>
                  <th>n_eps</th>
                </tr>
              </thead>
              <tbody>
                {details.map((d) => {
                  const ev = d.eval || {};
                  const summary = d.summary || {};
                  return (
                    <tr key={summary.run_id}>
                      <td className="mono" style={{ fontSize: 11 }}>{summary.run_id}</td>
                      <td className="mono">{fmt(ev.mean_reward)}</td>
                      <td className="mono">{fmt(ev.std_reward)}</td>
                      <td className="mono">{fmt(ev.mean_power, 0)}</td>
                      <td className="mono">{ev.sla_violation_rate != null ? (ev.sla_violation_rate * 100).toFixed(2) + "%" : "—"}</td>
                      <td className="mono">{fmt(ev.mean_jobs_completed, 1)}</td>
                      <td className="mono">{ev.n_episodes ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>

          <Card
            title="Constraint Diagnostics"
            sub="Read the slack column to tell whether the constraint was binding. Slack > 0 → policy is well within budget (constraint inactive). Slack ≈ 0 → constraint is binding (CMDP doing its job). Slack < 0 → policy still violates; λ hasn't ramped enough or dataset lacks cheap-SLA alternatives."
          >
            <table>
              <thead>
                <tr>
                  <th>Run</th>
                  <th>SLA budget ε</th>
                  <th>Mean Q_c(s, π(s))</th>
                  <th>Slack (ε − Q_c)</th>
                  <th>λ final</th>
                  <th>FQI iters</th>
                </tr>
              </thead>
              <tbody>
                {details.map((d) => {
                  const sf = d.summary_full || {};
                  const cfg = d.config || {};
                  const eps = sf.sla_budget ?? cfg.sla_budget;
                  const qc = sf.mean_qc_pi;
                  const slack = sf.constraint_slack;
                  const lam = sf.lambda_final;
                  const iters = sf.n_iterations ?? cfg.iterations;
                  return (
                    <tr key={d.summary?.run_id}>
                      <td className="mono" style={{ fontSize: 11 }}>{d.summary?.run_id}</td>
                      <td className="mono">{fmt(eps, 4)}</td>
                      <td className="mono">{fmt(qc, 4)}</td>
                      <td className="mono" style={{ color: slack != null ? (slack >= 0 ? "#065f46" : "#b91c1c") : undefined }}>
                        {fmt(slack, 4)}
                      </td>
                      <td className="mono">{fmt(lam, 3)}</td>
                      <td className="mono">{iters ?? "—"}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </Card>

          <div className="grid cols-2">
            <Card title="FQI loss — Q_r (reward critic)" sub="Smooth-L1 Bellman loss vs iteration. Should decay.">
              <MetricLine series={lossRSeries} yLabel="loss_r" xKey="episode" yKey="value" />
            </Card>
            <Card title="FQI loss — Q_c (cost / SLA critic)" sub="Bellman loss for the constraint critic.">
              <MetricLine series={lossCSeries} yLabel="loss_c" xKey="episode" yKey="value" />
            </Card>
          </div>

          <div className="grid cols-2">
            <Card title="Lagrangian λ" sub="Dual variable. Rises when the policy violates the SLA budget, falls when there's slack.">
              <MetricLine series={lambdaSeries} yLabel="λ" xKey="episode" yKey="value" />
            </Card>
            <Card title="Constraint violation" sub="E[Q_c(s, π(s))] − ε_sla. Should approach 0 as λ converges.">
              <MetricLine series={violationSeries} yLabel="E[Q_c]−ε" xKey="episode" yKey="value" referenceY={0} referenceLabel="0" />
            </Card>
          </div>
        </>
      )}
    </div>
  );
}

function fmt(v, digits = 2) {
  if (v == null) return "—";
  return Number(v).toFixed(digits);
}
