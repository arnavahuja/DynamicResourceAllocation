import { useEffect, useMemo, useRef, useState } from "react";
import { useSearchParams } from "react-router-dom";
import { useQuery } from "@tanstack/react-query";

import { api, metricsWebSocketUrl } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import Badge from "../components/UI/Badge.jsx";
import RewardCurve from "../components/Charts/RewardCurve.jsx";
import PowerSlaScatter from "../components/Charts/PowerSlaScatter.jsx";
import ServerHeatmap from "../components/Charts/ServerHeatmap.jsx";
import MetricLine, { rollingMean } from "../components/Charts/MetricLine.jsx";

export default function MonitorPage() {
  const [params, setParams] = useSearchParams();
  const runId = params.get("run") || "";

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
    refetchInterval: 5000,
  });

  const { data: status } = useQuery({
    queryKey: ["status", runId],
    queryFn: () => api.trainStatus(runId),
    enabled: !!runId,
    refetchInterval: 1500,
  });

  const [episodes, setEpisodes] = useState([]);
  const [iterations, setIterations] = useState([]);  // CMDP FQI events
  const [lastEvent, setLastEvent] = useState(null);
  const wsRef = useRef(null);

  useEffect(() => {
    if (!runId) return;
    setEpisodes([]);
    setIterations([]);
    setLastEvent(null);

    let alive = true;
    let attempts = 0;
    function connect() {
      const ws = new WebSocket(metricsWebSocketUrl(runId));
      wsRef.current = ws;
      ws.onmessage = (e) => {
        if (!alive) return;
        const msg = JSON.parse(e.data);
        setLastEvent(msg);
        if (msg.event === "episode") {
          setEpisodes((prev) => [...prev, msg]);
        } else if (msg.event === "iteration") {
          setIterations((prev) => [...prev, msg]);
        }
      };
      ws.onclose = () => {
        if (!alive) return;
        attempts += 1;
        if (attempts < 5) setTimeout(connect, 1000 * attempts);
      };
      ws.onerror = () => ws.close();
    }
    connect();
    return () => {
      alive = false;
      if (wsRef.current) wsRef.current.close();
    };
  }, [runId]);

  const rewardSeries = useMemo(() => {
    const raw = episodes.map((e) => e.reward);
    const smoothed = rollingMean(raw, 20);
    return [
      { name: "reward", data: episodes.map((e) => ({ episode: e.episode, reward: e.reward })) },
      {
        name: "rolling-20",
        color: "#F59E0B",
        data: episodes.map((e, i) => ({ episode: e.episode, "rolling-20": smoothed[i] })),
      },
    ];
  }, [episodes]);

  const powerSeries = useMemo(
    () => [
      { name: "power", color: "#EF4444", data: episodes.map((e) => ({ episode: e.episode, power: e.power })) },
    ],
    [episodes]
  );

  const slaSeries = useMemo(
    () => [
      { name: "sla", color: "#8B5CF6", data: episodes.map((e) => ({ episode: e.episode, sla: e.sla_violations })) },
    ],
    [episodes]
  );

  const scatter = useMemo(
    () => [{ name: runId, points: episodes.map((e) => ({ power: e.power, sla: e.sla_violations })) }],
    [episodes, runId]
  );

  // Approximate per-server load proxy: not yet streamed by backend at episode
  // granularity, so we render a placeholder grid sized to N servers from the
  // experiment config.
  const exp = experiments.find((x) => x.run_id === runId);
  const nServers = exp?.n_servers ?? 10;
  const optimalReward = exp?.optimal_reward ?? null;
  const isCmdp = exp?.agent === "cmdp";

  // CMDP-specific live series: pulled from "iteration" events.
  const lossRSeries = useMemo(
    () => [{
      name: "loss_r",
      color: "#1E3A8A",
      data: iterations.map((p) => ({ episode: p.iteration, value: p.loss_r })),
    }],
    [iterations]
  );
  const lossCSeries = useMemo(
    () => [{
      name: "loss_c",
      color: "#10B981",
      data: iterations.map((p) => ({ episode: p.iteration, value: p.loss_c })),
    }],
    [iterations]
  );
  const lambdaSeries = useMemo(
    () => [{
      name: "λ",
      color: "#F59E0B",
      data: iterations.map((p) => ({ episode: p.iteration, value: p.lambda })),
    }],
    [iterations]
  );
  const violationSeries = useMemo(
    () => [{
      name: "constraint violation",
      color: "#EF4444",
      data: iterations
        .filter((p) => p.constraint_violation != null)
        .map((p) => ({ episode: p.iteration, value: p.constraint_violation })),
    }],
    [iterations]
  );
  const placeholderUtils = useMemo(() => {
    const last = episodes[episodes.length - 1];
    if (!last) return new Array(nServers).fill(0);
    // Use a deterministic hash so the user gets a stable visual until we
    // add per-server streaming.
    const seed = (last.episode * 9301 + 49297) % 233280;
    return new Array(nServers).fill(0).map((_, i) => {
      const v = ((seed + i * 7919) % 1000) / 1000;
      return Math.min(1, 0.2 + v * 0.7);
    });
  }, [episodes, nServers]);

  return (
    <div>
      <Card
        title="Live Monitor"
        sub="WebSocket-streamed metrics from the training worker"
        action={
          <div style={{ display: "flex", gap: 12, alignItems: "center" }}>
            <select
              value={runId}
              onChange={(e) => setParams({ run: e.target.value })}
              style={{ minWidth: 240 }}
            >
              <option value="">— select run —</option>
              {experiments.map((x) => (
                <option key={x.run_id} value={x.run_id}>
                  {x.agent} · {x.run_id}
                </option>
              ))}
            </select>
            {status && <Badge status={status.status} />}
          </div>
        }
      >
        {!runId ? (
          <div className="empty">Pick an active run from the dropdown above.</div>
        ) : isCmdp ? (
          <div className="grid cols-3">
            <Stat label="iteration" value={status ? `${status.current_episode}/${status.total_episodes}` : "—"} />
            <Stat
              label="latest λ"
              value={iterations.length ? iterations[iterations.length - 1].lambda.toFixed(3) : "—"}
              hint="Lagrangian dual variable"
            />
            <Stat label="ETA" value={status?.eta_seconds ? `${Math.round(status.eta_seconds)}s` : "—"} />
          </div>
        ) : (
          <div className="grid cols-3">
            <Stat label="episode" value={status ? `${status.current_episode}/${status.total_episodes}` : "—"} />
            <Stat
              label="last reward"
              value={status?.last_reward != null ? status.last_reward.toFixed(2) : "—"}
              hint={optimalReward != null ? `optimal ${optimalReward.toFixed(2)}` : undefined}
            />
            <Stat label="ETA" value={status?.eta_seconds ? `${Math.round(status.eta_seconds)}s` : "—"} />
          </div>
        )}
      </Card>

      {runId && isCmdp && (
        <>
          <div className="grid cols-2">
            <Card title="FQI loss — Q_r (reward critic)" sub="Smooth-L1 Bellman loss vs FQI iteration. Should decay.">
              <MetricLine series={lossRSeries} yLabel="loss_r" />
            </Card>
            <Card title="FQI loss — Q_c (cost / SLA critic)" sub="Bellman + CQL loss for the constraint critic.">
              <MetricLine series={lossCSeries} yLabel="loss_c" />
            </Card>
          </div>

          <div className="grid cols-2">
            <Card title="Lagrangian λ" sub="Rises when policy violates the SLA budget; falls when there's slack.">
              <MetricLine series={lambdaSeries} yLabel="λ" />
            </Card>
            <Card title="Constraint violation" sub="E[Q_c(s, π(s))] − ε_sla. Should approach 0 as λ converges.">
              <MetricLine series={violationSeries} yLabel="E[Q_c]−ε" referenceY={0} referenceLabel="0" />
            </Card>
          </div>

          {lastEvent && lastEvent.event && (
            <Card title="Last event" sub="Raw WebSocket payload">
              <pre className="mono" style={{ fontSize: 12, overflow: "auto", margin: 0 }}>
                {JSON.stringify(lastEvent, null, 2)}
              </pre>
            </Card>
          )}
        </>
      )}

      {runId && !isCmdp && (
        <>
          <Card
            title="Reward (per episode)"
            sub={
              optimalReward != null
                ? `Theoretical ceiling under this run's α/P_idle/P_max/episode_length: ${optimalReward.toFixed(2)} (zero SLA violations + idle-only power)`
                : undefined
            }
          >
            <RewardCurve series={rewardSeries} optimalReward={optimalReward} />
          </Card>

          <div className="grid cols-2">
            <Card title="Power per episode" sub="Total cluster power consumption (W·timesteps) per episode">
              <MetricLine series={powerSeries} yLabel="Power (W·steps)" yKey="power" />
            </Card>
            <Card title="SLA violations per episode" sub="Cumulative breaches at episode end">
              <MetricLine series={slaSeries} yLabel="Violations" yKey="sla" />
            </Card>
          </div>

          <div className="grid cols-2">
            <Card title="Power vs. SLA" sub="Each point = one training episode">
              <PowerSlaScatter groups={scatter} />
            </Card>
            <Card title={`Server utilization (N=${nServers})`} sub="Last episode snapshot (placeholder gradient — per-server data not yet streamed)">
              <ServerHeatmap utilizations={placeholderUtils} />
            </Card>
          </div>

          {lastEvent && lastEvent.event && (
            <Card title="Last event" sub="Raw WebSocket payload">
              <pre className="mono" style={{ fontSize: 12, overflow: "auto", margin: 0 }}>
                {JSON.stringify(lastEvent, null, 2)}
              </pre>
            </Card>
          )}
        </>
      )}
    </div>
  );
}

function Stat({ label, value, hint }) {
  return (
    <div className="stat">
      <div className="label">{label}</div>
      <div className="value">{value}</div>
      {hint && <div className="label" style={{ textTransform: "none", letterSpacing: 0 }}>{hint}</div>}
    </div>
  );
}
