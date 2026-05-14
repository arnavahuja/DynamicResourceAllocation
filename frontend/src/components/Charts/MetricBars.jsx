import {
  BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Cell,
} from "recharts";

// Stable per-agent colors so the same agent has the same color across charts.
const AGENT_COLORS = {
  dqn: "#1E3A8A",
  ppo: "#10B981",
  agentic: "#F59E0B",
  cmdp: "#EF4444",
  round_robin: "#8B5CF6",
  sjf: "#06B6D4",
  ffd: "#EC4899",
};

export function colorForAgent(agent) {
  return AGENT_COLORS[agent] || "#6b7280";
}

export default function MetricBars({
  data,
  metric,
  height = 260,
  yLabel = "",
  formatY = (v) => v,
  labelKey = "label",
  agentKey = "agent",
}) {
  // data = [{ label: 'dqn_xxx', agent: 'dqn', [metric]: number }]
  if (!data || data.length === 0) return <div className="empty">No runs to chart</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <BarChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 24 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
        <XAxis
          dataKey={labelKey}
          tick={{ fontSize: 10, fontFamily: "IBM Plex Mono" }}
          angle={-30}
          textAnchor="end"
          height={70}
          interval={0}
        />
        <YAxis
          tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }}
          tickFormatter={formatY}
          domain={["auto", "auto"]}
          label={{ value: yLabel, angle: -90, position: "insideLeft", style: { fontSize: 11, fill: "#6b7280" } }}
        />
        <Tooltip
          contentStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12 }}
          formatter={(v) => formatY(v)}
        />
        <Bar
          dataKey={metric}
          label={{
            position: "top",
            fontSize: 10,
            fontFamily: "IBM Plex Mono",
            fill: "#0a1628",
            formatter: (v) => (v == null ? "" : formatY(v)),
          }}
        >
          {data.map((row, i) => (
            <Cell key={i} fill={colorForAgent(row[agentKey])} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}
