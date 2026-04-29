import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend, ReferenceLine,
} from "recharts";

const COLORS = ["#1E3A8A", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"];

export default function RewardCurve({ series, height = 280, yLabel = "Reward", optimalReward = null }) {
  // series = [{ name, data: [{episode, reward}, ...] }]
  if (!series || series.length === 0) return <div className="empty">No episodes yet</div>;

  // Merge into wide format keyed by episode for recharts. Each series's
  // data point may store its value under either `reward` (legacy) or under
  // a key matching the series name — accept both.
  const all = new Map();
  series.forEach((s) => {
    s.data.forEach((p) => {
      if (!all.has(p.episode)) all.set(p.episode, { episode: p.episode });
      const v = p[s.name] !== undefined ? p[s.name] : p.reward;
      all.get(p.episode)[s.name] = v;
    });
  });
  const data = [...all.values()].sort((a, b) => a.episode - b.episode);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
        <XAxis dataKey="episode" tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <YAxis tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} label={{ value: yLabel, angle: -90, position: "insideLeft", style: { fontSize: 11, fill: "#6b7280" } }} />
        <Tooltip contentStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12 }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {series.map((s, i) => (
          <Line
            key={s.name}
            type="monotone"
            dataKey={s.name}
            stroke={s.color || COLORS[i % COLORS.length]}
            strokeDasharray={s.dash ? "4 4" : undefined}
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
        ))}
        {optimalReward != null && (
          <ReferenceLine
            y={optimalReward}
            stroke="#10B981"
            strokeDasharray="6 4"
            strokeWidth={2}
            label={{
              value: `optimal ${optimalReward.toFixed(1)}`,
              position: "insideTopRight",
              fill: "#065f46",
              fontSize: 11,
              fontFamily: "IBM Plex Mono",
            }}
          />
        )}
      </LineChart>
    </ResponsiveContainer>
  );
}
