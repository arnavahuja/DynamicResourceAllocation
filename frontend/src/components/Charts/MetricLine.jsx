import {
  LineChart, Line, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend, ReferenceLine,
} from "recharts";

const COLORS = ["#1E3A8A", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4", "#EC4899", "#0EA5E9"];

/**
 * Generic per-episode line chart.
 *
 * series  = [{ name, data: [{episode, value}] }]
 *           ...one series per line. dataKey = "value" for every series.
 *           Use the `multiKey` prop to override (for stacked overlay).
 * yLabel  = string
 * referenceY = optional horizontal reference line value
 * referenceLabel = label for that reference line
 */
export default function MetricLine({
  series,
  height = 260,
  yLabel = "",
  xKey = "episode",
  yKey = "value",
  referenceY = null,
  referenceLabel = null,
  referenceColor = "#10B981",
}) {
  if (!series || series.length === 0)
    return <div className="empty">No data</div>;

  // Merge into wide format keyed by episode
  const all = new Map();
  series.forEach((s) => {
    s.data.forEach((p) => {
      if (!all.has(p[xKey])) all.set(p[xKey], { [xKey]: p[xKey] });
      all.get(p[xKey])[s.name] = p[yKey];
    });
  });
  const data = [...all.values()].sort((a, b) => a[xKey] - b[xKey]);

  return (
    <ResponsiveContainer width="100%" height={height}>
      <LineChart data={data} margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
        <XAxis dataKey={xKey} tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} />
        <YAxis
          tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }}
          label={{
            value: yLabel,
            angle: -90,
            position: "insideLeft",
            style: { fontSize: 11, fill: "#6b7280" },
          }}
        />
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
        {referenceY != null && (
          <ReferenceLine
            y={referenceY}
            stroke={referenceColor}
            strokeDasharray="6 4"
            strokeWidth={2}
            label={{
              value: referenceLabel ?? `${referenceY}`,
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

export function rollingMean(values, window = 20) {
  const out = [];
  let sum = 0;
  for (let i = 0; i < values.length; i++) {
    sum += values[i];
    if (i >= window) sum -= values[i - window];
    const denom = Math.min(i + 1, window);
    out.push(sum / denom);
  }
  return out;
}
