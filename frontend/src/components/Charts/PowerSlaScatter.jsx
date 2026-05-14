import {
  ScatterChart, Scatter, XAxis, YAxis, ZAxis, Tooltip, CartesianGrid, ResponsiveContainer, Legend,
} from "recharts";

const COLORS = ["#1E3A8A", "#10B981", "#F59E0B", "#EF4444", "#8B5CF6", "#06B6D4"];

export default function PowerSlaScatter({ groups, height = 280 }) {
  // groups = [{ name, points: [{power, sla}] }]
  if (!groups || groups.length === 0) return <div className="empty">No data</div>;
  return (
    <ResponsiveContainer width="100%" height={height}>
      <ScatterChart margin={{ top: 8, right: 16, left: 0, bottom: 0 }}>
        <CartesianGrid stroke="#e5e7eb" strokeDasharray="3 3" />
        <XAxis type="number" dataKey="power" name="Power" tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} label={{ value: "Power", position: "insideBottom", offset: -4, style: { fontSize: 11, fill: "#6b7280" } }} />
        <YAxis type="number" dataKey="sla" name="SLA viol." tick={{ fontSize: 11, fontFamily: "IBM Plex Mono" }} label={{ value: "SLA viol.", angle: -90, position: "insideLeft", style: { fontSize: 11, fill: "#6b7280" } }} />
        <ZAxis range={[40, 80]} />
        <Tooltip contentStyle={{ fontFamily: "IBM Plex Mono", fontSize: 12 }} cursor={{ strokeDasharray: "3 3" }} />
        <Legend wrapperStyle={{ fontSize: 12 }} />
        {groups.map((g, i) => (
          <Scatter key={g.name} name={g.name} data={g.points} fill={COLORS[i % COLORS.length]} />
        ))}
      </ScatterChart>
    </ResponsiveContainer>
  );
}
