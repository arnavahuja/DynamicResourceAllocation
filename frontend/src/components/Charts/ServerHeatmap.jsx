function utilColor(u) {
  // Green → Yellow → Red gradient over [0,1]
  const clamped = Math.max(0, Math.min(1, u));
  // HSL: 140 (green) → 0 (red)
  const hue = 140 - clamped * 140;
  const light = 38 + (1 - clamped) * 18;
  return `hsl(${hue}, 65%, ${light}%)`;
}

export default function ServerHeatmap({ utilizations }) {
  if (!utilizations || utilizations.length === 0)
    return <div className="empty">No server data</div>;
  const cols = Math.min(utilizations.length, 10);
  return (
    <div
      className="heatmap"
      style={{ gridTemplateColumns: `repeat(${cols}, 1fr)` }}
    >
      {utilizations.map((u, i) => (
        <div
          key={i}
          className="cell"
          style={{ background: utilColor(u) }}
          title={`Server ${i}: ${(u * 100).toFixed(0)}%`}
        >
          {(u * 100).toFixed(0)}
        </div>
      ))}
    </div>
  );
}
