export default function Card({ title, sub, children, action }) {
  return (
    <div className="card">
      {(title || action) && (
        <div style={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", marginBottom: 12 }}>
          <div>
            {title && <h2>{title}</h2>}
            {sub && <div className="sub">{sub}</div>}
          </div>
          {action}
        </div>
      )}
      {children}
    </div>
  );
}
