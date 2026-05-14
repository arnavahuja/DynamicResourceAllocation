export default function Badge({ status }) {
  const cls = `badge ${status || "pending"}`;
  return <span className={cls}>{status || "pending"}</span>;
}
