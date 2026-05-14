import { useEffect, useState } from "react";
import { api } from "../../lib/api.js";

export default function TopBar({ title }) {
  const [healthy, setHealthy] = useState(null);
  useEffect(() => {
    let on = true;
    const check = async () => {
      try { await api.health(); if (on) setHealthy(true); }
      catch { if (on) setHealthy(false); }
    };
    check();
    const id = setInterval(check, 10000);
    return () => { on = false; clearInterval(id); };
  }, []);
  return (
    <div className="topbar">
      <div className="title">{title}</div>
      <div className="api-status">
        <span className={`dot ${healthy ? "ok" : "bad"}`} />
        api {healthy === null ? "…" : healthy ? "online" : "offline"}
      </div>
    </div>
  );
}
