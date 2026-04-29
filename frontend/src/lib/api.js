// Thin fetch wrappers around the FastAPI backend.
// In dev, vite proxies /api → http://localhost:8000.

const BASE = "/api";

async function jsonFetch(path, opts = {}) {
  const r = await fetch(`${BASE}${path}`, {
    headers: { "Content-Type": "application/json" },
    ...opts,
  });
  if (!r.ok) throw new Error(`${r.status} ${r.statusText}`);
  return r.json();
}

export const api = {
  health: () => jsonFetch("/health"),
  startTraining: (body) =>
    jsonFetch("/train", { method: "POST", body: JSON.stringify(body) }),
  trainStatus: (runId) => jsonFetch(`/train/${runId}/status`),
  cancelTraining: (runId) =>
    jsonFetch(`/train/${runId}`, { method: "DELETE" }),
  listExperiments: () => jsonFetch("/experiments"),
  getExperiment: (runId) => jsonFetch(`/experiments/${runId}/results`),
  deleteExperiment: (runId) =>
    jsonFetch(`/experiments/${runId}`, { method: "DELETE" }),
  simulate: (body) =>
    jsonFetch("/simulate", { method: "POST", body: JSON.stringify(body) }),
};

export function metricsWebSocketUrl(runId) {
  // Dev: vite proxies /ws → ws://localhost:8000/ws
  // Prod: same-origin
  const proto = window.location.protocol === "https:" ? "wss:" : "ws:";
  return `${proto}//${window.location.host}/ws/metrics/${runId}`;
}
