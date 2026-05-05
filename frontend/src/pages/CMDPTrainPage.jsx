import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";

import { api } from "../lib/api.js";
import Card from "../components/UI/Card.jsx";
import Button from "../components/UI/Button.jsx";
import Badge from "../components/UI/Badge.jsx";

const FORM_DEFAULTS = {
  nServers: 10,
  epLen: 1000,
  alpha: 1.0,
  beta: 50.0,
  seed: 0,
  clusterType: "homogeneous",
  datasetPath: "data/processed/offline.parquet",
  autoGenerate: true,
  datasetEpisodes: 200,
  datasetRandomEps: 0.10,
  behaviorWeightRr: 0.10,
  behaviorWeightSjf: 0.55,
  behaviorWeightFfd: 0.35,
  iterations: 20000,
  batchSize: 256,
  slaBudget: 0.05,
  cqlAlpha: 1.0,
  rewardKind: "neg_power",
  targetUpdateFreq: 500,
  dualUpdateFreq: 200,
  lambdaInit: 0.5,
  lambdaLr: 0.05,
  dualSignal: "empirical",
  dualEvalSeeds: 3,
  dualEvalSteps: 200,
  nTestSeeds: 50,
};
const formCache = { ...FORM_DEFAULTS };
let _hydrated = false;

export default function CMDPTrainPage() {
  const nav = useNavigate();
  const qc = useQueryClient();

  const [nServers, setNServers] = useState(() => formCache.nServers);
  const [epLen, setEpLen] = useState(() => formCache.epLen);
  const [alpha, setAlpha] = useState(() => formCache.alpha);
  const [beta, setBeta] = useState(() => formCache.beta);
  const [seed, setSeed] = useState(() => formCache.seed);
  const [clusterType, setClusterType] = useState(() => formCache.clusterType);
  const [datasetPath, setDatasetPath] = useState(() => formCache.datasetPath);
  const [autoGenerate, setAutoGenerate] = useState(() => formCache.autoGenerate);
  const [datasetEpisodes, setDatasetEpisodes] = useState(() => formCache.datasetEpisodes);
  const [datasetRandomEps, setDatasetRandomEps] = useState(() => formCache.datasetRandomEps);
  const [behaviorWeightRr, setBehaviorWeightRr] = useState(() => formCache.behaviorWeightRr);
  const [behaviorWeightSjf, setBehaviorWeightSjf] = useState(() => formCache.behaviorWeightSjf);
  const [behaviorWeightFfd, setBehaviorWeightFfd] = useState(() => formCache.behaviorWeightFfd);
  const [iterations, setIterations] = useState(() => formCache.iterations);
  const [batchSize, setBatchSize] = useState(() => formCache.batchSize);
  const [slaBudget, setSlaBudget] = useState(() => formCache.slaBudget);
  const [cqlAlpha, setCqlAlpha] = useState(() => formCache.cqlAlpha);
  const [rewardKind, setRewardKind] = useState(() => formCache.rewardKind);
  const [targetUpdateFreq, setTargetUpdateFreq] = useState(() => formCache.targetUpdateFreq);
  const [dualUpdateFreq, setDualUpdateFreq] = useState(() => formCache.dualUpdateFreq);
  const [lambdaInit, setLambdaInit] = useState(() => formCache.lambdaInit);
  const [lambdaLr, setLambdaLr] = useState(() => formCache.lambdaLr);
  const [dualSignal, setDualSignal] = useState(() => formCache.dualSignal);
  const [dualEvalSeeds, setDualEvalSeeds] = useState(() => formCache.dualEvalSeeds);
  const [dualEvalSteps, setDualEvalSteps] = useState(() => formCache.dualEvalSteps);
  const [nTestSeeds, setNTestSeeds] = useState(() => formCache.nTestSeeds);
  const [datasetStatus, setDatasetStatus] = useState(null);

  useEffect(() => {
    if (_hydrated) return;
    api.configDefaults()
      .then((d) => {
        _hydrated = true;
        if (d.n_servers != null) { formCache.nServers = d.n_servers; setNServers(d.n_servers); }
        if (d.episode_length != null) { formCache.epLen = d.episode_length; setEpLen(d.episode_length); }
        if (d.alpha != null) { formCache.alpha = d.alpha; setAlpha(d.alpha); }
        if (d.beta != null) { formCache.beta = d.beta; setBeta(d.beta); }
      })
      .catch(() => {});
  }, []);

  useEffect(() => {
    Object.assign(formCache, {
      nServers, epLen, alpha, beta, seed, clusterType, datasetPath,
      autoGenerate, datasetEpisodes, datasetRandomEps,
      behaviorWeightRr, behaviorWeightSjf, behaviorWeightFfd,
      iterations, batchSize, slaBudget,
      cqlAlpha, rewardKind, targetUpdateFreq, dualUpdateFreq,
      lambdaInit, lambdaLr, dualSignal, dualEvalSeeds, dualEvalSteps,
      nTestSeeds,
    });
  }, [nServers, epLen, alpha, beta, seed, clusterType, datasetPath,
      autoGenerate, datasetEpisodes, datasetRandomEps,
      behaviorWeightRr, behaviorWeightSjf, behaviorWeightFfd,
      iterations, batchSize, slaBudget,
      cqlAlpha, rewardKind, targetUpdateFreq, dualUpdateFreq,
      lambdaInit, lambdaLr, dualSignal, dualEvalSeeds, dualEvalSteps,
      nTestSeeds]);

  useEffect(() => {
    api.offlineDatasetExists(datasetPath)
      .then(setDatasetStatus)
      .catch(() => setDatasetStatus(null));
  }, [datasetPath]);

  const { data: experiments = [] } = useQuery({
    queryKey: ["experiments"],
    queryFn: api.listExperiments,
    refetchInterval: 5000,
  });
  const cmdpRuns = experiments.filter((e) => e.agent === "cmdp");

  const launch = useMutation({
    mutationFn: api.startOfflineTraining,
    onSuccess: ({ run_id }) => {
      qc.invalidateQueries({ queryKey: ["experiments"] });
      nav(`/monitor?run=${run_id}`);
    },
  });

  return (
    <div>
      <Card
        title="Constrained MDP — Offline Training"
        sub="Lagrangian-relaxation FQI on a static Parquet dataset of logged transitions. No env interaction during training."
      >
        <div className="grid cols-2">
          <div>
            <label>Dataset path</label>
            <input
              type="text"
              value={datasetPath}
              onChange={(e) => setDatasetPath(e.target.value)}
              style={{ fontFamily: "monospace", fontSize: 12 }}
            />
            <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
              {datasetStatus
                ? datasetStatus.exists
                  ? `Found at ${datasetStatus.path}`
                  : `Missing — will auto-generate if box below is checked.`
                : "…"}
            </div>
          </div>
          <div>
            <label>Seed</label>
            <input type="number" value={seed} onChange={(e) => setSeed(+e.target.value)} />
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          <label style={{ display: "flex", alignItems: "center", gap: 8 }}>
            <input
              type="checkbox"
              checked={autoGenerate}
              onChange={(e) => setAutoGenerate(e.target.checked)}
            />
            Auto-generate dataset if missing (heuristic-mix behavior policy)
          </label>
          {autoGenerate && (
            <>
              <div className="grid cols-2" style={{ marginTop: 8 }}>
                <div>
                  <label>Behavior-rollout episodes: {datasetEpisodes}</label>
                  <input
                    type="range" min={5} max={1000} step={5}
                    value={datasetEpisodes}
                    onChange={(e) => setDatasetEpisodes(+e.target.value)}
                  />
                  <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
                    More episodes = wider state coverage. Q_c can only be
                    trusted on (s,a) the dataset has actually visited.
                  </div>
                </div>
                <div>
                  <label>Random-action injection: {(datasetRandomEps * 100).toFixed(0)}%</label>
                  <input
                    type="range" min={0} max={1} step={0.05}
                    value={datasetRandomEps}
                    onChange={(e) => setDatasetRandomEps(+e.target.value)}
                  />
                  <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
                    Fraction of steps where a uniform-random legal action is
                    taken. Higher = broader coverage but worse SLA in the
                    dataset (random hurts SLA on this env). Try 5–15%.
                  </div>
                </div>
              </div>

              <div style={{ marginTop: 12, padding: "8px 10px", border: "1px solid #e5e7eb", borderRadius: 6 }}>
                <div style={{ fontSize: 12, fontWeight: 600, marginBottom: 6 }}>
                  Behavior-policy mix
                </div>
                <div style={{ fontSize: 11, color: "#6b7280", marginBottom: 8 }}>
                  Weights are normalised. SJF/FFD violate SLA less than RR
                  on this env, so skewing toward them lowers the dataset's
                  SLA floor — which is the ceiling CMDP+CQL can converge to.
                  Sum:{" "}
                  <span className="mono">
                    {(behaviorWeightRr + behaviorWeightSjf + behaviorWeightFfd).toFixed(2)}
                  </span>
                </div>
                <div className="grid cols-3">
                  <div>
                    <label>RoundRobin: {behaviorWeightRr.toFixed(2)}</label>
                    <input type="range" min={0} max={1} step={0.05}
                      value={behaviorWeightRr}
                      onChange={(e) => setBehaviorWeightRr(+e.target.value)} />
                  </div>
                  <div>
                    <label>SJF: {behaviorWeightSjf.toFixed(2)}</label>
                    <input type="range" min={0} max={1} step={0.05}
                      value={behaviorWeightSjf}
                      onChange={(e) => setBehaviorWeightSjf(+e.target.value)} />
                  </div>
                  <div>
                    <label>FFD: {behaviorWeightFfd.toFixed(2)}</label>
                    <input type="range" min={0} max={1} step={0.05}
                      value={behaviorWeightFfd}
                      onChange={(e) => setBehaviorWeightFfd(+e.target.value)} />
                  </div>
                </div>
              </div>
            </>
          )}
        </div>

        <div className="grid cols-3" style={{ marginTop: 16 }}>
          <div>
            <label>Servers (N): {nServers}</label>
            <input type="range" min={2} max={100} value={nServers} onChange={(e) => setNServers(+e.target.value)} />
          </div>
          <div>
            <label>Episode length</label>
            <input type="number" value={epLen} onChange={(e) => setEpLen(+e.target.value)} />
          </div>
          <div>
            <label>Test (held-out) seeds</label>
            <input type="number" min={0} value={nTestSeeds} onChange={(e) => setNTestSeeds(+e.target.value)} />
          </div>
        </div>

        <div style={{ marginTop: 16 }}>
          <label>Cluster type</label>
          <div className="seg">
            <button className={clusterType === "homogeneous" ? "active" : ""} onClick={() => setClusterType("homogeneous")}>
              Homogeneous
            </button>
            <button className={clusterType === "heterogeneous" ? "active" : ""} onClick={() => setClusterType("heterogeneous")}>
              Heterogeneous
            </button>
          </div>
        </div>

        <div className="grid cols-3" style={{ marginTop: 16 }}>
          <div>
            <label>α (eval power weight): {alpha.toFixed(2)}</label>
            <input type="range" min={0} max={5} step={0.05} value={alpha} onChange={(e) => setAlpha(+e.target.value)} />
          </div>
          <div>
            <label>β (eval SLA weight): {beta.toFixed(1)}</label>
            <input type="range" min={0} max={200} step={1} value={beta} onChange={(e) => setBeta(+e.target.value)} />
          </div>
          <div>
            <label>SLA budget ε: {slaBudget.toFixed(3)}</label>
            <input type="range" min={0} max={1} step={0.005} value={slaBudget} onChange={(e) => setSlaBudget(+e.target.value)} />
            <div style={{ fontSize: 11, color: "#6b7280" }}>
              ε is the per-step violation rate budget (cost is rescaled by
              (1−γ) so Q_c ∈ [0,1]). ε=0.05 ≈ "≤5% of steps may violate."
              Directly comparable to the eval SLA rate. Try 0.02 for a
              tight constraint, 0.10 for loose.
            </div>
          </div>
        </div>

        <div className="grid cols-1" style={{ marginTop: 16 }}>
          <div>
            <label>Conservative-Q penalty α (CQL on Q_c): {cqlAlpha.toFixed(2)}</label>
            <input type="range" min={0} max={10} step={0.1} value={cqlAlpha} onChange={(e) => setCqlAlpha(+e.target.value)} />
            <div style={{ fontSize: 11, color: "#6b7280", marginTop: 4 }}>
              0 = vanilla FQI (Q_c trusts every action it predicts on).
              Higher = pushes Q_c UP on out-of-distribution actions, so the
              Lagrangian-greedy policy avoids them. Fights the offline-RL
              extrapolation error where the policy picks unseen actions
              that Q_c wrongly thinks are cheap. Start at 1.0; raise to 3–5
              if the eval SLA rate stays well above ε.
            </div>
          </div>
        </div>

        <div className="grid cols-3" style={{ marginTop: 16 }}>
          <div>
            <label>FQI iterations</label>
            <input type="number" value={iterations} onChange={(e) => setIterations(+e.target.value)} />
          </div>
          <div>
            <label>Batch size</label>
            <input type="number" value={batchSize} onChange={(e) => setBatchSize(+e.target.value)} />
          </div>
          <div>
            <label>Reward kind</label>
            <select value={rewardKind} onChange={(e) => setRewardKind(e.target.value)}>
              <option value="neg_power">neg_power (Q_r = -power)</option>
              <option value="env">env (Q_r = combined env reward)</option>
            </select>
          </div>
        </div>

        <div className="grid cols-2" style={{ marginTop: 16 }}>
          <div>
            <label>Target update freq (iters)</label>
            <input type="number" value={targetUpdateFreq} onChange={(e) => setTargetUpdateFreq(+e.target.value)} />
          </div>
          <div>
            <label>λ (dual) update freq (iters)</label>
            <input type="number" value={dualUpdateFreq} onChange={(e) => setDualUpdateFreq(+e.target.value)} />
          </div>
          <div>
            <label title="Initial λ. Warm-start at 0.5 → policy starts conservative; λ relaxes if there's slack.">λ init</label>
            <input type="number" step="0.05" value={lambdaInit} onChange={(e) => setLambdaInit(+e.target.value)} />
          </div>
          <div>
            <label title="Step size for λ ascent on (rate − ε). Bumped from 1e-2 → 5e-2 so λ converges within run.">λ learning rate</label>
            <input type="number" step="0.005" value={lambdaLr} onChange={(e) => setLambdaLr(+e.target.value)} />
          </div>
          <div>
            <label title="'empirical' = drive λ from a short on-policy SLA-rate rollout (recommended; bypasses CQL inflation of Q_c). 'q_c' = legacy.">Dual signal</label>
            <select value={dualSignal} onChange={(e) => setDualSignal(e.target.value)}>
              <option value="empirical">empirical (rollout)</option>
              <option value="q_c">q_c (legacy)</option>
            </select>
          </div>
          <div>
            <label title="# rollout seeds per dual update when using empirical signal.">Dual eval seeds</label>
            <input type="number" value={dualEvalSeeds} onChange={(e) => setDualEvalSeeds(+e.target.value)} />
          </div>
          <div>
            <label title="Max steps per dual rollout episode.">Dual eval steps</label>
            <input type="number" value={dualEvalSteps} onChange={(e) => setDualEvalSteps(+e.target.value)} />
          </div>
        </div>

        <div style={{ marginTop: 20, display: "flex", gap: 12, alignItems: "center" }}>
          <Button
            onClick={() =>
              launch.mutate({
                n_servers: nServers,
                episode_length: epLen,
                cluster_type: clusterType,
                alpha, beta, seed,
                dataset_path: datasetPath,
                auto_generate_dataset: autoGenerate,
                dataset_episodes: datasetEpisodes,
                dataset_random_eps: datasetRandomEps,
                behavior_weight_rr: behaviorWeightRr,
                behavior_weight_sjf: behaviorWeightSjf,
                behavior_weight_ffd: behaviorWeightFfd,
                iterations,
                batch_size: batchSize,
                sla_budget: slaBudget,
                cql_alpha: cqlAlpha,
                reward_kind: rewardKind,
                target_update_freq: targetUpdateFreq,
                dual_update_freq: dualUpdateFreq,
                lambda_init: lambdaInit,
                lambda_lr: lambdaLr,
                dual_signal: dualSignal,
                dual_eval_seeds: dualEvalSeeds,
                dual_eval_steps: dualEvalSteps,
                n_test_seeds: nTestSeeds,
              })
            }
            disabled={launch.isPending}
          >
            {launch.isPending ? "Launching…" : "Launch CMDP Run"}
          </Button>
          {launch.isError && <span style={{ color: "var(--danger)" }}>Failed: {launch.error.message}</span>}
        </div>
      </Card>

      <Card title="CMDP Runs" sub="Auto-refreshes every 5s. Click a run to open its results page.">
        {cmdpRuns.length === 0 ? (
          <div className="empty">No CMDP runs yet.</div>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Run</th>
                <th>Status</th>
                <th>Servers</th>
                <th>Eval mean R</th>
                <th>Mean Power</th>
                <th>SLA rate</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {cmdpRuns.map((e) => (
                <tr key={e.run_id} onClick={() => nav(`/cmdp/results?run=${e.run_id}`)}>
                  <td className="mono" style={{ fontSize: 11 }}>{e.run_id}</td>
                  <td><Badge status={e.status} /></td>
                  <td className="mono">{e.n_servers}</td>
                  <td className="mono">{e.mean_reward_eval != null ? e.mean_reward_eval.toFixed(2) : "—"}</td>
                  <td className="mono">{e.mean_power != null ? e.mean_power.toFixed(0) : "—"}</td>
                  <td className="mono">{e.sla_violation_rate != null ? (e.sla_violation_rate * 100).toFixed(1) + "%" : "—"}</td>
                  <td onClick={(ev) => ev.stopPropagation()}>
                    {e.status === "running" && <a href={`/monitor?run=${e.run_id}`}>monitor →</a>}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
