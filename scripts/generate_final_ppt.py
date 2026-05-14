"""Generate the EECS 6893 final-presentation deck for this project.

Style mirrors the proposal PDF: dark teal background, ivory body, accent
italic headings. Eight slides covering proposal recap, env/MDP, methods,
training results, the offline-RL detour, and conclusion. Numbers are
pulled live from experiments.db so the deck always reflects the current
state of the runs.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Inches, Pt, Emu

from backend.models import db


# ── Palette (sampled from the proposal PDF) ─────────────────────────────
BG = RGBColor(0x36, 0x4A, 0x4A)          # deep teal
BG_DARK = RGBColor(0x2C, 0x3D, 0x3D)     # darker teal accent ellipse
PANEL = RGBColor(0x42, 0x57, 0x57)       # slightly lighter for content panels
INK = RGBColor(0xEF, 0xE9, 0xD7)         # ivory body text
INK_MUTED = RGBColor(0xC0, 0xBC, 0xAE)
ACCENT = RGBColor(0x8E, 0xB3, 0xB0)      # muted teal-cyan for italics


def _solid(shape, color):
    f = shape.fill
    f.solid()
    f.fore_color.rgb = color
    shape.line.fill.background()


def _add_title(slide, text, italic_sub=None):
    box = slide.shapes.add_textbox(Inches(0.6), Inches(0.45), Inches(5.5), Inches(1.6))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = text
    r.font.name = "Cambria"
    r.font.bold = True
    r.font.size = Pt(40)
    r.font.color.rgb = INK
    if italic_sub:
        p2 = tf.add_paragraph()
        p2.alignment = PP_ALIGN.LEFT
        r2 = p2.add_run()
        r2.text = italic_sub
        r2.font.italic = True
        r2.font.name = "Cambria"
        r2.font.size = Pt(18)
        r2.font.color.rgb = ACCENT


def _add_panel(slide, x, y, w, h, header, body_lines, header_bold=True,
               header_size=16, body_size=12, header_color=INK):
    """Rounded panel with a bold header line and bullet/body text."""
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, x, y, w, h)
    shape.adjustments[0] = 0.06
    _solid(shape, PANEL)
    tf = shape.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.18)
    tf.margin_right = Inches(0.18)
    tf.margin_top = Inches(0.14)
    tf.margin_bottom = Inches(0.14)

    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.LEFT
    r = p.add_run()
    r.text = header
    r.font.name = "Cambria"
    r.font.bold = header_bold
    r.font.size = Pt(header_size)
    r.font.color.rgb = header_color

    for line in body_lines:
        para = tf.add_paragraph()
        para.alignment = PP_ALIGN.LEFT
        para.space_before = Pt(4)
        if isinstance(line, tuple):
            label, body = line
            r1 = para.add_run()
            r1.text = label + " "
            r1.font.bold = True
            r1.font.name = "Calibri"
            r1.font.size = Pt(body_size)
            r1.font.color.rgb = INK
            r2 = para.add_run()
            r2.text = body
            r2.font.name = "Calibri"
            r2.font.size = Pt(body_size)
            r2.font.color.rgb = INK_MUTED
        else:
            r1 = para.add_run()
            r1.text = line
            r1.font.name = "Calibri"
            r1.font.size = Pt(body_size)
            r1.font.color.rgb = INK_MUTED


def _set_bg(slide):
    bg = slide.shapes.add_shape(
        MSO_SHAPE.RECTANGLE, 0, 0,
        slide.part.package.presentation_part.presentation.slide_width,
        slide.part.package.presentation_part.presentation.slide_height,
    )
    _solid(bg, BG)
    bg.shadow.inherit = False


def _add_left_ellipse(slide):
    """The accent ellipse anchoring titles, like the proposal deck."""
    e = slide.shapes.add_shape(
        MSO_SHAPE.OVAL,
        Inches(-3.2), Inches(-1.5), Inches(8.5), Inches(9.0),
    )
    _solid(e, BG_DARK)


# ── Pull live metrics ──────────────────────────────────────────────────
def _row_view(r):
    cfg = json.loads(r.get("config_json") or "{}")
    ev = json.loads(r.get("eval_json") or "{}") if r.get("eval_json") else {}
    summ = json.loads(r.get("summary_json") or "{}") if r.get("summary_json") else {}
    return cfg, ev, summ


def collect_results_split():
    """Two tables, one per workload, each indexed by agent. Within each
    workload bucket we keep the most recent 2000-episode completed run
    per agent so the comparison is apples-to-apples.

    Returns (synthetic_dict, google_v2_dict). Each value is
    (run_id, sla, cfg, eval_dict, summary_dict).
    """
    rows = [r for r in db.list_experiments() if r["status"] == "completed"]
    # list_experiments returns DESC by created_at, so the first match per
    # agent is automatically the most recent.
    synth: dict[str, tuple] = {}
    gv2: dict[str, tuple] = {}
    for r in rows:
        cfg, ev, summ = _row_view(r)
        if not ev:
            continue
        if cfg.get("episodes") != 2000:
            continue
        a = r["agent"]
        is_real = bool(cfg.get("use_real_traces"))
        bucket = gv2 if is_real and cfg.get("trace_family") == "google_v2_sampled" \
            else (synth if not is_real else None)
        if bucket is None:
            continue
        if a not in bucket:
            bucket[a] = (r["run_id"], ev.get("sla_violation_rate", 0.0),
                         cfg, ev, summ)
    return synth, gv2


# ── Slides ─────────────────────────────────────────────────────────────
def slide_title(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s)
    _add_left_ellipse(s)

    box = s.shapes.add_textbox(Inches(0.7), Inches(2.3), Inches(7.5), Inches(2.8))
    tf = box.text_frame
    tf.word_wrap = True
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "Dynamic Resource\nAllocation in Cloud\nComputing"
    r.font.name = "Cambria"
    r.font.bold = True
    r.font.size = Pt(54)
    r.font.color.rgb = INK
    p2 = tf.add_paragraph()
    p2.space_before = Pt(18)
    r2 = p2.add_run()
    r2.text = "via Deep Reinforcement Learning"
    r2.font.italic = True
    r2.font.name = "Cambria"
    r2.font.size = Pt(22)
    r2.font.color.rgb = ACCENT

    # Right-side info card
    card = s.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE,
                              Inches(9.2), Inches(2.3), Inches(3.4), Inches(2.6))
    card.adjustments[0] = 0.05
    _solid(card, PANEL)
    tf = card.text_frame
    tf.word_wrap = True
    tf.margin_left = Inches(0.22); tf.margin_right = Inches(0.22)
    tf.margin_top = Inches(0.20); tf.margin_bottom = Inches(0.20)
    p = tf.paragraphs[0]
    r = p.add_run()
    r.text = "EECS 6893"
    r.font.bold = True; r.font.name = "Cambria"; r.font.size = Pt(22); r.font.color.rgb = INK
    p2 = tf.add_paragraph()
    r2 = p2.add_run()
    r2.text = "Final Presentation"
    r2.font.name = "Cambria"; r2.font.size = Pt(15); r2.font.color.rgb = INK_MUTED
    p3 = tf.add_paragraph(); p3.space_before = Pt(14)
    r3 = p3.add_run()
    r3.text = ("Minimizing power consumption\n"
               "while enforcing SLA constraints\n"
               "through intelligent scheduling")
    r3.font.name = "Calibri"; r3.font.size = Pt(12); r3.font.color.rgb = INK_MUTED


def slide_proposal_recap(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "Project\nRecap", "from proposal to final result")

    # Top panel: problem
    _add_panel(s, Inches(6.4), Inches(0.5), Inches(6.4), Inches(2.2),
               "What we set out to build",
               [
                "A dynamic scheduler that decides, in real time, which server "
                "handles each incoming job — minimising total cluster power "
                "while keeping SLA violations bounded.",
                "Driven by a custom Gymnasium simulation, learned policies in "
                "PyTorch, benchmarked against three classical heuristics and "
                "a Constrained-MDP offline learner.",
               ],
               header_size=15, body_size=11)

    # MDP panel
    _add_panel(s, Inches(6.4), Inches(2.9), Inches(3.1), Inches(4.2),
               "MDP",
               [
                ("State Sₜ:", "per-server CPU/mem utilisation, queue head, time."),
                ("Action Aₜ:", "assign head-of-queue job to server k ∈ {1, …, N} "
                              "+ a 'wait' no-op."),
                ("Reward Rₜ:", "−(α·active_power + β·SLA_violations)."),
                ("Mask:", "illegal placements pruned at selection AND target."),
               ],
               header_size=15, body_size=11)

    # Data panel
    _add_panel(s, Inches(9.6), Inches(2.9), Inches(3.2), Inches(4.2),
               "Data",
               [
                ("Synthetic:", "Poisson(0.8) arrivals, configurable cluster."),
                ("Google v2:", "real trace, pure replay & sampled-+-Poisson "
                              "variants — pure replay saturated all agents at "
                              "~99% SLA, so we use the sampled hybrid."),
                ("Heterogeneous fleet:", "3 server tiers (efficient / standard "
                                        "/ power-hungry), fixed cluster_seed for "
                                        "fair cross-agent comparison."),
                ("Eval protocol:", "50 train / 50 held-out test seeds; "
                                  "TEST_SEED_OFFSET = 10⁶."),
               ],
               header_size=15, body_size=11)


def slide_implementation(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "System\nImplementation", "what was actually built")

    # Stack panel
    _add_panel(s, Inches(6.4), Inches(0.5), Inches(6.4), Inches(2.0),
               "Stack",
               [
                ("Backend:", "FastAPI + ThreadPoolExecutor + asyncio MetricsBus, "
                            "SQLite store with cascade-delete on episodes & "
                            "cmdp_iterations tables."),
                ("Frontend:", "React + Vite + Recharts SPA — Train, Live "
                             "Monitor, Results, Compare, plus dedicated CMDP "
                             "Training & Results pages."),
                ("Live metrics:", "WebSocket streams per-episode (online) and "
                                "per-FQI-iteration (offline) events into the UI."),
               ],
               header_size=15, body_size=11)

    # Agents panel
    _add_panel(s, Inches(6.4), Inches(2.7), Inches(6.4), Inches(2.0),
               "Agents",
               [
                ("Heuristics:", "Round-Robin, Shortest-Job-First, "
                               "First-Fit-Decreasing — non-learning baselines."),
                ("Online RL:", "Double-DQN with action mask; PPO with GAE-λ, "
                              "clipped surrogate, masked Categorical, advantage "
                              "whitening guarded against std≈0."),
                ("Agentic RL (Approach 1):", "SLA sub-agent + Power sub-agent "
                              "(both DQN on decomposed rewards) + REINFORCE "
                              "supervisor that picks which sub-agent to follow."),
                ("Offline / CMDP (Approach 2):", "FQI on Q_r and Q_c with "
                              "Lagrangian dual λ; CQL-style conservative "
                              "penalty on Q_c; tunable behaviour-policy mix."),
               ],
               header_size=15, body_size=11)

    # Reward shaping
    _add_panel(s, Inches(6.4), Inches(4.85), Inches(6.4), Inches(2.3),
               "Reward shaping decisions that mattered",
               [
                "Active power (above idle floor) instead of total power → "
                "optimal-reward ceiling = 0, gap-from-optimal directly "
                "interpretable.",
                "Continuous SLA queue-pressure term in reward — provides "
                "credit assignment for actions that *will* cause a violation, "
                "not just the one that does.",
                "Single-source config (environment/config.py) auto-propagates "
                "α, β, ε_sla, episode_length, etc. to schema, frontend defaults, "
                "and all workload generators via /api/config/defaults.",
               ],
               header_size=15, body_size=11)


def slide_methods(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "Two\nApproaches", "as proposed, both implemented")

    # Approach 1
    _add_panel(s, Inches(6.4), Inches(0.5), Inches(3.1), Inches(6.6),
               "Approach 1 — Agentic RL",
               [
                ("SLA sub-agent:", "Double-DQN on r_sla = −step_violations."),
                ("Power sub-agent:", "Double-DQN on r_power = −step_power_norm."),
                ("Supervisor:", "small MLP → P(follow SLA agent); trained with "
                              "REINFORCE on the *combined* env reward."),
                ("State for supervisor:", "[remaining SLA budget, latest power "
                                        "level, time-in-episode] — small and "
                                        "interpretable."),
                ("Why this:", "decomposes the multi-objective tradeoff so each "
                             "sub-agent has a clean reward signal; the supervisor "
                             "learns when to prefer power vs SLA."),
               ],
               header_size=14, body_size=11)

    # Approach 2
    _add_panel(s, Inches(9.6), Inches(0.5), Inches(3.2), Inches(6.6),
               "Approach 2 — Offline CMDP",
               [
                ("Two critics:", "Q_r learns reward (−power_norm), Q_c learns "
                                "the SLA cost — Bellman targets via Double-DQN-"
                                "style FQI on a static Parquet dataset."),
                ("Lagrangian:", "policy = argmax_a (Q_r − λ·Q_c); λ updated by "
                              "gradient ascent on E[Q_c(s,π(s))] − ε."),
                ("Cost rescaling:", "step cost ∈ {0,1} × (1−γ) so Q_c ∈ [0,1] "
                                  "is directly comparable to ε."),
                ("CQL on Q_c:", "logsumexp penalty pushes Q_c UP on OOD actions "
                              "→ pessimistic constraint critic, fights offline-RL "
                              "extrapolation error."),
                ("Behaviour mix:", "RR/SJF/FFD weights + uniform-random injection "
                                  "are tunable per run from the UI."),
               ],
               header_size=14, body_size=11)


def slide_training(prs, results):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "Training\nProtocol", "headline sweep")

    # Setup panel
    _add_panel(s, Inches(6.4), Inches(0.5), Inches(6.4), Inches(2.4),
               "Headline sweep configuration",
               [
                ("Cluster:", "N = 50 servers, heterogeneous fleet (3 power tiers)."),
                ("Episode:", "1000 steps, α = 1.0 (power), β = 50.0 (SLA)."),
                ("Seeds:", "50-seed train pool + 50-seed held-out test pool. "
                          "Test offset = 10⁶ → no overlap."),
                ("Budgets:", "DQN/Agentic = 2000 episodes; PPO = 2 × 10⁵ env steps; "
                            "heuristics = 1 episode (non-learning). Eval = full "
                            "test pool (greedy)."),
               ],
               header_size=15, body_size=11)

    # What we measured
    _add_panel(s, Inches(6.4), Inches(3.1), Inches(6.4), Inches(4.0),
               "What we measure on each run",
               [
                ("mean_reward (eval):", "average held-out-test reward — primary "
                                       "headline number."),
                ("mean_power:", "total cluster power (W·steps) per episode."),
                ("sla_violation_rate:", "Σ violations / Σ jobs_completed across "
                                      "the eval pool."),
                ("gap_pct:", "(optimal − last10) / |optimal| × 100. With active-"
                            "power reward, optimal = 0, so this is just |last10| "
                            "normalised by episode length."),
                ("Live metrics:", "every per-episode and per-FQI-iteration "
                                 "event streamed to the React UI via WebSocket "
                                 "and persisted to SQLite for retrospective "
                                 "comparison."),
               ],
               header_size=15, body_size=11)


_AGENT_ORDER = ["sjf", "ffd", "round_robin", "ppo", "agentic", "dqn"]
_AGENT_PRETTY = {"sjf": "SJF", "ffd": "FFD", "round_robin": "RoundRobin",
                 "ppo": "PPO", "agentic": "Agentic-RL", "dqn": "DQN",
                 "cmdp": "CMDP"}


def _add_results_table(slide, results, x, y, w, h):
    rows = []
    for a in _AGENT_ORDER:
        if a not in results:
            continue
        _, _sla, _cfg, ev, _summ = results[a]
        rows.append([
            _AGENT_PRETTY[a],
            f"{ev.get('mean_reward', 0):.1f}",
            f"{ev.get('mean_power', 0):,.0f}",
            f"{ev.get('sla_violation_rate', 0)*100:.2f}%",
            f"{ev.get('mean_jobs_completed', 0):.0f}",
        ])
    headers = ["Agent", "mean R (eval)", "mean power", "SLA rate", "jobs/ep"]
    rows_total = max(2, len(rows) + 1)
    tbl_shape = slide.shapes.add_table(rows_total, len(headers), x, y, w, h)
    tbl = tbl_shape.table
    for j, hdr in enumerate(headers):
        cell = tbl.cell(0, j); cell.text = ""
        run = cell.text_frame.paragraphs[0].add_run()
        run.text = hdr
        run.font.bold = True; run.font.size = Pt(13)
        run.font.color.rgb = INK; run.font.name = "Cambria"
        cell.fill.solid(); cell.fill.fore_color.rgb = BG_DARK
    for i, row in enumerate(rows):
        for j, val in enumerate(row):
            cell = tbl.cell(i + 1, j); cell.text = ""
            run = cell.text_frame.paragraphs[0].add_run()
            run.text = val
            run.font.size = Pt(12); run.font.color.rgb = INK
            run.font.name = "Consolas" if j > 0 else "Calibri"
            cell.fill.solid()
            cell.fill.fore_color.rgb = PANEL if i % 2 == 0 else BG_DARK


def slide_results_synth(prs, synth):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "Synthetic\nResults", "Poisson · N=50 · het · 2000 ep")

    _add_results_table(s, synth, Inches(6.4), Inches(0.7),
                       Inches(6.4), Inches(3.6))

    sjf_sla = synth.get("sjf", (None, None, None, {}, None))[3].get("sla_violation_rate", 0) * 100
    ppo_sla = synth.get("ppo", (None, None, None, {}, None))[3].get("sla_violation_rate", 0) * 100
    sjf_r = synth.get("sjf", (None, None, None, {}, None))[3].get("mean_reward", 0)
    ppo_r = synth.get("ppo", (None, None, None, {}, None))[3].get("mean_reward", 0)

    _add_panel(s, Inches(6.4), Inches(4.5), Inches(6.4), Inches(2.6),
               "What the numbers say (synthetic)",
               [
                f"SJF leads every learned agent here: "
                f"SLA = {sjf_sla:.2f}% vs PPO {ppo_sla:.2f}%, "
                f"mean R = {sjf_r:.0f} vs {ppo_r:.0f}.",
                "PPO is the strongest learner — clearly beats RoundRobin and "
                "DQN on both axes, sits just behind SJF.",
                "DQN underperforms on the K·N+1 joint action space — "
                "consistent with the well-known Q-learning failure modes on "
                "large discrete action spaces.",
                "Clean Poisson arrivals expose little of RL's advantage: "
                "the workload is too easy for the simple heuristic to lose.",
               ],
               header_size=15, body_size=11)


def slide_results_gv2(prs, gv2):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "Google v2\nResults", "sampled + Poisson · N=50 · het · 2000 ep")

    _add_results_table(s, gv2, Inches(6.4), Inches(0.7),
                       Inches(6.4), Inches(3.6))

    # Build a comparative takeaway using actual numbers
    if gv2:
        slas = sorted(
            ((a, gv2[a][3].get("sla_violation_rate", 0) * 100) for a in gv2),
            key=lambda kv: kv[1],
        )
        best_agent, best_sla = slas[0]
        worst_agent, worst_sla = slas[-1]
    else:
        best_agent = worst_agent = "—"
        best_sla = worst_sla = 0.0

    _add_panel(s, Inches(6.4), Inches(4.5), Inches(6.4), Inches(2.6),
               "What the numbers say (Google v2)",
               [
                "Trace-distribution job specs (CPU / mem / duration sampled "
                "from the Google v2 trace) overlaid on synthetic Poisson "
                "timing — preserves the real workload's heaviness without "
                "the saturation of pure replay.",
                f"All agents land in a tight band — best {_AGENT_PRETTY[best_agent]} at "
                f"{best_sla:.2f}% SLA, worst {_AGENT_PRETTY[worst_agent]} at "
                f"{worst_sla:.2f}%. The realistic distribution is harder than "
                f"clean Poisson for every method.",
                "RL agents (PPO, Agentic, DQN) cluster around the heuristics "
                "rather than dominating — consistent with the synthetic-vs-"
                "Google v2 gap also widening for the heuristics. "
                "Distribution shift between training and eval is doing real "
                "work here.",
               ],
               header_size=15, body_size=11)


def slide_offline_detour(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "Offline RL\nfindings", "Approach 2: a careful negative result")

    _add_panel(s, Inches(6.4), Inches(0.5), Inches(6.4), Inches(2.4),
               "What we built",
               [
                "Full CMDP pipeline: dataset generation, FQI on Q_r/Q_c, dual "
                "λ updates, CQL on Q_c, eval against the same held-out seed pool "
                "as the online agents.",
                "Surfaced as its own page in the UI with live FQI loss curves, "
                "λ trajectory, constraint-violation chart, slack diagnostic, "
                "and behaviour-mix sliders.",
               ],
               header_size=15, body_size=11)

    _add_panel(s, Inches(6.4), Inches(3.05), Inches(6.4), Inches(2.0),
               "What broke",
               [
                "Vanilla FQI: Q_c falsely predicted ~0 cost on OOD actions → "
                "policy walked off-distribution, got 13% SLA while believing "
                "it was at 0%.",
                "CQL fixed the OOD belief → policy stayed in-distribution → "
                "but the dataset's behaviour policies *themselves* averaged "
                "~14% SLA, so policy regressed to the dataset, not to ε.",
                "Conclusion: the offline-RL ceiling is the demonstrator's "
                "ceiling — Lagrangian relaxation alone cannot deliver hard "
                "SLA guarantees.",
               ],
               header_size=15, body_size=11)

    _add_panel(s, Inches(6.4), Inches(5.2), Inches(6.4), Inches(1.95),
               "What this points to (future work)",
               [
                "Online safe RL (PPO-Lagrangian) — dual variable updates from "
                "real env violations, not frozen logged costs.",
                "Action shielding: hard runtime guarantee by overriding the "
                "policy when the empirical violation budget is depleted.",
                "Constrained Policy Optimisation (CPO) with hard rejection "
                "via a calibrated safety critic.",
               ],
               header_size=15, body_size=11)


def slide_conclusion(prs, synth, gv2):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    _set_bg(s); _add_left_ellipse(s)
    _add_title(s, "Conclusion", "what we learned, and what's next")

    # Build a comparative takeaway using actual numbers from both buckets.
    extra_lines = []
    if gv2 and synth:
        gv2_slas = [(a, gv2[a][3].get("sla_violation_rate", 0) * 100)
                    for a in gv2]
        synth_slas = [(a, synth[a][3].get("sla_violation_rate", 0) * 100)
                      for a in synth]
        gv2_avg = sum(v for _, v in gv2_slas) / max(1, len(gv2_slas))
        synth_avg = sum(v for _, v in synth_slas) / max(1, len(synth_slas))
        extra_lines.append(
            f"Across-workload finding: SLA rates jump from "
            f"~{synth_avg:.1f}% on synthetic Poisson to ~{gv2_avg:.1f}% on "
            f"Google v2 (sampled) — every method, learned or heuristic, gets "
            f"hit by the realistic workload's heavier tails."
        )
        extra_lines.append(
            "The headline gap between RL and SJF narrows on Google v2: when "
            "the workload departs from clean Poisson, the simple-heuristic "
            "advantage shrinks and learned policies become competitive."
        )

    _add_panel(s, Inches(6.4), Inches(0.5), Inches(6.4), Inches(2.4),
               "Headline takeaways",
               [
                "Both proposed approaches are implemented end-to-end and "
                "exercised through a single FastAPI + React UI.",
                "On a clean synthetic workload, SJF is a stronger baseline "
                "than the proposal anticipated — PPO closes the gap but "
                "doesn't beat it. Agentic-RL tracks closely with PPO at the "
                "cost of more moving parts.",
                "DQN is the weakest learner — its policy collapses on this "
                "K·N+1 joint action space, a known failure mode rather than "
                "a tuning issue.",
               ] + extra_lines,
               header_size=15, body_size=11)

    _add_panel(s, Inches(6.4), Inches(3.05), Inches(6.4), Inches(2.0),
               "The CMDP story",
               [
                "Approach 2 ran into the well-documented offline-RL "
                "extrapolation/feasibility wall: hard SLA guarantees require "
                "either online safe-RL (so the dual variable can react to "
                "real violations) or a hard runtime shield.",
                "Implemented and shipped as a separate flow in the UI so the "
                "negative result is reproducible from the frontend with one "
                "click.",
               ],
               header_size=15, body_size=11)

    _add_panel(s, Inches(6.4), Inches(5.2), Inches(6.4), Inches(1.95),
               "Next steps",
               [
                "Action-shielding wrapper around the trained policy → "
                "guaranteed per-episode violation cap.",
                "PPO-Lagrangian in the online loop to test the soft-"
                "constraint version of the same idea, end-to-end.",
                "Bigger / non-stationary workloads (real Google v2 burstiness "
                "instead of Poisson) — we expect the gap between RL and SJF "
                "to grow exactly where SJF's myopic policy fails.",
               ],
               header_size=15, body_size=11)


def main() -> int:
    out_path = Path(__file__).resolve().parent.parent / "Final_Presentation.pptx"

    prs = Presentation()
    prs.slide_width = Inches(13.333)
    prs.slide_height = Inches(7.5)

    synth, gv2 = collect_results_split()

    slide_title(prs)
    slide_proposal_recap(prs)
    slide_implementation(prs)
    slide_methods(prs)
    slide_training(prs, synth)
    slide_results_synth(prs, synth)
    slide_results_gv2(prs, gv2)
    slide_offline_detour(prs)
    slide_conclusion(prs, synth, gv2)

    prs.save(out_path)
    print(f"Wrote {out_path}  ({len(prs.slides)} slides)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
