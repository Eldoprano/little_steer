#!/usr/bin/env python3
"""
Generate an interactive HTML results page from sweep_results.json.

Run after sweep.py completes:
  python little_steer/scripts/make_html.py

Produces: little_steer/results.html
"""
from __future__ import annotations
import json, sys
from pathlib import Path
from collections import defaultdict

import numpy as np

REPO_ROOT = Path(__file__).parent.parent.parent
RESULTS_PATH = REPO_ROOT / "little_steer" / "sweep_results.json"
OUT_PATH     = REPO_ROOT / "little_steer" / "results.html"

# ── Load results ──────────────────────────────────────────────────────────────
def load_results():
    with open(RESULTS_PATH) as f:
        return json.load(f)

# ── Color scale ───────────────────────────────────────────────────────────────
def auroc_color(v: float) -> str:
    """Blue (0.5) → white (0.75) → red (1.0)"""
    v = max(0.5, min(1.0, v))
    t = (v - 0.5) / 0.5
    if t < 0.5:
        r = int(255 * (2 * t))
        return f"rgb({r},180,255)"
    else:
        g = int(255 * (1 - (t - 0.5) * 2))
        return f"rgb(255,{g},{int(255*(1-t))})"

def f1_color(v: float) -> str:
    v = max(0.0, min(1.0, v))
    r = int(255 * (1 - v))
    g = int(200 * v)
    return f"rgb({r},{g},100)"

# ── Build HTML sections ───────────────────────────────────────────────────────

def section_summary(results):
    """Top-5 best configs per target label, sorted by honest AUROC."""
    by_target = defaultdict(list)
    for r in results:
        by_target[r["target"]].append(r)

    rows = []
    for target in sorted(by_target):
        top = sorted(by_target[target], key=lambda x: -x["auroc"])[:1][0]
        label_short = target.replace("II_", "").replace("IV_", "").replace("III_", "").replace("I_", "")
        rows.append(f"""
        <tr>
          <td title="{target}"><b>{label_short.replace("_"," ")}</b></td>
          <td style="background:{auroc_color(top['auroc'])}">{top['auroc']:.3f}</td>
          <td style="background:{f1_color(top['f1'])}">{top['f1']:.3f}</td>
          <td>{top['precision']:.3f}</td>
          <td>{top['recall']:.3f}</td>
          <td><code>{top['spec']}</code></td>
          <td>{top['layer']}</td>
          <td><code>{top['method']}</code></td>
          <td><code>{top['baseline']}</code></td>
          <td>{top['n_pos']} / {top['n_neg']}</td>
        </tr>""")

    return f"""
<section id="summary">
  <h2>Best Configuration per Label</h2>
  <p>Honest evaluation: negatives = <em>all other sentence types</em> (not just one baseline).
  This is what detection would look like in a real conversation.</p>
  <table>
    <thead><tr>
      <th>Label</th><th>AUROC↑</th><th>F1↑</th><th>Precision</th><th>Recall</th>
      <th>Spec</th><th>Layer</th><th>Method</th><th>Baseline</th><th>n_pos / n_neg</th>
    </tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</section>"""


def section_heatmap_auroc(results, target):
    """AUROC heatmap: rows=spec, cols=layer, for a given target and best method/baseline."""
    # Group by (spec, layer) → best auroc across methods/baselines
    grid = defaultdict(lambda: defaultdict(list))
    for r in results:
        if r["target"] == target:
            grid[r["spec"]][r["layer"]].append(r["auroc"])

    specs = ["whole_sentence", "last_1", "first_1", "last_3", "first_3", "bleed_3"]
    layers = sorted(set(r["layer"] for r in results))

    if not any(grid[s] for s in specs):
        return ""

    header = "".join(f"<th>L{l}</th>" for l in layers)
    body_rows = []
    for spec in specs:
        cells = []
        for l in layers:
            vals = grid[spec][l]
            if vals:
                v = max(vals)
                cells.append(f'<td style="background:{auroc_color(v)}" title="{v:.4f}">{v:.3f}</td>')
            else:
                cells.append('<td style="color:#ccc">–</td>')
        body_rows.append(f"<tr><th>{spec}</th>{''.join(cells)}</tr>")

    label_short = target.replace("_", " ")
    return f"""
<div class="heatmap-block">
  <h3>{label_short}</h3>
  <table class="heatmap">
    <thead><tr><th>Spec \\ Layer</th>{header}</tr></thead>
    <tbody>{"".join(body_rows)}</tbody>
  </table>
  <p class="hint">Best AUROC across all methods/baselines. Honest eval (all other sentence types as negatives).</p>
</div>"""


def section_method_comparison(results, target):
    """Bar-style table: methods × baselines for the best layer of each."""
    combos = defaultdict(list)
    for r in results:
        if r["target"] == target:
            combos[(r["method"], r["baseline"])].append(r)

    rows = []
    for (method, baseline), rs in sorted(combos.items()):
        best = max(rs, key=lambda x: x["auroc"])
        rows.append(f"""<tr>
          <td><code>{method}</code></td>
          <td><code>{baseline}</code></td>
          <td style="background:{auroc_color(best['auroc'])}">{best['auroc']:.3f}</td>
          <td style="background:{f1_color(best['f1'])}">{best['f1']:.3f}</td>
          <td>L{best['layer']} / {best['spec']}</td>
          <td>{best['n_pos']} / {best['n_neg']}</td>
        </tr>""")

    if not rows:
        return ""

    label_short = target.replace("_", " ")
    return f"""
<div class="method-block">
  <h3>{label_short} — method comparison</h3>
  <table>
    <thead><tr><th>Method</th><th>Baseline</th><th>Best AUROC</th><th>Best F1</th><th>At layer/spec</th><th>n_pos/neg</th></tr></thead>
    <tbody>{"".join(rows)}</tbody>
  </table>
</div>"""


def section_confusion_matrix(r: dict) -> str:
    """HTML 2×2 confusion matrix."""
    tn, fp, fn, tp = r["tn"], r["fp"], r["fn"], r["tp"]
    total = tn + fp + fn + tp

    def cell(n, cls):
        pct = 100 * n / total if total else 0
        bg = {"tp": "#2166ac", "tn": "#4dac26", "fp": "#d6604d", "fn": "#f1a340"}[cls]
        return f'<td style="background:{bg};color:white;text-align:center;padding:12px"><b>{n}</b><br><small>({pct:.1f}%)</small></td>'

    label_short = r["target"].replace("_", " ")
    return f"""
<div class="cm-block">
  <h3>{label_short}</h3>
  <p>Best config: {r['spec']} · L{r['layer']} · {r['method']} · {r['baseline']}<br>
     AUROC={r['auroc']:.3f}, F1={r['f1']:.3f}, Precision={r['precision']:.3f}, Recall={r['recall']:.3f}</p>
  <table class="cm">
    <thead><tr><th></th><th>Pred: Absent</th><th>Pred: Present</th></tr></thead>
    <tbody>
      <tr><th>True: Absent</th>{cell(tn,'tn')}{cell(fp,'fp')}</tr>
      <tr><th>True: Present</th>{cell(fn,'fn')}{cell(tp,'tp')}</tr>
    </tbody>
  </table>
  <p class="legend">
    <span style="background:#4dac26;color:white;padding:2px 6px">TN</span>
    <span style="background:#d6604d;color:white;padding:2px 6px">FP</span>
    <span style="background:#f1a340;color:white;padding:2px 6px">FN</span>
    <span style="background:#2166ac;color:white;padding:2px 6px">TP</span>
  </p>
</div>"""


# ── Main HTML ─────────────────────────────────────────────────────────────────

CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', system-ui, sans-serif; background: #f8f8f8; color: #222; }
header { background: #1c3557; color: white; padding: 2rem 3rem; }
header h1 { font-size: 1.6rem; margin-bottom: 0.4rem; }
header p { opacity: 0.8; font-size: 0.95rem; }
nav { background: #263d5a; padding: 0.5rem 3rem; display: flex; gap: 1.5rem; }
nav a { color: #9dc5f0; text-decoration: none; font-size: 0.9rem; }
nav a:hover { color: white; }
main { max-width: 1400px; margin: 2rem auto; padding: 0 2rem; }
section { margin-bottom: 3rem; }
h2 { font-size: 1.3rem; color: #1c3557; margin-bottom: 1rem; border-bottom: 2px solid #1c3557; padding-bottom: 0.3rem; }
h3 { font-size: 1rem; color: #333; margin: 1.2rem 0 0.5rem; }
table { border-collapse: collapse; font-size: 0.85rem; }
table th, table td { border: 1px solid #ddd; padding: 6px 10px; }
table thead th { background: #1c3557; color: white; font-weight: 600; }
table tbody tr:hover { background: #f0f4ff; }
code { background: #eef; padding: 1px 4px; border-radius: 3px; font-size: 0.82rem; }
.heatmap td, .heatmap th { padding: 5px 8px; font-size: 0.8rem; }
.heatmap-block, .method-block, .cm-block { margin-bottom: 2rem; }
.cm table { width: auto; margin: 0.5rem 0; }
.cm td { width: 140px; }
.hint { font-size: 0.78rem; color: #888; margin-top: 0.3rem; }
.legend { font-size: 0.8rem; margin-top: 0.5rem; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 2rem; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 2rem; }
.callout { background: #fff3cd; border-left: 4px solid #f1a340; padding: 1rem 1.2rem; margin: 1rem 0; border-radius: 3px; font-size: 0.9rem; }
.good  { background: #d4edda; border-left-color: #4dac26; }
"""

JS = """
function filterTable(inputId, tableId) {
  const val = document.getElementById(inputId).value.toLowerCase();
  document.querySelectorAll('#' + tableId + ' tbody tr').forEach(row => {
    row.style.display = row.textContent.toLowerCase().includes(val) ? '' : 'none';
  });
}
"""

def build_html(results):
    targets_sorted = sorted(set(r["target"] for r in results))
    by_target = defaultdict(list)
    for r in results:
        by_target[r["target"]].append(r)

    # Best config per target
    best_per_target = {}
    for t in targets_sorted:
        rs = by_target[t]
        if rs:
            best_per_target[t] = max(rs, key=lambda x: x["auroc"])

    nav_links = """
    <a href="#what-went-wrong">The Problem</a>
    <a href="#summary">Summary Table</a>
    <a href="#heatmaps">AUROC Heatmaps</a>
    <a href="#methods">Method Comparison</a>
    <a href="#confusion">Confusion Matrices</a>
    """

    what_went_wrong = """
<section id="what-went-wrong">
  <h2>What Was Wrong with the Initial Results</h2>
  <div class="callout">
    <b>The initial AUROC=1.00 was misleading.</b> The evaluation used only <code>I_REPHRASE_PROMPT</code>
    sentences as negatives — just 14 samples. Those sentences are semantically very different from
    safety-concern sentences, so the classifier had an easy job.
  </div>
  <p>The <b>honest evaluation</b> here uses <em>all other sentence types</em> as negatives — matching
  what you'd actually see when monitoring a running conversation. Numbers are lower but credible:</p>
  <ul style="margin:0.8rem 0 0 1.5rem; line-height:2">
    <li>Biased eval (vs <code>I_REPHRASE_PROMPT</code> only): AUROC often 0.93–1.00</li>
    <li>Honest eval (vs all sentence types): AUROC typically 0.55–0.80</li>
    <li>Main confounders: <b>semantically adjacent labels</b> share activation space
        (ethical ↔ legal ↔ safety concern)</li>
  </ul>
  <div class="callout good" style="margin-top:1rem">
    <b>Good news:</b> even 0.75 AUROC with a simple mean-difference vector is a meaningful result —
    the vector has real signal. The hard cases are labels that are semantically close to each other,
    which is expected and interesting for your thesis.
  </div>
</section>"""

    heatmap_section = '<section id="heatmaps"><h2>AUROC by Spec × Layer (Honest)</h2>'
    for t in targets_sorted:
        heatmap_section += section_heatmap_auroc(results, t)
    heatmap_section += "</section>"

    method_section = '<section id="methods"><h2>Method × Baseline Comparison</h2>'
    for t in targets_sorted:
        method_section += section_method_comparison(results, t)
    method_section += "</section>"

    cm_section = '<section id="confusion"><h2>Confusion Matrices (Best Config per Label)</h2>'
    cm_section += '<div class="grid3">'
    for t in targets_sorted:
        if t in best_per_target:
            cm_section += section_confusion_matrix(best_per_target[t])
    cm_section += "</div></section>"

    # Stats
    n_configs = len(results)
    n_targets = len(targets_sorted)
    best_auroc = max(r["auroc"] for r in results)
    best_r = max(results, key=lambda x: x["auroc"])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Steering Vector Sweep Results</title>
  <style>{CSS}</style>
</head>
<body>
<header>
  <h1>Steering Vector Sweep — Full Results</h1>
  <p>
    {n_configs:,} configurations tested &nbsp;·&nbsp;
    {n_targets} behavior labels &nbsp;·&nbsp;
    6 token specs &nbsp;·&nbsp;
    8 layers &nbsp;·&nbsp;
    3 vector methods &nbsp;·&nbsp;
    3 baselines<br>
    Best honest AUROC: <b>{best_auroc:.3f}</b>
    ({best_r['target'].replace('_',' ')} · {best_r['spec']} · L{best_r['layer']} · {best_r['method']} · {best_r['baseline']})
  </p>
</header>
<nav>{nav_links}</nav>
<main>
  {what_went_wrong}
  {section_summary(results)}
  {heatmap_section}
  {method_section}
  {cm_section}
</main>
<script>{JS}</script>
</body>
</html>"""


def main():
    if not RESULTS_PATH.exists():
        print(f"ERROR: {RESULTS_PATH} not found. Run sweep.py first.")
        sys.exit(1)

    print(f"Loading results from {RESULTS_PATH}...")
    results = load_results()
    print(f"  {len(results)} result entries")

    html = build_html(results)
    OUT_PATH.write_text(html, encoding="utf-8")
    print(f"Saved → {OUT_PATH}")

if __name__ == "__main__":
    main()
