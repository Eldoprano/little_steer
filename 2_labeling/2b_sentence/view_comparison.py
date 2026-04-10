#!/usr/bin/env python3
"""view_comparison.py — Web viewer for judge comparison JSON files.

Shows each entry with judge annotations side-by-side, color-coded by label
group, with agreement highlighting across judges.

Usage:
    uv run view_comparison.py data/labeler_comparison/comparison_seed42_*.json
    uv run view_comparison.py  # auto-picks the latest comparison file
    uv run view_comparison.py --port 5050
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click
from flask import Flask, render_template_string

_HERE = Path(__file__).parent
DEFAULT_DATA_DIR = _HERE / "data" / "labeler_comparison"

app = Flask(__name__)
_data: dict = {}
_json_path: Path | None = None

# ── Label group colors ─────────────────────────────────────────────────────────

LABEL_COLORS: dict[str, str] = {
    "I":   "#3b82f6",  # blue
    "II":  "#f97316",  # orange
    "III": "#eab308",  # amber
    "IV":  "#ef4444",  # red
    "V":   "#a855f7",  # purple
    "VI":  "#14b8a6",  # teal
    "VII": "#6b7280",  # gray
}

TRAJECTORY_COLORS: dict[str, str] = {
    "safe_throughout":       "#22c55e",
    "concern_then_refuse":   "#3b82f6",
    "concern_then_comply":   "#f97316",
    "comply_no_deliberation":"#ef4444",
    "gradual_escalation":    "#dc2626",
    "mixed_inconclusive":    "#6b7280",
}

ALIGNMENT_COLORS: dict[str, str] = {
    "aligned":      "#22c55e",
    "partial":      "#f97316",
    "contradicted": "#ef4444",
}


def _label_group(label: str) -> str:
    """Return Roman numeral group prefix from label like 'II_FLAG_PROMPT_AS_HARMFUL'."""
    return label.split("_")[0] if "_" in label else "VII"


def _label_color(label: str) -> str:
    return LABEL_COLORS.get(_label_group(label), "#6b7280")


def _agreement_class(labels_by_judge: dict, entry_idx: int, sent_idx: int) -> str:
    """Compare primary labels at sent_idx across all judges. Returns 'agree', 'partial', 'disagree'."""
    primary_labels = []
    for judge_data in labels_by_judge.values():
        if "error" in judge_data:
            continue
        sents = judge_data.get("sentences", [])
        if sent_idx < len(sents):
            labels = sents[sent_idx].get("labels", [])
            if labels:
                primary_labels.append(labels[0])

    if len(primary_labels) < 2:
        return "unknown"
    if len(set(primary_labels)) == 1:
        return "agree"
    # Check group agreement
    groups = [_label_group(l) for l in primary_labels]
    if len(set(groups)) == 1:
        return "partial"  # same group, different label
    return "disagree"


def _build_template_data(data: dict) -> dict:
    """Pre-process raw JSON into template-friendly structures."""
    judge_names = []
    for entry in data.get("entries", []):
        for name in entry.get("labels_by_judge", {}):
            if name not in judge_names:
                judge_names.append(name)

    entries = []
    for entry in data.get("entries", []):
        labels_by_judge = entry.get("labels_by_judge", {})

        # Max sentence count across judges
        max_sents = max(
            (len(jd.get("sentences", [])) for jd in labels_by_judge.values() if "error" not in jd),
            default=0,
        )

        # Per-sentence agreement info (align by index — judges may vary, that's OK)
        sentence_rows = []
        for i in range(max_sents):
            agr = _agreement_class(labels_by_judge, 0, i)
            cells = []
            for jname in judge_names:
                jd = labels_by_judge.get(jname, {})
                if "error" in jd:
                    cells.append({"error": jd["error"]})
                    continue
                sents = jd.get("sentences", [])
                if i < len(sents):
                    s = sents[i]
                    label_info = [
                        {"label": l, "color": _label_color(l), "group": _label_group(l)}
                        for l in s.get("labels", [])
                    ]
                    cells.append({
                        "text": s.get("text", ""),
                        "labels": label_info,
                        "safety_score": s.get("safety_score", 0),
                    })
                else:
                    cells.append(None)
            sentence_rows.append({"agreement": agr, "cells": cells})

        # Assessment per judge
        assessments = {}
        for jname in judge_names:
            jd = labels_by_judge.get(jname, {})
            if "error" in jd:
                assessments[jname] = {"error": jd["error"]}
            else:
                a = jd.get("assessment", {})
                assessments[jname] = {
                    "trajectory": a.get("trajectory", "?"),
                    "trajectory_color": TRAJECTORY_COLORS.get(a.get("trajectory", ""), "#6b7280"),
                    "alignment": a.get("alignment", "?"),
                    "alignment_color": ALIGNMENT_COLORS.get(a.get("alignment", ""), "#6b7280"),
                    "turning_point": a.get("turning_point", -1),
                }

        # Agreement summary for assessment
        trajectories = [a["trajectory"] for a in assessments.values() if "error" not in a and "trajectory" in a]
        alignments = [a["alignment"] for a in assessments.values() if "error" not in a and "alignment" in a]
        assessment_agree = {
            "trajectory": len(set(trajectories)) == 1 if trajectories else False,
            "alignment": len(set(alignments)) == 1 if alignments else False,
        }

        entries.append({
            "source_file": entry.get("source_file", ""),
            "model_name": entry.get("source_file", "").replace("_clear_harm.jsonl", ""),
            "entry_id": entry.get("entry_id", "")[:16],
            "user_prompt": entry.get("user_prompt", ""),
            "reasoning": entry.get("reasoning", ""),
            "model_response": entry.get("model_response", ""),
            "assessments": assessments,
            "assessment_agree": assessment_agree,
            "sentence_rows": sentence_rows,
        })

    return {
        "judge_names": judge_names,
        "entries": entries,
        "seed": data.get("seed"),
        "generated_at": data.get("generated_at", ""),
        "json_path": str(_json_path) if _json_path else "",
    }


# ── HTML template ──────────────────────────────────────────────────────────────

TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Judge Comparison Viewer</title>
<style>
  :root {
    --bg: #0f0f13;
    --surface: #1a1a24;
    --surface2: #22222f;
    --border: #2e2e3e;
    --text: #e2e2f0;
    --muted: #6b6b8a;
    --agree: rgba(34,197,94,0.12);
    --partial: rgba(234,179,8,0.12);
    --disagree: rgba(239,68,68,0.12);
    --agree-border: #22c55e;
    --partial-border: #eab308;
    --disagree-border: #ef4444;
  }
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: var(--bg); color: var(--text); font-family: 'Segoe UI', system-ui, sans-serif; font-size: 14px; }

  header {
    background: var(--surface);
    border-bottom: 1px solid var(--border);
    padding: 16px 24px;
    position: sticky; top: 0; z-index: 100;
    display: flex; align-items: center; gap: 16px;
  }
  header h1 { font-size: 18px; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; }

  .legend {
    display: flex; flex-wrap: wrap; gap: 6px; margin-left: auto;
  }
  .legend-item {
    display: flex; align-items: center; gap: 4px; font-size: 11px; color: var(--muted);
  }
  .legend-dot { width: 10px; height: 10px; border-radius: 50%; flex-shrink: 0; }

  main { padding: 24px; max-width: 1600px; margin: 0 auto; }

  .toc {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 16px;
    margin-bottom: 24px;
  }
  .toc h2 { font-size: 14px; margin-bottom: 10px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.05em; }
  .toc-list { list-style: none; display: flex; flex-direction: column; gap: 4px; }
  .toc-list a { color: var(--text); text-decoration: none; font-size: 13px; }
  .toc-list a:hover { color: #818cf8; }

  .entry {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: 8px;
    margin-bottom: 32px;
    overflow: hidden;
  }

  .entry-header {
    background: var(--surface2);
    padding: 14px 20px;
    border-bottom: 1px solid var(--border);
    display: flex; align-items: baseline; gap: 12px;
  }
  .entry-model { font-weight: 700; font-size: 16px; color: #818cf8; }
  .entry-id { font-family: monospace; font-size: 11px; color: var(--muted); }

  .section { padding: 16px 20px; border-bottom: 1px solid var(--border); }
  .section:last-child { border-bottom: none; }
  .section-label {
    font-size: 11px; font-weight: 600; text-transform: uppercase; letter-spacing: 0.08em;
    color: var(--muted); margin-bottom: 8px;
  }

  .collapsible-toggle {
    background: none; border: 1px solid var(--border); color: var(--muted);
    border-radius: 4px; padding: 2px 8px; font-size: 11px; cursor: pointer;
    margin-left: 8px;
  }
  .collapsible-toggle:hover { border-color: #818cf8; color: #818cf8; }
  .collapsible { display: none; }
  .collapsible.open { display: block; }

  .prompt-text {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 13px;
    line-height: 1.6;
    max-height: 200px;
    overflow-y: auto;
  }
  .reasoning-text {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 6px;
    padding: 10px 14px;
    white-space: pre-wrap;
    word-break: break-word;
    font-size: 12px;
    line-height: 1.6;
    color: #a0a0c0;
    max-height: 300px;
    overflow-y: auto;
  }

  /* Assessment comparison */
  .assessment-grid {
    display: grid;
    gap: 10px;
  }
  .assessment-row {
    display: flex; align-items: center; gap: 12px; flex-wrap: wrap;
  }
  .judge-name { font-size: 12px; font-weight: 600; color: var(--muted); min-width: 160px; }
  .badge {
    display: inline-block;
    border-radius: 4px;
    padding: 3px 8px;
    font-size: 11px;
    font-weight: 600;
    color: #fff;
    letter-spacing: 0.02em;
  }
  .badge-outline {
    display: inline-block;
    border-radius: 4px;
    padding: 2px 7px;
    font-size: 11px;
    font-weight: 600;
    border: 1.5px solid;
    letter-spacing: 0.02em;
  }
  .agree-icon { color: #22c55e; font-size: 16px; margin-left: 4px; }
  .disagree-icon { color: #ef4444; font-size: 16px; margin-left: 4px; }

  /* Sentence table */
  .sentence-table { width: 100%; border-collapse: collapse; table-layout: fixed; }
  .sentence-table th {
    background: var(--surface2);
    padding: 8px 12px;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: var(--muted);
    border-bottom: 1px solid var(--border);
    text-align: left;
  }
  .th-agr { width: 60px; }
  .sentence-table td {
    padding: 10px 12px;
    border-bottom: 1px solid var(--border);
    vertical-align: top;
  }
  .sentence-table tr:last-child td { border-bottom: none; }

  .row-agree   td:first-child { border-left: 3px solid var(--agree-border); }
  .row-partial td:first-child { border-left: 3px solid var(--partial-border); }
  .row-disagree td:first-child { border-left: 3px solid var(--disagree-border); }
  .row-unknown td:first-child { border-left: 3px solid transparent; }

  .row-agree   { background: var(--agree); }
  .row-partial { background: var(--partial); }
  .row-disagree { background: var(--disagree); }

  .agr-dot {
    width: 10px; height: 10px; border-radius: 50%; display: inline-block;
  }

  .sent-text {
    font-size: 12px; line-height: 1.6; color: var(--text);
    margin-bottom: 6px; white-space: pre-wrap; word-break: break-word;
  }
  .label-row { display: flex; flex-wrap: wrap; gap: 4px; align-items: center; }
  .label-chip {
    border-radius: 3px;
    padding: 1px 6px;
    font-size: 10px;
    font-weight: 700;
    color: #fff;
    letter-spacing: 0.02em;
  }
  .score-chip {
    font-size: 10px; font-weight: 700;
    padding: 1px 5px; border-radius: 3px;
    color: #fff;
    margin-left: auto;
  }
  .cell-empty { color: var(--muted); font-size: 11px; font-style: italic; }
  .cell-error { color: #ef4444; font-size: 11px; }
</style>
</head>
<body>

<header>
  <div>
    <h1>Judge Comparison Viewer</h1>
    <div class="meta">seed={{ seed }} &nbsp;·&nbsp; {{ generated_at[:19] }} &nbsp;·&nbsp; {{ entries|length }} entries &nbsp;·&nbsp; {{ judge_names|length }} judges</div>
  </div>
  <div class="legend">
    <div class="legend-item"><div class="legend-dot" style="background:#22c55e"></div> agree</div>
    <div class="legend-item"><div class="legend-dot" style="background:#eab308"></div> same group</div>
    <div class="legend-item"><div class="legend-dot" style="background:#ef4444"></div> disagree</div>
    {% for grp, color in label_groups %}
    <div class="legend-item"><div class="legend-dot" style="background:{{ color }}"></div> {{ grp }}</div>
    {% endfor %}
  </div>
</header>

<main>

<div class="toc">
  <h2>Entries</h2>
  <ul class="toc-list">
    {% for e in entries %}
    <li><a href="#entry-{{ loop.index0 }}">{{ e.model_name }} &nbsp;<span style="color:var(--muted);font-size:11px">{{ e.entry_id }}</span></a></li>
    {% endfor %}
  </ul>
</div>

{% for e in entries %}
<div class="entry" id="entry-{{ loop.index0 }}">

  <div class="entry-header">
    <span class="entry-model">{{ e.model_name }}</span>
    <span class="entry-id">{{ e.entry_id }}</span>
  </div>

  <!-- User prompt -->
  <div class="section">
    <div class="section-label">
      User Prompt
      <button class="collapsible-toggle" onclick="toggle(this)">show</button>
    </div>
    <div class="collapsible">
      <div class="prompt-text">{{ e.user_prompt }}</div>
    </div>
  </div>

  <!-- Reasoning -->
  {% if e.reasoning %}
  <div class="section">
    <div class="section-label">
      Reasoning (cropped)
      <button class="collapsible-toggle" onclick="toggle(this)">show</button>
    </div>
    <div class="collapsible">
      <div class="reasoning-text">{{ e.reasoning }}</div>
    </div>
  </div>
  {% endif %}

  <!-- Model response -->
  {% if e.model_response %}
  <div class="section">
    <div class="section-label">
      Model Response
      <button class="collapsible-toggle" onclick="toggle(this)">show</button>
    </div>
    <div class="collapsible">
      <div class="prompt-text">{{ e.model_response }}</div>
    </div>
  </div>
  {% endif %}

  <!-- Assessment comparison -->
  <div class="section">
    <div class="section-label">Assessment</div>
    <div class="assessment-grid">
      {% for jname in judge_names %}
      {% set a = e.assessments.get(jname, {}) %}
      <div class="assessment-row">
        <span class="judge-name">{{ jname }}</span>
        {% if a.get('error') %}
          <span style="color:#ef4444;font-size:12px">{{ a.error }}</span>
        {% else %}
          <span class="badge" style="background:{{ a.trajectory_color }}">{{ a.trajectory }}</span>
          <span class="badge-outline" style="color:{{ a.alignment_color }};border-color:{{ a.alignment_color }}">{{ a.alignment }}</span>
          {% if a.turning_point >= 0 %}
          <span style="font-size:11px;color:var(--muted)">tp={{ a.turning_point }}</span>
          {% endif %}
        {% endif %}
      </div>
      {% endfor %}
      <div style="font-size:11px;color:var(--muted);margin-top:4px">
        trajectory:
        {% if e.assessment_agree.trajectory %}
        <span style="color:#22c55e">✓ all agree</span>
        {% else %}
        <span style="color:#ef4444">✗ differ</span>
        {% endif %}
        &nbsp; alignment:
        {% if e.assessment_agree.alignment %}
        <span style="color:#22c55e">✓ all agree</span>
        {% else %}
        <span style="color:#ef4444">✗ differ</span>
        {% endif %}
      </div>
    </div>
  </div>

  <!-- Sentence annotations -->
  <div class="section">
    <div class="section-label">Sentence Annotations</div>
    {% if e.sentence_rows %}
    <table class="sentence-table">
      <thead>
        <tr>
          <th class="th-agr">Agr</th>
          {% for jname in judge_names %}
          <th>{{ jname }}</th>
          {% endfor %}
        </tr>
      </thead>
      <tbody>
        {% for row in e.sentence_rows %}
        <tr class="row-{{ row.agreement }}">
          <td>
            {% if row.agreement == 'agree' %}
              <div class="agr-dot" style="background:#22c55e" title="All judges agree"></div>
            {% elif row.agreement == 'partial' %}
              <div class="agr-dot" style="background:#eab308" title="Same group, different label"></div>
            {% elif row.agreement == 'disagree' %}
              <div class="agr-dot" style="background:#ef4444" title="Judges disagree"></div>
            {% else %}
              <div class="agr-dot" style="background:var(--border)"></div>
            {% endif %}
          </td>
          {% for cell in row.cells %}
          <td>
            {% if cell is none %}
              <span class="cell-empty">—</span>
            {% elif cell.get('error') %}
              <span class="cell-error">{{ cell.error }}</span>
            {% else %}
              <div class="sent-text">{{ cell.text }}</div>
              <div class="label-row">
                {% for li in cell.labels %}
                <span class="label-chip" style="background:{{ li.color }}" title="{{ li.label }}">{{ li.label.split('_', 1)[1] if '_' in li.label else li.label }}</span>
                {% endfor %}
                {% set score = cell.safety_score %}
                {% if score > 0 %}
                  <span class="score-chip" style="background:#22c55e;margin-left:auto">+{{ score }}</span>
                {% elif score < 0 %}
                  <span class="score-chip" style="background:#ef4444;margin-left:auto">{{ score }}</span>
                {% else %}
                  <span class="score-chip" style="background:#6b7280;margin-left:auto">0</span>
                {% endif %}
              </div>
            {% endif %}
          </td>
          {% endfor %}
        </tr>
        {% endfor %}
      </tbody>
    </table>
    {% else %}
      <div class="cell-empty">No sentence annotations available.</div>
    {% endif %}
  </div>

</div>
{% endfor %}

</main>

<script>
function toggle(btn) {
  const content = btn.parentElement.nextElementSibling;
  const isOpen = content.classList.contains('open');
  content.classList.toggle('open', !isOpen);
  btn.textContent = isOpen ? 'show' : 'hide';
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    tdata = _build_template_data(_data)
    # Build label group legend
    tdata["label_groups"] = [
        ("I: Prompt", "#3b82f6"),
        ("II: Safety", "#f97316"),
        ("III: Reframe", "#eab308"),
        ("IV: Intent", "#ef4444"),
        ("V: Knowledge", "#a855f7"),
        ("VI: Meta", "#14b8a6"),
        ("VII: Neutral", "#6b7280"),
    ]
    return render_template_string(TEMPLATE, **tdata)


# ── CLI ────────────────────────────────────────────────────────────────────────

@click.command()
@click.argument("json_files", nargs=-1, type=click.Path(exists=True))
@click.option("--port", default=5001, show_default=True, help="Port to serve on.")
@click.option("--host", default="127.0.0.1", show_default=True, help="Host to bind.")
def main(json_files: tuple[str, ...], port: int, host: str) -> None:
    """Serve a web viewer for judge comparison JSON files.

    If multiple files are provided, they are merged by entry_id.
    If no file is given, auto-picks the most recent one.
    """
    global _data, _json_path

    if not json_files:
        candidates = sorted(DEFAULT_DATA_DIR.glob("comparison_*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if not candidates:
            click.echo(f"No comparison JSON files found in {DEFAULT_DATA_DIR}")
            sys.exit(1)
        json_files = (str(candidates[0]),)
        click.echo(f"Auto-selected latest: {json_files[0]}")

    merged = None
    for f in json_files:
        path = Path(f)
        d = json.loads(path.read_text())
        
        if merged is None:
            merged = d
            _json_path = path
        else:
            # Merge logic
            master_entries = {e["entry_id"]: e for e in merged.get("entries", [])}
            for other_entry in d.get("entries", []):
                eid = other_entry["entry_id"]
                if eid in master_entries:
                    # VALIDATION CHECK: Ensure text content matches
                    if other_entry.get("user_prompt") != master_entries[eid].get("user_prompt"):
                        click.echo(f"WARNING: Prompt mismatch for ID {eid} between files! Are these the same entries?")
                    
                    # Merge labels from this file into the master entry
                    master_entries[eid]["labels_by_judge"].update(other_entry.get("labels_by_judge", {}))
                else:
                    # Entry only exists in the new file, add it
                    merged["entries"].append(other_entry)
            
            if d.get("seed") != merged.get("seed"):
                click.echo(f"Notice: Merging files with different seeds ({merged.get('seed')} vs {d.get('seed')})")

    _data = merged
    click.echo(f"Loaded and merged {len(json_files)} files. Total entries: {len(_data.get('entries', []))}")
    click.echo(f"Open: http://{host}:{port}/")
    app.run(host=host, port=port, debug=False)


if __name__ == "__main__":
    main()
