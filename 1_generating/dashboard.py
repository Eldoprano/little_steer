#!/usr/bin/env python3
"""Dashboard to visualize sample generation progress."""

from __future__ import annotations

import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import os

from flask import Flask, render_template, jsonify
from rich.console import Console

app = Flask(__name__)
console = Console()

# Configuration
DATA_DIR = Path(__file__).parent.parent / "data" / "1_generated"


def load_stats() -> dict:
    """Load and aggregate statistics from all JSONL files."""
    stats = {
        "total_samples": 0,
        "approved": 0,
        "unapproved": 0,
        "by_model": defaultdict(lambda: {"total": 0, "approved": 0, "unapproved": 0}),
        "by_dataset": defaultdict(lambda: {"total": 0, "approved": 0, "unapproved": 0, "original_size": 0}),
        "by_model_dataset": defaultdict(lambda: {"total": 0, "approved": 0, "unapproved": 0}),
        "last_updated": datetime.now().isoformat(),
    }

    if not DATA_DIR.exists():
        return stats

    for file in DATA_DIR.glob("*.jsonl"):
        try:
            with open(file) as f:
                for line in f:
                    if not line.strip():
                        continue
                    entry = json.loads(line)

                    # Get model name from the entry itself, extract just the model name (after last /)
                    full_model = entry.get("model", "unknown")
                    model_key = full_model.split("/")[-1] if "/" in full_model else full_model

                    # Get dataset name from metadata
                    dataset_key = entry.get("metadata", {}).get("dataset_name", "unknown")

                    model_dataset_key = f"{model_key}:{dataset_key}"

                    is_approved = entry.get("metadata", {}).get("approved", False)

                    stats["total_samples"] += 1
                    if is_approved:
                        stats["approved"] += 1
                    else:
                        stats["unapproved"] += 1

                    for key_dict in [
                        stats["by_model"][model_key],
                        stats["by_dataset"][dataset_key],
                        stats["by_model_dataset"][model_dataset_key],
                    ]:
                        key_dict["total"] += 1
                        if is_approved:
                            key_dict["approved"] += 1
                        else:
                            key_dict["unapproved"] += 1
        except Exception as e:
            console.print(f"[red]Error reading {file.name}: {e}[/red]")

    # Convert defaultdicts to regular dicts
    stats["by_model"] = dict(stats["by_model"])
    stats["by_dataset"] = dict(stats["by_dataset"])
    stats["by_model_dataset"] = dict(stats["by_model_dataset"])

    return stats


@app.route("/")
def index():
    return render_template("dashboard.html")


@app.route("/api/stats")
def api_stats():
    """Return aggregated statistics."""
    stats = load_stats()
    return jsonify(stats)


if __name__ == "__main__":
    console.print("[bold cyan]Starting Dashboard...[/bold cyan]")
    console.print(f"[cyan]Data directory: {DATA_DIR}[/cyan]")
    console.print("[cyan]Visit: http://localhost:5000[/cyan]")
    app.run(debug=False, host="127.0.0.1", port=5000)
