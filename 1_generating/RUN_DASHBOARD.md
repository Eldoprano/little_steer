# 📊 Sample Generation Dashboard

A real-time web dashboard to monitor your sample generation progress, with per-model and per-dataset statistics, filtering by approval status, and auto-refresh.

## Quick Start

```bash
uv run python dashboard.py
```

Then open your browser to: **http://localhost:5000**

The dashboard will:
- Display total samples, approved/unapproved counts
- Show per-model and per-dataset breakdowns with stacked bar charts
- Provide a detailed table of all model-dataset combinations
- Auto-refresh every 60 seconds
- Allow filtering by approval status

## Features

### Summary Cards
- **Total Samples**: Count of all generated samples
- **Approved**: Count of samples tagged as approved (in metadata)
- **Unapproved**: Count of pending samples
- **Models**: Number of different models generating samples

### Visualizations
1. **Samples by Model** — Stacked bar chart showing approved vs unapproved per model
2. **Samples by Dataset** — Stacked bar chart showing approved vs unapproved per dataset
3. **Detailed Table** — Full breakdown with model, dataset, counts, and approval rates

### Filtering
- **Show**: Toggle between All Samples, Approved Only, or Unapproved Only
- **Model**: Filter to a specific model (for detailed analysis)
- **Dataset**: Filter to a specific dataset (for detailed analysis)

## Data Source

The dashboard reads JSONL files from `../data/1_generated/` and aggregates:
- **Filename format**: `{model}_{dataset}.jsonl`
- **Approval status**: Checked via `metadata.approved` field (added by `fix_quality.py --tag`)

## Auto-Refresh

The dashboard automatically refreshes every 60 seconds. You can also click "Refresh Now" for immediate updates.

## Running in Background

To run the dashboard as a background service:

```bash
nohup uv run python dashboard.py > dashboard.log 2>&1 &
```

Or use `screen`/`tmux`:

```bash
screen -S dashboard -d -m uv run python dashboard.py
```

To reconnect: `screen -r dashboard`

## Troubleshooting

If the dashboard is already running on port 5000, you can change the port in `dashboard.py`:
```python
app.run(debug=False, host="127.0.0.1", port=5001)
```
