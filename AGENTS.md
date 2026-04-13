# AGENTS.md — Development Conventions

This file documents conventions that human developers **and AI coding agents**
should follow when working in this repository. Read it before writing new code
or making significant changes.

---

## Terminal Output

**Always use [Rich](https://github.com/Textualize/rich) for terminal output.**

Do not use bare `print()`. Rich gives us consistent, readable, coloured output
with progress bars, formatted tables, and structured error messages.

### Key patterns

```python
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, MofNCompleteColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table

console = Console()

# Styled messages
console.print("[bold]Starting…[/bold]")
console.print("[green]OK[/green]")
console.print("[red]FAILED[/red]")
console.print("[yellow]Warning: …[/yellow]")

# Horizontal section divider
console.rule("[bold blue]Judge 1/3: gpt-5-mini[/bold blue]")

# Inline progress (no newline until result is known)
console.print(f"  [1/N] file.jsonl → entry_id … ", end="")
# … do work …
console.print("[green]OK[/green]")   # appended to same line

# Summary table
table = Table(title="Results", show_lines=True)
table.add_column("File", style="cyan")
table.add_column("Labeled", justify="right", style="green")
table.add_column("Skipped", justify="right", style="yellow")
table.add_column("Failed",  justify="right", style="red")
table.add_row("run1.jsonl", "42", "3", "0")
console.print(table)

# Progress bar (file processing)
with Progress(
    SpinnerColumn(),
    TextColumn("[progress.description]{task.description}"),
    BarColumn(),
    MofNCompleteColumn(),
    TimeElapsedColumn(),
    TimeRemainingColumn(),
    console=console,
) as progress:
    task = progress.add_task("[cyan]run1.jsonl[/cyan]", total=100)
    for i in range(100):
        # … do work …
        progress.advance(task)
```

---

## Web Pages & Viewers

Any HTML/CSS produced in this repository **must** follow the rules below.

### Everforest Color Scheme

Use these CSS custom properties in every page.  Populate both dark and light
variants so the theme switcher works automatically.

```css
/* ── Everforest dark (medium contrast) ── */
[data-theme="dark"] {
  --bg:        #2d353b;
  --bg-dim:    #232a2e;
  --bg-1:      #343f44;
  --bg-2:      #3d484d;
  --bg-3:      #475258;
  --fg:        #d3c6aa;
  --fg-dim:    #9da9a0;
  --red:       #e67e80;
  --orange:    #e69875;
  --yellow:    #dbbc7f;
  --green:     #a7c080;
  --aqua:      #83c092;
  --blue:      #7fbbb3;
  --purple:    #d699b6;
  --grey:      #7a8478;
}

/* ── Everforest light (medium contrast) ── */
[data-theme="light"] {
  --bg:        #fdf6e3;
  --bg-dim:    #f4f0d9;
  --bg-1:      #e9e4ca;
  --bg-2:      #ddd8be;
  --bg-3:      #cac9ad;
  --fg:        #5c6a72;
  --fg-dim:    #829181;
  --red:       #f85552;
  --orange:    #f57d26;
  --yellow:    #dfa000;
  --green:     #8da101;
  --aqua:      #35a77c;
  --blue:      #3a94c5;
  --purple:    #df69ba;
  --grey:      #a6b0a0;
}
```

### Dark / Light Mode Switcher

Every page must include a **dark/light mode toggle button**.

```html
<!-- Minimal example — adapt styling to fit the page -->
<button id="theme-toggle" aria-label="Toggle dark/light mode">🌙</button>

<script>
  (function () {
    const root = document.documentElement;
    const btn  = document.getElementById("theme-toggle");
    const saved = localStorage.getItem("theme") || "dark";
    root.dataset.theme = saved;
    btn.textContent = saved === "dark" ? "☀️" : "🌙";

    btn.addEventListener("click", () => {
      const next = root.dataset.theme === "dark" ? "light" : "dark";
      root.dataset.theme = next;
      localStorage.setItem("theme", next);
      btn.textContent = next === "dark" ? "☀️" : "🌙";
    });
  })();
</script>
```

### Card Layout for Multiple Results

When displaying a list of results (entries, labels, models, files…),
**use cards** — not bare `<div>` rows, not `<table>` inside the main content
area.

A card:
- Has a visible background (`var(--bg-1)`) that lifts it off the page background
- Has rounded corners (`border-radius: 8px` or similar)
- Has a subtle border (`1px solid var(--bg-3)`)
- Has a box-shadow for depth on the dark theme
- Has consistent internal padding (`16px`)

```css
.card {
  background:    var(--bg-1);
  border:        1px solid var(--bg-3);
  border-radius: 8px;
  padding:       16px;
  box-shadow:    0 2px 6px rgba(0, 0, 0, 0.25);
}
```

When a grid of cards is needed:

```css
.card-grid {
  display:               grid;
  grid-template-columns: repeat(auto-fill, minmax(280px, 1fr));
  gap:                   16px;
}
```

### Typography

Use [Inter](https://fonts.google.com/specimen/Inter) (from Google Fonts).

```html
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap" rel="stylesheet">
```

```css
body {
  font-family: 'Inter', system-ui, sans-serif;
  background:  var(--bg-dim);
  color:       var(--fg);
}
```

---

## Config-Driven Labeling Runs

All labeling runs are configured via **YAML config files** — never hardcoded
in Python scripts.  See `config.yaml`, `lmstudio.yaml`, `compare_api.yaml`,
and `compare_local.yaml` for examples.

The entry point for all labeling runs is:

```bash
uv run run.py [FILES...] [OPTIONS]
```

There is **no** `sentence-labeler` CLI command — `uv run run.py` is the sole
entry point.

### Adding a new judge

Add a new item to the `judges:` list in the appropriate config file:

```yaml
judges:
  - name: my-new-model
    model_id: my-new-model-id
    backend: openai          # openai | gemini | openrouter | vllm | custom
    api_key_source: openai-api-key
    temperature: 0.4
    max_tokens: 8192
    timeout: 120
```

### Running a comparison across multiple judges

```bash
# Sample 1 entry per input file, run all judges, write merged JSON:
uv run run.py --config compare_api.yaml --compare-output ../../data/1_generated/

# Override seed and sample size:
uv run run.py --config compare_api.yaml --compare-output --seed 99 --sample 3 ../../data/1_generated/

# Run only specific judges from a config:
uv run run.py --config compare_api.yaml --judge gpt-5-mini ../../data/1_generated/
```
