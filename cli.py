#!/usr/bin/env python3
"""cli.py — Unified interactive TUI for the little_steer pipeline.

Usage:
    uv run cli.py
"""
from __future__ import annotations

import asyncio
import shlex
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from textual import on, work
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, ScrollableContainer
from textual.screen import ModalScreen, Screen
from textual.widgets import (
    Button,
    Checkbox,
    Footer,
    Header,
    Input,
    Label,
    ListItem,
    ListView,
    RadioButton,
    RadioSet,
    RichLog,
    SelectionList,
    Static,
)
from textual.widgets.selection_list import Selection

ROOT = Path(__file__).parent


# ── Config loaders ─────────────────────────────────────────────────────────────

def load_models() -> list[str]:
    cfg = yaml.safe_load((ROOT / "1_generating/config.yaml").read_text())
    return [m["name"] for m in cfg.get("models", [])]


def load_datasets() -> list[str]:
    cfg = yaml.safe_load((ROOT / "1_generating/config.yaml").read_text())
    return [d["name"] for d in cfg.get("datasets", [])]


def load_labelers() -> list[tuple[str, str, bool]]:
    cfg = yaml.safe_load((ROOT / "2_labeling/2b_sentence/labelers.yaml").read_text())
    return [
        (lb["judge"]["name"], lb["judge"].get("backend", "?"), lb.get("enabled", True))
        for lb in cfg.get("labelers", [])
    ]


# ── Step metadata ──────────────────────────────────────────────────────────────

@dataclass
class Step:
    id: str
    number: str
    title: str
    description: str
    cwd: str
    details: list[str] = field(default_factory=list)
    configurable: bool = True  # False = Enter/D launches directly, no modal


STEPS: list[Step] = [
    # ── Phase 1: Generation ───────────────────────────────────────────────────
    Step(
        id="generate",
        number="1",
        title="Generate Responses",
        description="Run LLM inference to generate model responses for configured datasets.",
        cwd="1_generating",
        details=[
            "Script: generate_responses.py",
            "Config:  config.yaml",
            "Options: models (checkboxes), datasets (checkboxes)",
        ],
    ),
    Step(
        id="dashboard",
        number="1.1",
        title="Generation Dashboard",
        description="Flask web server that shows real-time generation progress and sample output.",
        cwd="1_generating",
        details=[
            "Script: dashboard.py",
            "Opens:   http://localhost:5000",
        ],
        configurable=False,
    ),
    Step(
        id="quality",
        number="1.5",
        title="Quality Audit",
        description="Detect and repair quality issues: think-tag artifacts, repetition, foreign scripts.",
        cwd="1_generating",
        details=[
            "Script: fix_quality.py",
            "Actions: tag (detect) / fix (repair) / remove (delete)",
            "Filters: --model, --dataset, --dry-run",
        ],
    ),
    Step(
        id="safety_score",
        number="1.6",
        title="Safety Scoring",
        description="Run WildGuard or Qwen3Guard safety scoring on generated files via vLLM.",
        cwd="1_generating/safety_scoring",
        details=[
            "Script: score.py",
            "Guards: wildguard / qwen3guard",
            "Options: file filter, dry-run",
        ],
    ),
    Step(
        id="safety_stats",
        number="1.7",
        title="Safety Score Stats",
        description="Print safety score statistics from generated data.",
        cwd="1_generating/safety_scoring",
        details=[
            "Script: stats.py",
            "Options: --guard wildguard / qwen3guard",
        ],
    ),
    # ── Phase 2a: Label evolution ─────────────────────────────────────────────
    Step(
        id="evolve",
        number="2a",
        title="Evolve Taxonomy",
        description="Run the iterative taxonomy evolution loop using an LLM labeler.",
        cwd="2_labeling/2a_evolving",
        details=[
            "Script: run.py",
            "Subcommands: run / visualize / list",
            "Options: labeler model, steps, resume run ID",
        ],
    ),
    Step(
        id="evolve_visualizer",
        number="2a.1",
        title="Evolution Visualizer",
        description="Lightweight HTTP server that opens a browser viewer for label evolution runs.",
        cwd="2_labeling/2a_evolving",
        details=[
            "Script: visualizer.py",
            "Options: --port (default 7860), --run-id",
            "Opens:   http://localhost:7860",
        ],
    ),
    Step(
        id="evolve_explorer",
        number="2a.2",
        title="Evolution Explorer",
        description="Flask backend API for the label evolution explorer (serves run data and taxonomy).",
        cwd="2_labeling/2a_evolving",
        details=[
            "Script: serve.py",
            "Opens:   http://localhost:5000  (or configured port)",
        ],
        configurable=False,
    ),
    # ── Phase 2b: Sentence labeling ───────────────────────────────────────────
    Step(
        id="label",
        number="2b",
        title="Sentence Labeling",
        description="Run multi-judge sentence-level span labeling over the canonical dataset.",
        cwd="2_labeling/2b_sentence",
        details=[
            "Script: run.py",
            "Config:  labelers.yaml",
            "Options: select judges (checkboxes)",
        ],
    ),
    Step(
        id="label_all",
        number="2b.1",
        title="Run All Judges",
        description="Launch all enabled judges in parallel with a live Rich progress dashboard.",
        cwd="2_labeling/2b_sentence",
        details=[
            "Script: run_all.py",
            "Runs all enabled judges from labelers.yaml in parallel",
        ],
        configurable=False,
    ),
    Step(
        id="label_viewer",
        number="2b.2",
        title="Labeled Data Viewer",
        description="Flask web viewer for labeled reasoning traces with filtering, search, and stats.",
        cwd="2_labeling/2b_sentence",
        details=[
            "Script: viewer/app.py",
            "Opens:   http://localhost:5000",
        ],
        configurable=False,
    ),
    Step(
        id="iaa_viewer",
        number="2b.3",
        title="IAA Viewer",
        description="Inter-Annotator Agreement heatmap: compare agreement between judges over shared entries.",
        cwd="2_labeling/2b_sentence",
        details=[
            "Script: iaa/app.py",
            "Opens:   http://localhost:5051",
            "Metric:  Cohen's κ (character-weighted, handles sentence boundary mismatches)",
        ],
        configurable=False,
    ),
    # ── Phase 2c: Human labeling ──────────────────────────────────────────────
    Step(
        id="human_label",
        number="2c",
        title="Human Labeling UI",
        description="Vite dev server for the React-based human annotation frontend.",
        cwd="2_labeling/2c_human_labeling",
        details=[
            "Command: npm run dev",
            "Opens:   http://localhost:5173",
        ],
        configurable=False,
    ),
]

STEP_BY_ID = {s.id: s for s in STEPS}


# ── Command builders ───────────────────────────────────────────────────────────

def cmd_generate(models: list[str], datasets: list[str]) -> list[str]:
    cmd = ["uv", "run", "generate_responses.py", "--config", "config.yaml"]
    if models:
        cmd += ["--models"] + models
    if datasets:
        cmd += ["--datasets"] + datasets
    return cmd


def cmd_quality(action: str, model: str, dataset: str, dry_run: bool) -> list[str]:
    cmd = ["uv", "run", "fix_quality.py", f"--{action}"]
    if model.strip():
        cmd += ["--model", model.strip()]
    if dataset.strip():
        cmd += ["--dataset", dataset.strip()]
    if dry_run:
        cmd += ["--dry-run"]
    return cmd


def cmd_safety_score(guard: str, files: str, dry_run: bool) -> list[str]:
    cmd = ["uv", "run", "score.py", "--guard", guard]
    if files.strip():
        cmd += ["--files", files.strip()]
    if dry_run:
        cmd += ["--dry-run"]
    return cmd


def cmd_safety_stats(guard: str) -> list[str]:
    return ["uv", "run", "stats.py", "--guard", guard]


def cmd_evolve(labeler: str, steps: str, run_id: str, subcommand: str = "run") -> list[str]:
    if subcommand != "run":
        return ["uv", "run", "run.py", subcommand]
    cmd = ["uv", "run", "run.py", "run", "--labeler", labeler or "gpt-4o-mini"]
    try:
        cmd += ["--steps", str(int(steps))]
    except ValueError:
        cmd += ["--steps", "100"]
    if run_id.strip():
        cmd += ["--run-id", run_id.strip()]
    return cmd


def cmd_evolve_visualizer(port: str, run_id: str) -> list[str]:
    cmd = ["uv", "run", "visualizer.py"]
    try:
        cmd += ["--port", str(int(port))]
    except ValueError:
        cmd += ["--port", "7860"]
    if run_id.strip():
        cmd += ["--run-id", run_id.strip()]
    return cmd


def cmd_label(judges: list[str]) -> list[str]:
    cmd = ["uv", "run", "run.py", "--config", "labelers.yaml"]
    for j in judges:
        cmd += ["--judge", j]
    cmd += ["../../data/dataset.jsonl"]
    return cmd


DEFAULT_CMDS: dict[str, list[str]] = {
    "generate":        cmd_generate([], []),
    "dashboard":       ["uv", "run", "dashboard.py"],
    "quality":         cmd_quality("tag", "", "", False),
    "safety_score":    cmd_safety_score("wildguard", "", False),
    "safety_stats":    cmd_safety_stats("wildguard"),
    "evolve":          cmd_evolve("gpt-4o-mini", "100", ""),
    "evolve_visualizer": cmd_evolve_visualizer("7860", ""),
    "evolve_explorer": ["uv", "run", "serve.py"],
    "label":           cmd_label([]),
    "label_all":       ["uv", "run", "run_all.py"],
    "label_viewer":    ["uv", "run", "viewer/app.py"],
    "iaa_viewer":      ["uv", "run", "iaa/app.py"],
    "human_label":     ["npm", "run", "dev"],
}


# ── Run Screen ─────────────────────────────────────────────────────────────────

class RunScreen(Screen):
    # Q is always available — kills process if running, goes back when done
    BINDINGS = [Binding("q", "stop_or_back", "Stop / Back")]

    DEFAULT_CSS = """
    RunScreen { background: $surface; }
    #cmd-label {
        background: $panel;
        color: $text-muted;
        padding: 0 2;
        height: 1;
    }
    #output-log {
        border: solid $primary;
        margin: 0 1 1 1;
    }
    #status-bar {
        height: 1;
        padding: 0 2;
        background: $panel;
    }
    """

    def __init__(self, cmd: list[str], cwd: str) -> None:
        super().__init__()
        self._cmd = cmd
        self._cwd = str(ROOT / cwd)
        self._done = False
        self._stopping = False
        self._proc: asyncio.subprocess.Process | None = None

    def compose(self) -> ComposeResult:
        yield Header()
        yield Label(f"$ {shlex.join(self._cmd)}", id="cmd-label")
        yield RichLog(highlight=True, markup=True, id="output-log")
        yield Label("Running…  (Q to stop)", id="status-bar")
        yield Footer()

    def on_mount(self) -> None:
        self._stream()

    @work(exclusive=True)
    async def _stream(self) -> None:
        log = self.query_one("#output-log", RichLog)
        try:
            self._proc = await asyncio.create_subprocess_exec(
                *self._cmd,
                cwd=self._cwd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
            )
            assert self._proc.stdout is not None
            async for chunk in self._proc.stdout:
                log.write(chunk.decode(errors="replace").rstrip())
            rc = await self._proc.wait()
        except FileNotFoundError as exc:
            log.write(f"[red]Error: {exc}[/red]")
            rc = 127

        self._done = True
        if self._stopping:
            self.app.pop_screen()
            return
        color = "green" if rc == 0 else "red"
        self.query_one("#status-bar", Label).update(
            f"[{color}]Exited with code {rc}[/{color}]   Press Q to go back"
        )

    def action_stop_or_back(self) -> None:
        if self._done:
            self.app.pop_screen()
        elif self._proc is not None and not self._stopping:
            self._stopping = True
            self._proc.terminate()
            self.query_one("#status-bar", Label).update(
                "[yellow]Stopping…[/yellow]"
            )


# ── Config Modals ──────────────────────────────────────────────────────────────

MODAL_CSS = """
ModalScreen { align: center middle; }
#dialog {
    background: $surface;
    border: thick $primary;
    width: 80;
    max-height: 44;
    padding: 1 2;
}
#dialog-title {
    text-style: bold;
    color: $accent;
    margin-bottom: 1;
}
.section-label { color: $text-muted; margin-top: 1; }
#btn-row { height: 3; margin-top: 1; align: right middle; }
Button { margin-left: 1; }
"""


class GenerateConfigModal(ModalScreen):
    DEFAULT_CSS = MODAL_CSS

    def __init__(self) -> None:
        super().__init__()
        self._models = load_models()
        self._datasets = load_datasets()

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Configure: Generate Responses", id="dialog-title")
            yield Label("Models  (Space to toggle; all = no filter)", classes="section-label")
            yield SelectionList(
                *[Selection(m, m, initial_state=True) for m in self._models],
                id="model-list",
            )
            yield Label("Datasets  (Space to toggle; all = no filter)", classes="section-label")
            yield SelectionList(
                *[Selection(d, d, initial_state=True) for d in self._datasets],
                id="dataset-list",
            )
            with Horizontal(id="btn-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Run", variant="primary", id="btn-run")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-run")
    def run_it(self) -> None:
        models = list(self.query_one("#model-list", SelectionList).selected)
        datasets = list(self.query_one("#dataset-list", SelectionList).selected)
        if len(models) == len(self._models):
            models = []
        if len(datasets) == len(self._datasets):
            datasets = []
        self.dismiss(cmd_generate(models, datasets))


class QualityConfigModal(ModalScreen):
    DEFAULT_CSS = MODAL_CSS

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Configure: Quality Audit", id="dialog-title")
            yield Label("Action", classes="section-label")
            with RadioSet(id="action-set"):
                yield RadioButton("tag    — detect and report issues", value=True)
                yield RadioButton("fix    — repair detected issues")
                yield RadioButton("remove — delete bad entries")
            yield Label("Filter by model name (optional)", classes="section-label")
            yield Input(placeholder="e.g. deepseek", id="model-input")
            yield Label("Filter by dataset name (optional)", classes="section-label")
            yield Input(placeholder="e.g. strong_reject", id="dataset-input")
            yield Checkbox("Dry run (preview only)", id="dry-run")
            with Horizontal(id="btn-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Run", variant="primary", id="btn-run")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-run")
    def run_it(self) -> None:
        radio = self.query_one("#action-set", RadioSet)
        actions = ["tag", "fix", "remove"]
        action = actions[radio.pressed_index] if radio.pressed_index is not None else "tag"
        model = self.query_one("#model-input", Input).value
        dataset = self.query_one("#dataset-input", Input).value
        dry_run = self.query_one("#dry-run", Checkbox).value
        self.dismiss(cmd_quality(action, model, dataset, dry_run))


class SafetyScoringModal(ModalScreen):
    DEFAULT_CSS = MODAL_CSS

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Configure: Safety Scoring", id="dialog-title")
            yield Label("Guard model", classes="section-label")
            with RadioSet(id="guard-set"):
                yield RadioButton("wildguard", value=True)
                yield RadioButton("qwen3guard")
            yield Label("File filter (optional glob, e.g. gpt-oss*)", classes="section-label")
            yield Input(placeholder="leave blank for all files", id="files-input")
            yield Checkbox("Dry run (preview only)", id="dry-run")
            with Horizontal(id="btn-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Run", variant="primary", id="btn-run")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-run")
    def run_it(self) -> None:
        radio = self.query_one("#guard-set", RadioSet)
        guard = "qwen3guard" if radio.pressed_index == 1 else "wildguard"
        files = self.query_one("#files-input", Input).value
        dry_run = self.query_one("#dry-run", Checkbox).value
        self.dismiss(cmd_safety_score(guard, files, dry_run))


class SafetyStatsModal(ModalScreen):
    DEFAULT_CSS = MODAL_CSS

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Configure: Safety Score Stats", id="dialog-title")
            yield Label("Guard model", classes="section-label")
            with RadioSet(id="guard-set"):
                yield RadioButton("wildguard", value=True)
                yield RadioButton("qwen3guard")
            with Horizontal(id="btn-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Run", variant="primary", id="btn-run")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-run")
    def run_it(self) -> None:
        radio = self.query_one("#guard-set", RadioSet)
        guard = "qwen3guard" if radio.pressed_index == 1 else "wildguard"
        self.dismiss(cmd_safety_stats(guard))


class EvolveConfigModal(ModalScreen):
    DEFAULT_CSS = MODAL_CSS

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Configure: Evolve Taxonomy", id="dialog-title")
            yield Label("Subcommand", classes="section-label")
            with RadioSet(id="subcmd-set"):
                yield RadioButton("run       — start or resume evolution loop", value=True)
                yield RadioButton("visualize — show current taxonomy")
                yield RadioButton("list      — list all runs")
            yield Label("Labeler model (for 'run')", classes="section-label")
            yield Input(placeholder="gpt-4o-mini", id="labeler-input")
            yield Label("Steps (for 'run', default 100)", classes="section-label")
            yield Input(placeholder="100", id="steps-input")
            yield Label("Resume run ID (optional, blank = new run)", classes="section-label")
            yield Input(placeholder="leave blank to start a new run", id="runid-input")
            with Horizontal(id="btn-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Run", variant="primary", id="btn-run")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-run")
    def run_it(self) -> None:
        subcmds = ["run", "visualize", "list"]
        radio = self.query_one("#subcmd-set", RadioSet)
        subcmd = subcmds[radio.pressed_index] if radio.pressed_index is not None else "run"
        labeler = self.query_one("#labeler-input", Input).value or "gpt-4o-mini"
        steps = self.query_one("#steps-input", Input).value or "100"
        run_id = self.query_one("#runid-input", Input).value
        self.dismiss(cmd_evolve(labeler, steps, run_id, subcmd))


class EvolveVisualizerModal(ModalScreen):
    DEFAULT_CSS = MODAL_CSS

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Configure: Evolution Visualizer", id="dialog-title")
            yield Label("Port (default 7860)", classes="section-label")
            yield Input(placeholder="7860", id="port-input")
            yield Label("Run ID to visualize (optional, blank = most recent)", classes="section-label")
            yield Input(placeholder="leave blank for most recent run", id="runid-input")
            with Horizontal(id="btn-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Launch", variant="primary", id="btn-run")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-run")
    def run_it(self) -> None:
        port = self.query_one("#port-input", Input).value or "7860"
        run_id = self.query_one("#runid-input", Input).value
        self.dismiss(cmd_evolve_visualizer(port, run_id))


class LabelConfigModal(ModalScreen):
    DEFAULT_CSS = MODAL_CSS

    def __init__(self) -> None:
        super().__init__()
        self._labelers = load_labelers()

    def compose(self) -> ComposeResult:
        with Container(id="dialog"):
            yield Label("Configure: Sentence Labeling", id="dialog-title")
            yield Label("Judges  (Space to toggle; enabled ones pre-checked)", classes="section-label")
            yield SelectionList(
                *[
                    Selection(f"{name}  [{backend}]", name, initial_state=enabled)
                    for name, backend, enabled in self._labelers
                ],
                id="judge-list",
            )
            with Horizontal(id="btn-row"):
                yield Button("Cancel", variant="default", id="btn-cancel")
                yield Button("Run", variant="primary", id="btn-run")

    @on(Button.Pressed, "#btn-cancel")
    def cancel(self) -> None:
        self.dismiss(None)

    @on(Button.Pressed, "#btn-run")
    def run_it(self) -> None:
        judges = list(self.query_one("#judge-list", SelectionList).selected)
        self.dismiss(cmd_label(judges))


MODAL_BY_STEP: dict[str, type[ModalScreen]] = {
    "generate":         GenerateConfigModal,
    "quality":          QualityConfigModal,
    "safety_score":     SafetyScoringModal,
    "safety_stats":     SafetyStatsModal,
    "evolve":           EvolveConfigModal,
    "evolve_visualizer": EvolveVisualizerModal,
    "label":            LabelConfigModal,
}


# ── Main Screen ────────────────────────────────────────────────────────────────

class MainScreen(Screen):
    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("d", "run_defaults", "Run defaults"),
    ]

    DEFAULT_CSS = """
    MainScreen { background: $surface; }
    #layout { layout: horizontal; }
    #step-list-pane {
        width: 36;
        border: solid $primary;
        margin: 1 0 1 1;
        padding: 0;
    }
    #detail-pane {
        border: solid $primary-darken-1;
        margin: 1 1 1 1;
        padding: 1 2;
    }
    #step-number { color: $accent; text-style: bold; }
    #step-title  { text-style: bold; margin-bottom: 1; }
    #step-desc   { color: $text; margin-bottom: 1; }
    #step-details { color: $text-muted; }
    #hint        { color: $text-muted; margin-top: 2; }
    ListView > ListItem { padding: 0 1; }
    ListView > ListItem.--highlight {
        background: $primary 30%;
        color: $text;
    }
    """

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Horizontal(id="layout"):
            with Container(id="step-list-pane"):
                yield ListView(
                    *[
                        ListItem(Label(f"[{s.number}]  {s.title}"), id=f"item-{s.id}")
                        for s in STEPS
                    ],
                    id="step-list",
                )
            with ScrollableContainer(id="detail-pane"):
                yield Label("", id="step-number")
                yield Label("", id="step-title")
                yield Static("", id="step-desc")
                yield Static("", id="step-details")
                yield Label("", id="hint")
        yield Footer()

    def on_mount(self) -> None:
        self.query_one("#step-list", ListView).focus()
        self._refresh_detail(STEPS[0])

    @on(ListView.Highlighted)
    def _on_highlighted(self, event: ListView.Highlighted) -> None:
        idx = event.list_view.index
        if idx is not None and 0 <= idx < len(STEPS):
            self._refresh_detail(STEPS[idx])

    @on(ListView.Selected)
    def _on_selected(self, event: ListView.Selected) -> None:
        idx = event.list_view.index
        if idx is None or not (0 <= idx < len(STEPS)):
            return
        step = STEPS[idx]
        if step.configurable:
            modal_cls = MODAL_BY_STEP.get(step.id)
            if modal_cls:
                self.app.push_screen(modal_cls(), callback=lambda cmd, s=step: self._launch(s, cmd))
        else:
            self._launch(step, DEFAULT_CMDS[step.id])

    def _refresh_detail(self, step: Step) -> None:
        self.query_one("#step-number", Label).update(f"Step {step.number}")
        self.query_one("#step-title", Label).update(step.title)
        self.query_one("#step-desc", Static).update(step.description)
        self.query_one("#step-details", Static).update(
            "\n".join(f"  {d}" for d in step.details)
        )
        if step.configurable:
            hint = "  Enter  Configure & run\n  D      Run with defaults"
        else:
            hint = "  Enter / D  Launch"
        self.query_one("#hint", Label).update(hint)

    def action_run_defaults(self) -> None:
        lv = self.query_one("#step-list", ListView)
        idx = lv.index
        if idx is None or not (0 <= idx < len(STEPS)):
            return
        step = STEPS[idx]
        self._launch(step, DEFAULT_CMDS[step.id])

    def _launch(self, step: Step, cmd: list[str] | None) -> None:
        if cmd is None:
            return
        self.app.push_screen(RunScreen(cmd, step.cwd))

    def action_quit(self) -> None:
        self.app.exit()


# ── App ────────────────────────────────────────────────────────────────────────

class LittleSteerApp(App):
    TITLE = "little_steer"
    SUB_TITLE = "Unified pipeline CLI"

    def on_mount(self) -> None:
        self.push_screen(MainScreen())


if __name__ == "__main__":
    LittleSteerApp().run()
