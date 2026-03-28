"""
little_steer.models.vram

VRAM detection and batch configuration management.
Provides automatic estimation of safe batch sizes and optional benchmarking.
"""

from __future__ import annotations

import gc
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from .model import LittleSteerModel


CONFIG_FILE = Path(".little_steer_vram_config.json")


@dataclass
class BatchConfig:
    """Recommended configuration for a model extraction run."""

    batch_size: int
    """Number of conversations to process in parallel."""

    max_seq_len: int
    """Maximum token sequence length before truncation."""

    quantization: str | None
    """Quantization used: '8bit', '4bit', or None."""

    throughput_toks_per_sec: float = 0.0
    """Tokens/second measured during benchmarking (0 if not benchmarked)."""

    notes: str = ""

    def describe(self) -> str:
        q = self.quantization or "none"
        return (
            f"BatchConfig(batch_size={self.batch_size}, "
            f"max_seq_len={self.max_seq_len}, "
            f"quantization={q!r}, "
            f"throughput={self.throughput_toks_per_sec:.0f} tok/s)"
        )

    def __repr__(self) -> str:
        return self.describe()


class VRAMManager:
    """Detect available VRAM and provide batch configuration recommendations."""

    def __init__(self, config_file: Path | str = CONFIG_FILE):
        self.config_file = Path(config_file)

    # ------------------------------------------------------------------
    # VRAM detection
    # ------------------------------------------------------------------

    @staticmethod
    def detect_available_vram() -> dict[int, float]:
        """Return available VRAM in GB for each GPU.

        Returns:
            {gpu_idx: available_gb} — empty dict if no CUDA.
        """
        if not torch.cuda.is_available():
            return {}
        result: dict[int, float] = {}
        for i in range(torch.cuda.device_count()):
            free, total = torch.cuda.mem_get_info(i)
            result[i] = free / (1024 ** 3)
        return result

    @staticmethod
    def total_available_vram_gb() -> float:
        """Sum of free VRAM across all GPUs in GB."""
        return sum(VRAMManager.detect_available_vram().values())

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    @staticmethod
    def estimate_batch_config(
        model_params_b: float,
        seq_len: int = 2048,
        num_layers_to_save: int = 1,
        quantization: str | None = None,
    ) -> BatchConfig:
        """Estimate a safe batch configuration without running the model.

        Rule of thumb:
          - Model weights: ~2 bytes/param (fp16) → model_params_b * 2 GB
          - KV cache per token per layer: ~0.5 MB for 7B models, scales with seq_len
          - Activation saving overhead: num_layers_to_save * seq_len * hidden_dim * 4 bytes
          - Safety margin: use 80% of available VRAM

        Args:
            model_params_b: Model parameter count in billions.
            seq_len: Expected sequence length.
            num_layers_to_save: How many layers' activations to hold in memory.
            quantization: '8bit', '4bit', or None.

        Returns:
            Estimated BatchConfig.
        """
        total_vram_gb = VRAMManager.total_available_vram_gb()
        if total_vram_gb == 0:
            # CPU fallback
            return BatchConfig(
                batch_size=1,
                max_seq_len=min(seq_len, 512),
                quantization=quantization,
                notes="CPU mode — no CUDA detected",
            )

        # Estimate model size in VRAM
        bytes_per_param = 2.0  # fp16
        if quantization == "8bit":
            bytes_per_param = 1.0
        elif quantization == "4bit":
            bytes_per_param = 0.5

        model_gb = model_params_b * bytes_per_param
        remaining_gb = total_vram_gb * 0.80 - model_gb  # 80% safety margin

        if remaining_gb <= 0.5:
            return BatchConfig(
                batch_size=1,
                max_seq_len=min(seq_len, 256),
                quantization=quantization,
                notes=f"Low VRAM ({total_vram_gb:.1f}GB) after model load",
            )

        # Activation memory per sequence: rough estimate
        # hidden_dim ~ 128 * sqrt(params_in_millions * 1e6) is a rough heuristic
        # For simplicity, use 0.5GB per sequence per 1B params at seq_len=2048
        act_gb_per_seq = (model_params_b * 0.5 * seq_len / 2048) * num_layers_to_save
        act_gb_per_seq = max(act_gb_per_seq, 0.1)  # floor

        batch_size = max(1, int(remaining_gb / act_gb_per_seq))
        batch_size = min(batch_size, 32)  # cap at 32

        return BatchConfig(
            batch_size=batch_size,
            max_seq_len=seq_len,
            quantization=quantization,
            notes=f"Estimated from {total_vram_gb:.1f}GB available VRAM",
        )

    # ------------------------------------------------------------------
    # Benchmarking
    # ------------------------------------------------------------------

    def benchmark_throughput(
        self,
        model: "LittleSteerModel",
        sample_seq_len: int = 512,
        max_batch_size: int = 16,
        n_warmup: int = 2,
        n_trials: int = 5,
    ) -> BatchConfig:
        """Find the optimal batch size by binary search + timing.

        Starts from batch_size=1, doubles until OOM, then backs off.
        Prints results and saves config.

        Args:
            model: The LittleSteerModel to test.
            sample_seq_len: Sequence length to test with.
            max_batch_size: Upper bound on batch size to try.
            n_warmup: Number of warmup passes.
            n_trials: Number of timing passes per batch size.

        Returns:
            Best BatchConfig found.
        """
        from rich.console import Console
        from rich.table import Table

        console = Console()
        console.print(
            f"[bold cyan]🔍 Benchmarking throughput for [green]{model.model_id}[/green]...[/bold cyan]"
        )

        # Generate a dummy input token sequence
        dummy_input = torch.randint(
            0, model.tokenizer.vocab_size or 32000,
            (sample_seq_len,),
        )

        results: list[tuple[int, float]] = []  # (batch_size, toks/sec)
        best_batch_size = 1
        best_throughput = 0.0

        batch_size = 1
        while batch_size <= max_batch_size:
            try:
                # Warmup
                for _ in range(n_warmup):
                    _run_dummy_extraction(model, dummy_input)

                # Time it
                gc.collect()
                torch.cuda.empty_cache()
                t0 = time.perf_counter()
                for _ in range(n_trials):
                    # Run batch_size independent passes (our extractor processes
                    # conversations one-by-one with nnsight, but we test throughput)
                    for _ in range(batch_size):
                        _run_dummy_extraction(model, dummy_input)
                elapsed = time.perf_counter() - t0

                toks_per_sec = (batch_size * sample_seq_len * n_trials) / elapsed
                results.append((batch_size, toks_per_sec))

                if toks_per_sec > best_throughput:
                    best_throughput = toks_per_sec
                    best_batch_size = batch_size

                batch_size *= 2

            except torch.cuda.OutOfMemoryError:
                console.print(
                    f"  [yellow]⚠️  OOM at batch_size={batch_size} — stopping[/yellow]"
                )
                gc.collect()
                torch.cuda.empty_cache()
                break
            except Exception as e:
                console.print(f"  [red]Error at batch_size={batch_size}: {e}[/red]")
                break

        # Display results
        table = Table(title="Throughput Benchmark Results")
        table.add_column("Batch Size", style="cyan")
        table.add_column("Tokens/sec", style="green")
        table.add_column("", style="yellow")
        for bs, tps in results:
            marker = "⭐ Best" if bs == best_batch_size else ""
            table.add_row(str(bs), f"{tps:.0f}", marker)
        console.print(table)

        config = BatchConfig(
            batch_size=best_batch_size,
            max_seq_len=sample_seq_len,
            quantization=None,
            throughput_toks_per_sec=best_throughput,
            notes=f"Benchmarked with seq_len={sample_seq_len}",
        )

        # Save and print
        self.save_config(config, model.model_id)
        console.print(
            f"[bold green]✅ Best config: {config.describe()}[/bold green]"
        )
        return config

    # ------------------------------------------------------------------
    # Config persistence
    # ------------------------------------------------------------------

    def save_config(self, config: BatchConfig, model_id: str) -> None:
        """Save best config to the config file."""
        existing: dict = {}
        if self.config_file.exists():
            try:
                existing = json.loads(self.config_file.read_text())
            except json.JSONDecodeError:
                pass

        existing[model_id] = asdict(config)
        self.config_file.write_text(json.dumps(existing, indent=2))
        print(f"💾 Saved VRAM config for '{model_id}' → {self.config_file}")

    def load_config(self, model_id: str) -> BatchConfig | None:
        """Load previously saved config for a model, or None."""
        if not self.config_file.exists():
            return None
        try:
            data = json.loads(self.config_file.read_text())
            if model_id in data:
                return BatchConfig(**data[model_id])
        except (json.JSONDecodeError, TypeError, KeyError):
            pass
        return None

    def list_configs(self) -> dict[str, BatchConfig]:
        """Return all saved configs."""
        if not self.config_file.exists():
            return {}
        try:
            data = json.loads(self.config_file.read_text())
            return {k: BatchConfig(**v) for k, v in data.items()}
        except Exception:
            return {}


def _run_dummy_extraction(model: "LittleSteerModel", token_ids: torch.Tensor) -> None:
    """Run one forward pass and extract the last layer output."""
    with torch.no_grad():
        with model.st.trace(token_ids) as tracer:
            _ = model.st.layers_output[-1].detach().cpu().save()
            tracer.stop()
