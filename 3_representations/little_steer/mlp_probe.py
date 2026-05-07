"""
little_steer.mlp_probe

Joint multi-label behavior classifiers for K-Steering (Oozeer et al. 2025).

Two probe types with the same interface:

  MLPProbe
      2-layer MLP (256 hidden units each, ReLU) with a multi-label sigmoid output.
      Captures non-linear interactions between activation dimensions.

  LinearProbeMultilabel
      N independent logistic regressions (one per label) wrapped as a single
      nn.Linear for autograd compatibility.  Use to compare against MLPProbe.

Both are trained via MLPProbeTrainer.train(), which reads activations directly
from an ExtractionResult so no extra forward passes are needed.

Training data construction:
  Labels are ordered by importance (as stored in the ExtractionResult labels
  list, which mirrors the order passed by the caller).  The ``max_labels``
  parameter limits how many labels are counted per annotation — e.g.
  ``max_labels=1`` trains only on the primary (first) label for each span,
  ignoring any secondary labels.

Save / load:
  probe.save("probe.pt")
  probe = MLPProbe.load("probe.pt")         # or LinearProbeMultilabel.load(...)
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import torch
import torch.nn as nn
from tqdm.auto import tqdm

if TYPE_CHECKING:
    from .extraction.result import ExtractionResult


# ---------------------------------------------------------------------------
# Probe classes
# ---------------------------------------------------------------------------


class MLPProbe(nn.Module):
    """Joint multi-label MLP classifier over model activations.

    Architecture (matching the K-Steering paper):
        Linear(d_model, 256) → ReLU → Linear(256, 256) → ReLU → Linear(256, n_labels)

    Forward returns raw logits — apply sigmoid for probabilities.
    Labels are not mutually exclusive: use sigmoid per label, not softmax.

    Attributes:
        labels:       Ordered list of label names.
        label_to_idx: {label_name → column index in forward output}.
        input_dim:    Activation dimension the probe was trained on.
        hidden_dim:   Width of the two hidden layers (default 256).
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        hidden_dim: int = 256,
        labels: list[str] | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.labels: list[str] = labels or []
        self.label_to_idx: dict[str, int] = {l: i for i, l in enumerate(self.labels)}

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_labels),
        )

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (..., num_labels)."""
        return self.net(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions of shape (..., num_labels)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)) >= threshold

    def save(self, path: str | Path) -> None:
        """Save probe weights and metadata to a .pt file."""
        torch.save({
            "type": "MLPProbe",
            "state_dict": self.state_dict(),
            "labels": self.labels,
            "input_dim": self.input_dim,
            "hidden_dim": self.hidden_dim,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "MLPProbe":
        """Load a saved MLPProbe from a .pt file."""
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("type") not in ("MLPProbe", None):
            raise ValueError(
                f"File contains a {data.get('type')!r}, not MLPProbe. "
                f"Use LinearProbeMultilabel.load() instead."
            )
        probe = cls(
            input_dim=data["input_dim"],
            num_labels=len(data["labels"]),
            hidden_dim=data.get("hidden_dim", 256),
            labels=data["labels"],
        )
        probe.load_state_dict(data["state_dict"])
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return (
            f"MLPProbe(input_dim={self.input_dim}, hidden_dim={self.hidden_dim}, "
            f"num_labels={self.num_labels})"
        )


class LinearProbeMultilabel(nn.Module):
    """N independent linear probes wrapped as a single nn.Linear.

    Each column of the weight matrix corresponds to one label's logistic
    regression.  Training uses sklearn's LogisticRegression per label so
    regularisation and class-balancing are handled properly; weights are then
    copied into the nn.Linear for autograd-compatible inference (needed by
    K-Steering gradient computation).

    Attributes:
        labels:       Ordered list of label names.
        label_to_idx: {label_name → column index in forward output}.
        input_dim:    Activation dimension.
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
        labels: list[str] | None = None,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.labels: list[str] = labels or []
        self.label_to_idx: dict[str, int] = {l: i for i, l in enumerate(self.labels)}
        self.linear = nn.Linear(input_dim, num_labels)

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits of shape (..., num_labels)."""
        return self.linear(x)

    def predict(self, x: torch.Tensor, threshold: float = 0.5) -> torch.Tensor:
        """Return binary predictions of shape (..., num_labels)."""
        with torch.no_grad():
            return torch.sigmoid(self.forward(x)) >= threshold

    def save(self, path: str | Path) -> None:
        torch.save({
            "type": "LinearProbeMultilabel",
            "state_dict": self.state_dict(),
            "labels": self.labels,
            "input_dim": self.input_dim,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LinearProbeMultilabel":
        data = torch.load(path, map_location="cpu", weights_only=False)
        if data.get("type") not in ("LinearProbeMultilabel", None):
            raise ValueError(
                f"File contains a {data.get('type')!r}, not LinearProbeMultilabel. "
                f"Use MLPProbe.load() instead."
            )
        probe = cls(
            input_dim=data["input_dim"],
            num_labels=len(data["labels"]),
            labels=data["labels"],
        )
        probe.load_state_dict(data["state_dict"])
        probe.eval()
        return probe

    def __repr__(self) -> str:
        return (
            f"LinearProbeMultilabel(input_dim={self.input_dim}, "
            f"num_labels={self.num_labels})"
        )


def load_probe(path: str | Path) -> "MLPProbe | LinearProbeMultilabel":
    """Load either probe type from a saved file (auto-detects type)."""
    data = torch.load(path, map_location="cpu", weights_only=False)
    probe_type = data.get("type", "MLPProbe")
    if probe_type == "MLPProbe":
        return MLPProbe.load(path)
    elif probe_type == "LinearProbeMultilabel":
        return LinearProbeMultilabel.load(path)
    else:
        raise ValueError(f"Unknown probe type in file: {probe_type!r}")


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class MLPProbeTrainer:
    """Train MLPProbe or LinearProbeMultilabel from an ExtractionResult.

    Example:
        trainer = MLPProbeTrainer()

        # MLP probe on all labels
        probe = trainer.train(result, spec="last_token", layer=20)

        # Linear probe for comparison
        linear_probe = trainer.train(result, spec="last_token", layer=20,
                                     method="linear")

        # Use only the primary label per annotation
        probe_primary = trainer.train(result, spec="last_token", layer=20,
                                      max_labels=1)
    """

    def train_from_tensors(
        self,
        X: torch.Tensor,
        Y: torch.Tensor,
        labels: list[str],
        method: Literal["mlp", "linear"] = "mlp",
        *,
        epochs: int = 30,
        batch_size: int | None = None,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        hidden_dim: int = 256,
        show_progress: bool = True,
        progress_fn=None,
        return_history: bool = False,
        return_on_device: bool = False,
    ) -> "MLPProbe | LinearProbeMultilabel | tuple":
        """Train from pre-computed (X, Y) tensors, skipping data construction.

        Use this in tight loops (e.g. iterating over layers) to avoid
        re-running _build_training_data on every call.  Pass X and Y already
        on the target device for maximum efficiency.

        batch_size defaults to len(X) (full-batch) when None.
        return_on_device keeps the probe on the training device after training
        (useful for computing gradient directions before moving to CPU).
        """
        if len(X) == 0:
            raise ValueError("X is empty — no activations to train on.")

        input_dim = X.shape[1]
        _batch_size = len(X) if batch_size is None else batch_size

        history: list[dict] = []
        if method == "mlp":
            probe: MLPProbe | LinearProbeMultilabel = MLPProbe(
                input_dim=input_dim,
                num_labels=len(labels),
                hidden_dim=hidden_dim,
                labels=labels,
            )
            history = self._train_mlp(probe, X, Y, epochs, _batch_size, lr, device, show_progress, progress_fn, return_on_device=return_on_device)
        elif method == "linear":
            probe = LinearProbeMultilabel(
                input_dim=input_dim,
                num_labels=len(labels),
                labels=labels,
            )
            self._train_linear(probe, X.cpu(), Y.cpu())
        else:
            raise ValueError(f"Unknown method {method!r}. Choose 'mlp' or 'linear'.")

        probe.eval()
        if return_history:
            return probe, history
        return probe

    def train(
        self,
        extraction_result: "ExtractionResult",
        spec: str,
        layer: int,
        labels: list[str] | None = None,
        method: Literal["mlp", "linear"] = "mlp",
        *,
        epochs: int = 30,
        batch_size: int = 32,
        lr: float = 1e-3,
        device: str | torch.device = "cpu",
        hidden_dim: int = 256,
        max_labels: int | None = None,
        show_progress: bool = True,
        progress_fn=None,
        return_history: bool = False,
    ) -> "MLPProbe | LinearProbeMultilabel | tuple":
        """Train and return a probe.

        Args:
            extraction_result: Output from ActivationExtractor.extract().
            spec:              Extraction spec name to train on (e.g. "last_token").
            layer:             Transformer layer index.
            labels:            Ordered list of label names.  The order defines
                               label priority for ``max_labels`` — earlier = higher
                               priority.  Defaults to sorted(result.labels()).
            method:            ``"mlp"`` (default) or ``"linear"``.
            epochs:            Training epochs (MLP only; linear uses sklearn).
            batch_size:        Mini-batch size (MLP only).
            lr:                Adam learning rate (MLP only).
            device:            Torch device for MLP training.
            hidden_dim:        Hidden layer width (MLP only).
            max_labels:        Max labels per annotation to include in training.
                               ``None`` = use all labels.
                               ``1`` = use only the primary label (first in ``labels``
                               order), ignoring secondary/tertiary labels.
            show_progress:     Show tqdm progress bar (MLP only).
            return_history:    If True, return ``(probe, history)`` instead of just
                               the probe.  ``history`` is a list of dicts with keys
                               ``epoch``, ``loss``, ``acc`` (MLP only; linear returns
                               an empty list).

        Returns:
            Trained MLPProbe or LinearProbeMultilabel, on CPU, in eval mode.
            If return_history=True, returns (probe, history) where history is
            list[dict] with per-epoch {"epoch", "loss", "acc"}.
        """
        if labels is None:
            labels = sorted(extraction_result.labels())

        # Validate
        available = set(extraction_result.labels())
        missing = [l for l in labels if l not in available]
        if missing:
            warnings.warn(
                f"Labels not found in extraction result and will be skipped: {missing}"
            )
            labels = [l for l in labels if l in available]
        if not labels:
            raise ValueError("No valid labels to train on.")

        X, Y = self._build_training_data(extraction_result, spec, layer, labels, max_labels)
        if len(X) == 0:
            raise ValueError(
                f"No activations found for spec={spec!r}, layer={layer}. "
                f"Check that the ExtractionResult contains data for this spec/layer."
            )

        input_dim = X.shape[1]

        history: list[dict] = []
        if method == "mlp":
            probe: MLPProbe | LinearProbeMultilabel = MLPProbe(
                input_dim=input_dim,
                num_labels=len(labels),
                hidden_dim=hidden_dim,
                labels=labels,
            )
            history = self._train_mlp(probe, X, Y, epochs, batch_size, lr, device, show_progress, progress_fn)
        elif method == "linear":
            probe = LinearProbeMultilabel(
                input_dim=input_dim,
                num_labels=len(labels),
                labels=labels,
            )
            self._train_linear(probe, X, Y)
        else:
            raise ValueError(f"Unknown method {method!r}. Choose 'mlp' or 'linear'.")

        probe.eval()
        print(
            f"✅ Trained {type(probe).__name__} | "
            f"spec={spec!r} layer={layer} labels={len(labels)} samples={len(X)}"
        )
        if return_history:
            return probe, history
        return probe

    # ------------------------------------------------------------------
    # Training data construction
    # ------------------------------------------------------------------

    def _build_training_data(
        self,
        extraction_result: "ExtractionResult",
        spec: str,
        layer: int,
        labels: list[str],
        max_labels: int | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (X, Y) tensors from the ExtractionResult.

        Deduplicates activations by tensor content: if an annotation had two
        labels, the same activation tensor appears under both label keys in the
        ExtractionResult.  We recover the per-annotation multi-hot label vector
        by collecting all labels for each unique activation.

        Labels are iterated in the order provided, so ``max_labels`` naturally
        selects the highest-priority labels first.
        """
        n_labels = len(labels)

        # sample_map: tensor_bytes → (activation_tensor, list_of_label_indices)
        sample_map: dict[bytes, tuple[torch.Tensor, list[int]]] = {}

        for lbl_idx, lbl in enumerate(labels):
            acts = extraction_result.get(spec, lbl, layer)
            for act in acts:
                # Ensure 1-D (hidden_dim,) for consistency
                if act.dim() > 1:
                    act = act.float().mean(dim=0)
                key = act.float().numpy().tobytes()
                if key not in sample_map:
                    sample_map[key] = (act, [])
                current = sample_map[key][1]
                if max_labels is None or len(current) < max_labels:
                    current.append(lbl_idx)

        if not sample_map:
            return torch.zeros(0, 1), torch.zeros(0, n_labels)

        X_list: list[torch.Tensor] = []
        Y_list: list[torch.Tensor] = []
        for act, lbl_indices in sample_map.values():
            y = torch.zeros(n_labels)
            for idx in lbl_indices:
                y[idx] = 1.0
            X_list.append(act.float())
            Y_list.append(y)

        return torch.stack(X_list), torch.stack(Y_list)

    # ------------------------------------------------------------------
    # MLP training loop
    # ------------------------------------------------------------------

    def _train_mlp(
        self,
        probe: MLPProbe,
        X: torch.Tensor,
        Y: torch.Tensor,
        epochs: int,
        batch_size: int,
        lr: float,
        device: str | torch.device,
        show_progress: bool,
        progress_fn=None,
        return_on_device: bool = False,
    ) -> list[dict]:
        """Train MLP probe and return per-epoch history."""
        probe.to(device)
        X = X.to(device)
        Y = Y.to(device)

        optimizer = torch.optim.Adam(probe.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss()

        n = len(X)
        if progress_fn is not None:
            iterator = progress_fn(range(epochs))
        elif show_progress:
            iterator = tqdm(range(epochs), desc="Training MLP probe")
        else:
            iterator = range(epochs)
        history: list[dict] = []

        for epoch_idx in iterator:
            probe.train()
            perm = torch.randperm(n, device=device)
            epoch_loss = 0.0
            correct = 0
            total = 0
            for start in range(0, n, batch_size):
                batch_idx = perm[start : start + batch_size]
                logits = probe(X[batch_idx])
                loss = criterion(logits, Y[batch_idx])
                optimizer.zero_grad()
                torch.autograd.backward([loss])
                optimizer.step()
                epoch_loss += loss.item() * len(batch_idx)
                preds = (torch.sigmoid(logits.detach()) >= 0.5).float()
                correct += (preds == Y[batch_idx]).all(dim=1).sum().item()
                total += len(batch_idx)

            avg_loss = epoch_loss / n
            acc = correct / total if total > 0 else 0.0
            history.append({"epoch": epoch_idx + 1, "loss": avg_loss, "acc": acc})

            if show_progress and hasattr(iterator, "set_postfix"):
                iterator.set_postfix(loss=f"{avg_loss:.4f}", acc=f"{acc:.3f}")

        if not return_on_device:
            probe.cpu()
        return history

    # ------------------------------------------------------------------
    # Linear training (sklearn per-label, weights copied into nn.Linear)
    # ------------------------------------------------------------------

    def _train_linear(
        self,
        probe: LinearProbeMultilabel,
        X: torch.Tensor,
        Y: torch.Tensor,
    ) -> None:
        try:
            from sklearn.linear_model import LogisticRegression
            from sklearn.preprocessing import StandardScaler
        except ImportError as e:
            raise ImportError(
                "scikit-learn is required for method='linear'. "
                "Install with: uv add scikit-learn"
            ) from e

        import numpy as np

        X_np = X.numpy().astype(np.float64)
        Y_np = Y.numpy()

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_np)

        sigma = X_np.std(axis=0) + 1e-8
        mu = X_np.mean(axis=0)

        weights: list[np.ndarray] = []
        biases: list[float] = []

        for j in range(Y_np.shape[1]):
            y_j = Y_np[:, j]
            n_pos = y_j.sum()

            if n_pos == 0 or n_pos == len(y_j):
                # Trivial class — use zero weights, constant bias
                weights.append(np.zeros(X_np.shape[1]))
                p = y_j.mean()
                biases.append(float(np.log(p + 1e-7) - np.log(1 - p + 1e-7)))
                continue

            lr_clf = LogisticRegression(C=1.0, max_iter=1000, class_weight="balanced")
            lr_clf.fit(X_scaled, y_j)

            # Project weights back from standardised to raw activation space
            w_raw = lr_clf.coef_[0] / sigma
            b_raw = float(lr_clf.intercept_[0]) - float((w_raw * mu).sum())
            weights.append(w_raw)
            biases.append(b_raw)

        W = torch.tensor(np.stack(weights), dtype=torch.float32)   # (n_labels, input_dim)
        b = torch.tensor(biases, dtype=torch.float32)               # (n_labels,)
        with torch.no_grad():
            probe.linear.weight.copy_(W)
            probe.linear.bias.copy_(b)
