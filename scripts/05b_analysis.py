"""Cross-model calibration and uncertainty analysis.

Loads the predictions saved by tasks 3 (logistic regression), 4 (random forest)
and 5 (deep ensemble), asserts that the three models were evaluated on
identical outer test folds, then produces five publication-quality figures
that compare their calibration and confidence behaviour.

Run from the project root::

    uv run python scripts/05b_analysis.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.data import CLASS_NAMES  # noqa: E402
from galaxy_uq.metrics import (  # noqa: E402
    accuracy,
    expected_calibration_error,
    reliability_diagram_data,
)

logger = logging.getLogger("05b_analysis")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}

MODELS: list[tuple[str, str, str, str]] = [
    # (key, label, npz path, colour)
    ("lr", "Logistic regression", "results/metrics/03_logreg_predictions.npz", "tab:blue"),
    ("rf", "Random forest",       "results/metrics/04_rf_predictions.npz",    "seagreen"),
    ("cnn", "Deep ensemble CNN",  "results/metrics/05_cnn_predictions.npz",    "purple"),
]
MODEL_MARKERS: dict[str, str] = {"lr": "o", "rf": "s", "cnn": "^"}


def load_predictions() -> dict:
    """Load all three prediction sets and verify they share the same y_true.

    Returns
    -------
    dict
        ``{model_key: {"label", "color", "y_true", "y_pred", "y_prob"}}``.

    Raises
    ------
    ValueError
        If the ``y_true`` arrays differ across models — that would mean the
        models were not evaluated on the same outer folds, breaking the
        comparison.
    """
    bundle: dict = {}
    reference_y_true: np.ndarray | None = None
    for key, label, path, color in MODELS:
        data = np.load(Path(path))
        y_true = data["y_true"]
        y_pred = data["y_pred"]
        y_prob = data["y_prob"]
        if reference_y_true is None:
            reference_y_true = y_true
        elif not np.array_equal(reference_y_true, y_true):
            raise ValueError(
                f"y_true arrays differ between models: '{key}' does not match "
                "the first loaded model — the predictions were not evaluated "
                "on the same outer test folds (seed/k mismatch?)"
            )
        bundle[key] = {
            "label": label,
            "color": color,
            "y_true": y_true,
            "y_pred": y_pred,
            "y_prob": y_prob,
        }
    logger.info("Loaded predictions for %d models on %d samples",
                len(bundle), reference_y_true.size)
    return bundle


def plot_reliability_comparison(bundle: dict, out_path: Path) -> None:
    """Overlay reliability curves with an ECE bar-chart inset."""
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="black",
            label="Perfect calibration", linewidth=1.2)
    eces: dict[str, float] = {}
    for key, info in bundle.items():
        data = reliability_diagram_data(info["y_true"], info["y_prob"], n_bins=15)
        ece = expected_calibration_error(info["y_true"], info["y_prob"], n_bins=15)
        eces[key] = ece
        mask = ~np.isnan(data["bin_accuracies"])
        ax.plot(
            data["bin_confidences"][mask],
            data["bin_accuracies"][mask],
            marker=MODEL_MARKERS[key],
            color=info["color"],
            label=f"{info['label']} (ECE={ece:.3f})",
        )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence in bin")
    ax.set_ylabel("Empirical accuracy in bin")
    ax.set_title("Reliability diagram — all three models")
    ax.legend(loc="upper left")

    inset = fig.add_axes([0.62, 0.18, 0.25, 0.20])
    keys = list(bundle.keys())
    xs = [bundle[k]["label"].split()[0] for k in keys]
    ys = [eces[k] for k in keys]
    colors = [bundle[k]["color"] for k in keys]
    inset.bar(xs, ys, color=colors)
    inset.set_ylim(0, max(ys) * 1.25)
    inset.set_ylabel("ECE", fontsize=8)
    inset.tick_params(axis="both", labelsize=7)
    for x, y in zip(xs, ys):
        inset.text(x, y + max(ys) * 0.02, f"{y:.3f}", ha="center", fontsize=7)
    inset.set_title("ECE", fontsize=8)

    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def risk_coverage_curve(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray,
    coverage_grid: np.ndarray,
) -> np.ndarray:
    """Accuracy on the top ``c * N`` most-confident samples for each ``c``."""
    confidence = y_prob.max(axis=1)
    order = np.argsort(-confidence)  # descending
    n = y_true.size
    accuracies = np.empty_like(coverage_grid, dtype=np.float64)
    for i, c in enumerate(coverage_grid):
        keep = max(1, int(round(c * n)))
        idx = order[:keep]
        accuracies[i] = accuracy(y_true[idx], y_pred[idx])
    return accuracies


def plot_risk_coverage(bundle: dict, out_path: Path) -> None:
    """Selective-prediction curves; annotate accuracy at 50% coverage."""
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 4)
    fig, ax = plt.subplots(figsize=(9, 6))
    accs_at_50: dict[str, float] = {}
    for key, info in bundle.items():
        accs = risk_coverage_curve(info["y_true"], info["y_pred"], info["y_prob"], coverages)
        ax.plot(coverages, accs, marker=MODEL_MARKERS[key], color=info["color"],
                label=info["label"])
        overall = accuracy(info["y_true"], info["y_pred"])
        ax.axhline(overall, color=info["color"], linestyle="--", alpha=0.3)
        accs_at_50[key] = float(np.interp(0.5, coverages, accs))
    ax.set_xlabel("Coverage (fraction of samples retained)")
    ax.set_ylabel("Accuracy on retained samples")
    ax.set_title("Selective prediction — accuracy vs coverage fraction")
    ax.set_xlim(0.08, 1.02)
    ax.legend(loc="lower left")
    text_lines = [
        f"{bundle[k]['label']} @50%: {accs_at_50[k]:.3f}" for k in bundle
    ]
    ax.text(
        0.98, 0.97, "\n".join(text_lines), transform=ax.transAxes,
        va="top", ha="right", fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
    )
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def per_class_ece_filtered(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_bins: int = 10, min_per_bin: int = 20,
) -> float:
    """ECE on this class's samples, skipping bins with < ``min_per_bin`` items.

    Returns ``np.nan`` if every bin is below the threshold.
    """
    if y_true.size == 0:
        return float("nan")
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_idx = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, n_bins - 1)

    weighted_gap = 0.0
    total = 0
    for b in range(n_bins):
        mask = bin_idx == b
        n = int(mask.sum())
        if n < min_per_bin:
            continue
        gap = abs(correct[mask].mean() - confidences[mask].mean())
        weighted_gap += n * gap
        total += n
    if total == 0:
        return float("nan")
    return float(weighted_gap / total)


def plot_per_class_ece(bundle: dict, out_path: Path) -> None:
    """3×10 ECE heatmap with NaN-tolerant annotation."""
    keys = list(bundle.keys())
    matrix = np.full((len(keys), 10), np.nan)
    for r, k in enumerate(keys):
        info = bundle[k]
        for c in range(10):
            class_mask = info["y_true"] == c
            matrix[r, c] = per_class_ece_filtered(
                info["y_true"][class_mask],
                info["y_prob"][class_mask],
                n_bins=10, min_per_bin=20,
            )
    vmax = np.nanmax(matrix) if np.any(~np.isnan(matrix)) else 1.0
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(matrix, cmap="YlOrRd", vmin=0.0, vmax=vmax, aspect="auto")
    fig.colorbar(im, ax=ax, label="ECE")
    ax.set_xticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(len(keys)))
    ax.set_yticklabels([bundle[k]["label"] for k in keys])
    ax.set_title("Per-class ECE — all three models")
    for r in range(len(keys)):
        for c in range(10):
            v = matrix[r, c]
            if np.isnan(v):
                text = "—"
            else:
                text = f"{v:.2f}"
            color = "white" if (not np.isnan(v) and v > vmax * 0.6) else "black"
            ax.text(c, r, text, ha="center", va="center", color=color, fontsize=9)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def accuracy_after_abstention(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, abstain_rate: float,
) -> float:
    """Accuracy after dropping the ``abstain_rate`` least-confident samples."""
    if abstain_rate <= 0:
        return accuracy(y_true, y_pred)
    confidence = y_prob.max(axis=1)
    n = y_true.size
    keep = n - int(round(abstain_rate * n))
    keep = max(1, keep)
    order = np.argsort(-confidence)[:keep]
    return accuracy(y_true[order], y_pred[order])


def plot_abstention_gain(bundle: dict, out_path: Path) -> None:
    """Grouped bars of accuracy at 0/10/20/30% abstention."""
    abstain_rates = (0.0, 0.10, 0.20, 0.30)
    keys = list(bundle.keys())
    accs = {
        k: [accuracy_after_abstention(
                bundle[k]["y_true"], bundle[k]["y_pred"], bundle[k]["y_prob"], r,
            ) for r in abstain_rates]
        for k in keys
    }
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(abstain_rates))
    width = 0.25
    for i, k in enumerate(keys):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, accs[k], width=width,
                      color=bundle[k]["color"], label=bundle[k]["label"])
        for b, v in zip(bars, accs[k]):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(r * 100)}%" for r in abstain_rates])
    ax.set_xlabel("Abstention rate (samples dropped by lowest confidence)")
    ax.set_ylabel("Accuracy on retained samples")
    ax.set_title("Accuracy gain from uncertainty-based abstention")
    ax.set_ylim(0, min(1.0, max(max(v) for v in accs.values()) + 0.08))
    ax.legend(loc="upper left")
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_confidence_distributions(bundle: dict, out_path: Path) -> None:
    """1×3 step-histogram panels of confidence split by correctness."""
    keys = list(bundle.keys())
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5))
    for ax, k in zip(axes, keys):
        info = bundle[k]
        conf = info["y_prob"].max(axis=1)
        correct = info["y_pred"] == info["y_true"]
        bins = np.linspace(0, 1, 31)
        ax.hist(conf[correct], bins=bins, histtype="step", linewidth=2.0,
                color=info["color"], alpha=0.85,
                label=f"correct (n={int(correct.sum())})")
        ax.hist(conf[~correct], bins=bins, histtype="step", linewidth=2.0,
                color="firebrick", alpha=0.85,
                label=f"incorrect (n={int((~correct).sum())})")
        mu_c = float(conf[correct].mean()) if correct.any() else float("nan")
        mu_i = float(conf[~correct].mean()) if (~correct).any() else float("nan")
        ax.axvline(mu_c, color=info["color"], linestyle="--", linewidth=1.2)
        ax.axvline(mu_i, color="firebrick", linestyle="--", linewidth=1.2)
        ax.set_xlim(0, 1)
        ax.set_xlabel("Max predicted probability (confidence)")
        ax.set_ylabel("Count")
        acc = accuracy(info["y_true"], info["y_pred"])
        ax.set_title(f"{info['label']}  (acc={acc:.3f})")
        ax.text(
            0.02, 0.97,
            f"μ correct   = {mu_c:.3f}\nμ incorrect = {mu_i:.3f}\nΔ          = {mu_c - mu_i:+.3f}",
            transform=ax.transAxes, va="top", ha="left", fontsize=9,
            family="monospace",
            bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "gray"},
        )
        ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Confidence distribution — correct vs incorrect predictions", y=1.02)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def log_summary_table(bundle: dict) -> None:
    """Print a five-row ASCII table comparing all three models."""
    coverages = np.round(np.arange(0.10, 1.001, 0.05), 4)
    rows: list[tuple[str, dict[str, float]]] = []
    accs = {k: accuracy(bundle[k]["y_true"], bundle[k]["y_pred"]) for k in bundle}
    eces = {k: expected_calibration_error(bundle[k]["y_true"], bundle[k]["y_prob"]) for k in bundle}
    rc = {k: risk_coverage_curve(bundle[k]["y_true"], bundle[k]["y_pred"], bundle[k]["y_prob"], coverages) for k in bundle}
    acc50 = {k: float(np.interp(0.50, coverages, rc[k])) for k in bundle}
    acc80 = {k: float(np.interp(0.80, coverages, rc[k])) for k in bundle}
    acc_at_20_abst = {
        k: accuracy_after_abstention(
            bundle[k]["y_true"], bundle[k]["y_pred"], bundle[k]["y_prob"], 0.20,
        ) for k in bundle
    }
    gain_at_20_abst = {k: acc_at_20_abst[k] - accs[k] for k in bundle}

    rows.append(("Accuracy",            accs))
    rows.append(("ECE",                 eces))
    rows.append(("Accuracy @ 50% cov",  acc50))
    rows.append(("Accuracy @ 80% cov",  acc80))
    rows.append(("Acc gain @ 20% abst", gain_at_20_abst))

    header = f"{'Metric':<22} {'LR':>8} {'RF':>8} {'CNN':>8}"
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for name, values in rows:
        logger.info(
            "%-22s %8.4f %8.4f %8.4f",
            name, values["lr"], values["rf"], values["cnn"],
        )
    logger.info(sep)


def main() -> None:
    """Load predictions, generate five figures, log the summary table."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    figures_dir = Path("results") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    bundle = load_predictions()

    plot_reliability_comparison(bundle, figures_dir / "05b_reliability_comparison.png")
    logger.info("Wrote 05b_reliability_comparison.png")
    plot_risk_coverage(bundle, figures_dir / "05b_risk_coverage.png")
    logger.info("Wrote 05b_risk_coverage.png")
    plot_per_class_ece(bundle, figures_dir / "05b_per_class_ece.png")
    logger.info("Wrote 05b_per_class_ece.png")
    plot_abstention_gain(bundle, figures_dir / "05b_abstention_gain.png")
    logger.info("Wrote 05b_abstention_gain.png")
    plot_confidence_distributions(bundle, figures_dir / "05b_confidence_distributions.png")
    logger.info("Wrote 05b_confidence_distributions.png")

    log_summary_table(bundle)


if __name__ == "__main__":
    main()
