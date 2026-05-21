"""Post-hoc calibration: temperature scaling for LR/CNN, isotonic for RF.

Loads the saved nested-CV predictions, splits them into a 20% stratified
calibration set and an 80% evaluation set, fits a one-parameter temperature
(LR, CNN) or a PAVA isotonic regression (RF) on the calibration set, and
reports ECE before and after on the evaluation set.

Run from the project root::

    uv run python scripts/05c_temperature_scaling.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize_scalar

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.cv import stratified_kfold_split  # noqa: E402
from galaxy_uq.metrics import (  # noqa: E402
    expected_calibration_error,
    reliability_diagram_data,
)

logger = logging.getLogger("05c_calibration")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}
_EPS = 1e-12

MODELS: list[tuple[str, str, str, str, str]] = [
    # key, label, path, colour, method
    ("lr",  "Logistic regression", "results/metrics/03_logreg_predictions.npz", "tab:blue", "temperature"),
    ("rf",  "Random forest",       "results/metrics/04_rf_predictions.npz",    "seagreen", "isotonic"),
    ("cnn", "Deep ensemble CNN",   "results/metrics/05_cnn_predictions.npz",    "purple",   "temperature"),
]


def stratified_holdout(
    y: np.ndarray, fraction: float = 0.2, seed: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """Return (evaluation_indices_80%, calibration_indices_20%) by stratified split."""
    k = int(round(1.0 / fraction))
    folds = stratified_kfold_split(y, k=k, seed=seed)
    eval_idx, calib_idx = folds[0]
    return eval_idx, calib_idx


def softmax_with_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    """Recompose ``softmax(log_prob / T)`` from a probability matrix.

    Working in log-space recovers pseudo-logits up to a per-row constant,
    which softmax is invariant to — so this is equivalent to applying
    temperature to the original logits.
    """
    log_p = np.log(np.clip(probs, _EPS, 1.0))
    scaled = log_p / T
    scaled = scaled - scaled.max(axis=1, keepdims=True)
    exp = np.exp(scaled)
    return exp / exp.sum(axis=1, keepdims=True)


def fit_temperature(
    calib_probs: np.ndarray, calib_y: np.ndarray,
) -> float:
    """Find ``T ∈ [0.1, 10.0]`` minimising calibration-set NLL."""

    def nll(T: float) -> float:
        scaled = softmax_with_temperature(calib_probs, T)
        chosen = scaled[np.arange(calib_y.size), calib_y]
        return float(-np.mean(np.log(np.clip(chosen, _EPS, 1.0))))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def pava(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool Adjacent Violators on ``(x, y)`` pairs.

    Returns the ``x`` values sorted ascending and the isotonic non-decreasing
    fit ``y_hat`` at those locations. Implementation uses a block stack:
    whenever a new value violates monotonicity, merge it with the previous
    block; cascading merges until the stack is monotonic.
    """
    order = np.argsort(x)
    xs = x[order]
    ys = y[order].astype(np.float64)

    means: list[float] = []
    counts: list[int] = []
    for v in ys:
        means.append(float(v))
        counts.append(1)
        while len(means) >= 2 and means[-2] > means[-1]:
            c1 = counts.pop(); m1 = means.pop()
            c2 = counts.pop(); m2 = means.pop()
            new_count = c1 + c2
            new_mean = (m1 * c1 + m2 * c2) / new_count
            means.append(new_mean)
            counts.append(new_count)

    y_hat = np.empty(ys.size, dtype=np.float64)
    pos = 0
    for m, c in zip(means, counts):
        y_hat[pos : pos + c] = m
        pos += c
    return xs, y_hat


def apply_isotonic(
    new_conf: np.ndarray, xs: np.ndarray, y_hat: np.ndarray
) -> np.ndarray:
    """Piecewise-constant lookup of the isotonic fit at ``new_conf``."""
    idx = np.searchsorted(xs, new_conf, side="right") - 1
    idx = np.clip(idx, 0, y_hat.size - 1)
    return y_hat[idx]


def fit_isotonic_from_probs(
    calib_probs: np.ndarray, calib_y: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fit ``confidence → empirical accuracy`` via PAVA on the calibration set."""
    confidence = calib_probs.max(axis=1)
    correct = (calib_probs.argmax(axis=1) == calib_y).astype(np.float64)
    return pava(confidence, correct)


def apply_isotonic_to_probs(
    probs: np.ndarray, xs: np.ndarray, y_hat: np.ndarray,
) -> np.ndarray:
    """Replace each row's max with the isotonic-calibrated confidence.

    The remaining ``K - 1`` probabilities are rescaled to sum to
    ``1 - calibrated_confidence``, preserving their relative ratios. For
    ECE — which depends only on the top probability — this is what matters.
    """
    n, k = probs.shape
    pred = probs.argmax(axis=1)
    raw_conf = probs.max(axis=1)
    new_conf = apply_isotonic(raw_conf, xs, y_hat)

    rest_sum = 1.0 - raw_conf
    safe_rest = np.where(rest_sum < _EPS, 1.0, rest_sum)
    scale = (1.0 - new_conf) / safe_rest

    new_probs = probs * scale[:, None]
    new_probs[np.arange(n), pred] = new_conf
    # Guard against any numerical underflow leaving rows that don't sum to 1.
    row_sums = new_probs.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums < _EPS, 1.0, row_sums)
    return new_probs / row_sums


def plot_reliability_panel(
    ax: plt.Axes, y_true: np.ndarray, y_prob: np.ndarray,
    color: str, panel_title: str,
) -> float:
    """Draw a single reliability curve; return the ECE shown in the title."""
    data = reliability_diagram_data(y_true, y_prob, n_bins=15)
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    ax.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1.0)
    mask = ~np.isnan(data["bin_accuracies"])
    ax.plot(
        data["bin_confidences"][mask], data["bin_accuracies"][mask],
        marker="o", color=color,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean confidence")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{panel_title}  (ECE = {ece:.3f})")
    return float(ece)


def main() -> None:
    """Calibrate each model post-hoc and emit a comparison figure."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    figures_dir = Path("results") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Load predictions; assume all three share y_true (already checked in 05b).
    bundles: dict = {}
    for key, label, path, color, method in MODELS:
        d = np.load(Path(path))
        bundles[key] = {
            "label": label, "color": color, "method": method,
            "y_true": d["y_true"], "y_prob": d["y_prob"],
        }
    y_true = bundles["lr"]["y_true"]
    eval_idx, calib_idx = stratified_holdout(y_true, fraction=0.2, seed=42)
    logger.info(
        "Calibration split: calib=%d (20%%), eval=%d (80%%)",
        calib_idx.size, eval_idx.size,
    )

    rows: list[tuple[str, float, str, float]] = []
    fig, axes = plt.subplots(3, 2, figsize=(13, 14))

    for row, (key, info) in enumerate(bundles.items()):
        y_true_full = info["y_true"]
        y_prob_full = info["y_prob"]
        calib_probs = y_prob_full[calib_idx]
        calib_y = y_true_full[calib_idx]
        eval_probs = y_prob_full[eval_idx]
        eval_y = y_true_full[eval_idx]

        # Before
        ece_before = plot_reliability_panel(
            axes[row, 0], eval_y, eval_probs, info["color"],
            f"{info['label']} — before",
        )

        # Fit + apply
        if info["method"] == "temperature":
            T = fit_temperature(calib_probs, calib_y)
            method_label = f"T={T:.3f}"
            eval_probs_cal = softmax_with_temperature(eval_probs, T)
        elif info["method"] == "isotonic":
            xs, y_hat = fit_isotonic_from_probs(calib_probs, calib_y)
            method_label = "isotonic (PAVA)"
            eval_probs_cal = apply_isotonic_to_probs(eval_probs, xs, y_hat)
        else:  # pragma: no cover
            raise ValueError(f"unknown calibration method: {info['method']}")

        ece_after = plot_reliability_panel(
            axes[row, 1], eval_y, eval_probs_cal, info["color"],
            f"{info['label']} — after ({method_label})",
        )
        rows.append((info["label"], ece_before, method_label, ece_after))

    fig.suptitle("Post-hoc calibration — reliability diagrams before vs after", y=1.005)
    fig.tight_layout()
    out_path = figures_dir / "05c_calibration_comparison.png"
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)
    logger.info("Wrote %s", out_path)

    # Summary table.
    header = f"{'Model':<22} {'ECE before':>11} {'Method':>18} {'ECE after':>10} {'Δ':>9}"
    sep = "-" * len(header)
    logger.info(sep)
    logger.info(header)
    logger.info(sep)
    for label, before, method, after in rows:
        logger.info(
            "%-22s %11.4f %18s %10.4f %+9.4f",
            label, before, method, after, after - before,
        )
    logger.info(sep)


if __name__ == "__main__":
    main()
