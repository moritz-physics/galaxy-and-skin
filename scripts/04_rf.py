"""Random Forest baseline with nested cross-validation.

Runs the same nested CV harness used by the logistic-regression baseline, then
fits one final forest on the entire dataset with the most-commonly-selected
hyperparameters to expose feature importances.

Run from the project root::

    uv run python scripts/04_rf.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.cv import nested_cv  # noqa: E402
from galaxy_uq.data import CLASS_NAMES  # noqa: E402
from galaxy_uq.metrics import (  # noqa: E402
    accuracy,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
    per_class_precision_recall_f1,
    reliability_diagram_data,
)
from galaxy_uq.models.rf import RandomForestModel  # noqa: E402

logger = logging.getLogger("04_rf")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}

N_ESTIMATORS_GRID: tuple[int, ...] = (50, 200, 500)
MAX_DEPTH_GRID: tuple[int | None, ...] = (10, 30, None)


def build_hp_grid() -> list[dict]:
    """Cartesian product of ``n_estimators`` × ``max_depth`` constructor kwargs."""
    grid: list[dict] = []
    for ne in N_ESTIMATORS_GRID:
        for md in MAX_DEPTH_GRID:
            grid.append(
                {
                    "n_estimators": ne,
                    "max_depth": md,
                    "min_samples_leaf": 1,
                    "class_weight": "balanced",
                    "seed": 42,
                }
            )
    return grid


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Save a row-normalised confusion matrix annotated with percentages."""
    cm = confusion_matrix(y_true, y_pred, n_classes=10).astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    cm_norm = cm / row_sums

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="Row-normalised proportion")
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.set_xticklabels(CLASS_NAMES, rotation=40, ha="right", fontsize=8)
    ax.set_yticklabels(CLASS_NAMES, fontsize=8)
    ax.set_xlabel("Predicted class")
    ax.set_ylabel("True class")
    ax.set_title("Random Forest — row-normalised confusion matrix")
    for i in range(10):
        for j in range(10):
            v = cm_norm[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center",
                    color=color, fontsize=7)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_hp_selection(outer_fold_results: list[dict], out_path: Path) -> None:
    """Two-panel bar chart: counts of selected ``n_estimators`` and ``max_depth``."""
    ne_counts = Counter(r["best_hyperparams"]["n_estimators"] for r in outer_fold_results)
    md_counts = Counter(
        ("None" if r["best_hyperparams"]["max_depth"] is None
         else r["best_hyperparams"]["max_depth"])
        for r in outer_fold_results
    )

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ne_x = [str(v) for v in N_ESTIMATORS_GRID]
    ne_y = [ne_counts.get(v, 0) for v in N_ESTIMATORS_GRID]
    axes[0].bar(ne_x, ne_y, color="steelblue")
    for x, y in zip(ne_x, ne_y):
        axes[0].text(x, y + 0.05, str(y), ha="center", va="bottom")
    axes[0].set_xlabel("n_estimators")
    axes[0].set_ylabel("Outer folds selecting value")
    axes[0].set_ylim(0, max(ne_y) + 1.5)

    md_labels = [("None" if v is None else str(v)) for v in MAX_DEPTH_GRID]
    md_y = [md_counts.get(lbl, 0) for lbl in md_labels]
    axes[1].bar(md_labels, md_y, color="seagreen")
    for x, y in zip(md_labels, md_y):
        axes[1].text(x, y + 0.05, str(y), ha="center", va="bottom")
    axes[1].set_xlabel("max_depth")
    axes[1].set_ylabel("Outer folds selecting value")
    axes[1].set_ylim(0, max(md_y) + 1.5)

    fig.suptitle("RF HP selection — check middle values are most common")
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_reliability(y_true: np.ndarray, y_prob: np.ndarray, out_path: Path) -> None:
    """Reliability diagram with diagonal reference and ECE in the title."""
    data = reliability_diagram_data(y_true, y_prob, n_bins=15)
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfect calibration")
    mask = ~np.isnan(data["bin_accuracies"])
    ax.plot(
        data["bin_confidences"][mask],
        data["bin_accuracies"][mask],
        marker="o",
        color="seagreen",
        label="Random Forest",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence in bin")
    ax.set_ylabel("Empirical accuracy in bin")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Horizontal bar chart of per-class F1 with the worst class in red."""
    stats = per_class_precision_recall_f1(y_true, y_pred, n_classes=10)
    f1 = stats["f1"]
    worst = int(np.argmin(f1))
    colors = ["firebrick" if i == worst else "seagreen" for i in range(10)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(10), f1, color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.set_title("Random Forest — per-class F1 (lowest in red)")
    for i, v in enumerate(f1):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_feature_importance(
    importances: np.ndarray, feature_names: list[str], out_path: Path,
    top_k: int = 20,
) -> None:
    """Top-``k`` impurity-based feature importances as a horizontal bar chart."""
    order = np.argsort(importances)[::-1][:top_k]
    names = [feature_names[i] for i in order]
    values = importances[order]
    fig, ax = plt.subplots(figsize=(9, 8))
    ax.barh(range(top_k), values, color="seagreen")
    ax.set_yticks(range(top_k))
    ax.set_yticklabels(names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Mean impurity decrease")
    ax.set_title(f"Random Forest — top {top_k} feature importances")
    for i, v in enumerate(values):
        ax.text(v + values.max() * 0.005, i, f"{v:.3f}", va="center", fontsize=8)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def most_common_hyperparams(outer_fold_results: list[dict]) -> dict:
    """Per-parameter mode of the HPs chosen across outer folds.

    Per-key mode (rather than mode of full dicts) lets us recover a stable
    "centre" of the grid even when no single combination dominates.
    """
    keys = list(outer_fold_results[0]["best_hyperparams"].keys())
    chosen: dict = {}
    for k in keys:
        counts = Counter(r["best_hyperparams"][k] for r in outer_fold_results)
        chosen[k] = counts.most_common(1)[0][0]
    return chosen


def _to_jsonable(obj):
    """Convert numpy / nested containers to JSON-compatible types."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def log_deltas_vs_logreg(agg: dict) -> None:
    """Log improvement vs the previously saved logistic-regression aggregates."""
    lr_path = Path("results") / "metrics" / "03_logreg_results.json"
    if not lr_path.exists():
        logger.warning("Logistic regression results not found at %s; skipping deltas",
                       lr_path)
        return
    lr_agg = json.loads(lr_path.read_text())["aggregate_metrics"]
    for key, label in (("accuracy", "accuracy"), ("macro_f1", "macro F1"),
                       ("ece", "ECE"), ("nll", "NLL")):
        delta = agg[key]["mean"] - lr_agg[key]["mean"]
        sign = "+" if delta >= 0 else ""
        logger.info(
            "  %-9s LR=%.4f  RF=%.4f  Δ=%s%.4f",
            label, lr_agg[key]["mean"], agg[key]["mean"], sign, delta,
        )


def main() -> None:
    """Run nested CV, emit metrics + figures, fit a final forest for importances."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    figures_dir = Path("results") / "figures"
    metrics_dir = Path("results") / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(Path("data") / "processed" / "features.npz")
    feature_matrix = data["features"].astype(np.float64)
    labels = data["labels"].astype(np.int64)
    feature_names = [str(n) for n in data["feature_names"]]
    logger.info("Loaded features: X=%s y=%s", feature_matrix.shape, labels.shape)

    hp_grid = build_hp_grid()
    logger.info("HP grid: %d combinations", len(hp_grid))

    start = time.perf_counter()
    results = nested_cv(
        X=feature_matrix,
        y=labels,
        model_factory=RandomForestModel,
        hyperparam_grid=hp_grid,
        outer_k=10,
        inner_k=3,
        seed=42,
        n_classes=10,
    )
    elapsed = time.perf_counter() - start
    logger.info("Nested CV finished in %.1f s (%.1f min)", elapsed, elapsed / 60.0)

    agg = results["aggregate_metrics"]
    logger.info(
        "Aggregate: accuracy=%.4f±%.4f macro_f1=%.4f±%.4f ECE=%.4f NLL=%.4f",
        agg["accuracy"]["mean"], agg["accuracy"]["std"],
        agg["macro_f1"]["mean"], agg["macro_f1"]["std"],
        agg["ece"]["mean"], agg["nll"]["mean"],
    )
    logger.info("Improvement vs logistic regression:")
    log_deltas_vs_logreg(agg)

    # Persist results.
    json_payload = {
        "wall_clock_seconds": round(elapsed, 2),
        "hyperparam_grid": hp_grid,
        "aggregate_metrics": agg,
        "outer_fold_results": [
            {
                "fold_idx": r["fold_idx"],
                "best_hyperparams": r["best_hyperparams"],
                "test_indices": r["test_indices"],
                "inner_scores": r["inner_scores"],
            }
            for r in results["outer_fold_results"]
        ],
    }
    json_path = metrics_dir / "04_rf_results.json"
    json_path.write_text(json.dumps(_to_jsonable(json_payload), indent=2))
    logger.info("Wrote %s", json_path)

    pred_path = metrics_dir / "04_rf_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_true=results["all_y_true"],
        y_pred=results["all_y_pred"],
        y_prob=results["all_y_prob"],
    )
    logger.info("Wrote %s", pred_path)

    # Plots driven by the nested-CV holdout predictions.
    plot_confusion(
        results["all_y_true"], results["all_y_pred"],
        figures_dir / "04_rf_confusion.png",
    )
    plot_hp_selection(
        results["outer_fold_results"], figures_dir / "04_rf_hp_selection.png"
    )
    plot_reliability(
        results["all_y_true"], results["all_y_prob"],
        figures_dir / "04_rf_reliability.png",
    )
    plot_per_class_f1(
        results["all_y_true"], results["all_y_pred"],
        figures_dir / "04_rf_per_class_f1.png",
    )

    # Final forest on the full dataset for feature importances. We
    # standardise here for parity with the per-fold pipeline even though tree
    # splits are scale-invariant.
    final_hp = most_common_hyperparams(results["outer_fold_results"])
    logger.info("Fitting final RF on all data with HPs: %s", final_hp)
    mean = feature_matrix.mean(axis=0)
    std = feature_matrix.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_full_s = (feature_matrix - mean) / std
    final_rf = RandomForestModel(**final_hp)
    final_rf.fit(X_full_s, labels)
    plot_feature_importance(
        final_rf.feature_importances_, feature_names,
        figures_dir / "04_rf_feature_importance.png",
    )

    final_acc = accuracy(results["all_y_true"], results["all_y_pred"])
    final_f1 = macro_f1(results["all_y_true"], results["all_y_pred"], 10)
    final_ece = expected_calibration_error(results["all_y_true"], results["all_y_prob"])
    final_nll = negative_log_likelihood(results["all_y_true"], results["all_y_prob"])
    logger.info(
        "Concatenated holdout metrics: acc=%.4f macro_f1=%.4f ECE=%.4f NLL=%.4f",
        final_acc, final_f1, final_ece, final_nll,
    )


if __name__ == "__main__":
    main()
