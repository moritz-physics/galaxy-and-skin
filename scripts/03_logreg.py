"""Logistic regression baseline with nested cross-validation.

The script first verifies the from-scratch :class:`MultinomialLogisticRegression`
against scikit-learn on a stratified 2000-sample subset, then runs nested CV
on the full feature matrix. Results, plots and predictions are written
under ``results/``.

Run from the project root::

    uv run python scripts/03_logreg.py
"""

from __future__ import annotations

import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.cv import nested_cv  # noqa: E402
from galaxy_uq.data import CLASS_NAMES, stratified_subset  # noqa: E402
from galaxy_uq.metrics import (  # noqa: E402
    accuracy,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
    per_class_precision_recall_f1,
    reliability_diagram_data,
)
from galaxy_uq.models.logreg import MultinomialLogisticRegression  # noqa: E402

logger = logging.getLogger("03_logreg")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}

HP_GRID: list[dict] = [
    {"C": 0.01, "lr": 0.05, "max_iter": 1000, "tol": 1e-6},
    {"C": 0.1, "lr": 0.05, "max_iter": 1000, "tol": 1e-6},
    {"C": 1.0, "lr": 0.05, "max_iter": 1000, "tol": 1e-6},
    {"C": 10.0, "lr": 0.05, "max_iter": 1000, "tol": 1e-6},
    {"C": 100.0, "lr": 0.05, "max_iter": 1000, "tol": 1e-6},
]


def verify_against_sklearn(
    feature_matrix: np.ndarray, labels: np.ndarray, seed: int = 0
) -> None:
    """Compare the from-scratch LR + metric helpers against scikit-learn.

    Raises
    ------
    AssertionError
        If accuracies diverge by more than 5 percentage points or if metric
        helpers disagree with their sklearn counterparts.
    """
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix as sk_confusion_matrix,
        f1_score,
    )

    rng = np.random.default_rng(seed)
    subset_idx = stratified_subset(labels, n_per_class=200, seed=seed)
    X = feature_matrix[subset_idx]
    y = labels[subset_idx]

    # Standardise the entire subset; this is the only place we ever do so
    # outside fold-local standardisation. It is fine here because we are
    # cross-checking implementations, not estimating generalisation.
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    X_std = (X - mean) / std

    perm = rng.permutation(X_std.shape[0])
    n_train = int(0.8 * X_std.shape[0])
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    ours = MultinomialLogisticRegression(C=1.0, lr=0.01, max_iter=2000, tol=1e-8)
    ours.fit(X_std[train_idx], y[train_idx])
    y_pred_ours = ours.predict(X_std[test_idx])
    ours_acc = accuracy(y[test_idx], y_pred_ours)

    try:
        sk = LogisticRegression(
            C=1.0, max_iter=2000, multi_class="multinomial", solver="lbfgs"
        )
    except TypeError:
        # scikit-learn >= 1.7 removed the multi_class kwarg; multinomial is
        # now the default behaviour for multi-class targets with lbfgs.
        sk = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
    sk.fit(X_std[train_idx], y[train_idx])
    y_pred_sk = sk.predict(X_std[test_idx])
    sk_acc = accuracy_score(y[test_idx], y_pred_sk)

    logger.info(
        "Verification: from-scratch acc=%.4f sklearn acc=%.4f diff=%.4f",
        ours_acc,
        sk_acc,
        abs(ours_acc - sk_acc),
    )
    if abs(ours_acc - sk_acc) > 0.05:
        raise AssertionError(
            "from-scratch LR accuracy diverges from sklearn by more than 5pp: "
            f"ours={ours_acc:.4f} sklearn={sk_acc:.4f}"
        )

    # Cross-check metric helpers against sklearn for the same predictions.
    cm_ours = confusion_matrix(y[test_idx], y_pred_sk, n_classes=10)
    cm_sk = sk_confusion_matrix(y[test_idx], y_pred_sk, labels=list(range(10)))
    if not np.array_equal(cm_ours, cm_sk):
        raise AssertionError("from-scratch confusion matrix disagrees with sklearn")

    acc_ours = accuracy(y[test_idx], y_pred_sk)
    acc_sk = accuracy_score(y[test_idx], y_pred_sk)
    if not np.isclose(acc_ours, acc_sk):
        raise AssertionError(
            f"accuracy mismatch: ours={acc_ours} sklearn={acc_sk}"
        )

    f1_ours = macro_f1(y[test_idx], y_pred_sk, n_classes=10)
    f1_sk = f1_score(y[test_idx], y_pred_sk, average="macro", labels=list(range(10)))
    if not np.isclose(f1_ours, f1_sk):
        raise AssertionError(f"macro F1 mismatch: ours={f1_ours} sklearn={f1_sk}")
    logger.info("Metric helpers match sklearn exactly.")


def plot_confusion(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path
) -> None:
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
    ax.set_title("Logistic regression — row-normalised confusion matrix")
    for i in range(10):
        for j in range(10):
            value = cm_norm[i, j]
            color = "white" if value > 0.5 else "black"
            ax.text(
                j, i, f"{value * 100:.0f}%", ha="center", va="center",
                color=color, fontsize=7,
            )
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_hp_selection(outer_fold_results: list[dict], out_path: Path) -> None:
    """Bar chart of how often each ``C`` is picked across outer folds."""
    c_values = [hp["C"] for hp in HP_GRID]
    counts = {c: 0 for c in c_values}
    for r in outer_fold_results:
        counts[r["best_hyperparams"]["C"]] += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [str(c) for c in c_values]
    ys = [counts[c] for c in c_values]
    ax.bar(xs, ys, color="steelblue")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.05, str(y), ha="center", va="bottom")
    ax.set_xlabel("C (inverse regularisation strength)")
    ax.set_ylabel("Number of outer folds selecting this C")
    ax.set_title("HP selection frequency — check middle is most common")
    ax.set_ylim(0, max(ys) + 1.5)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_reliability(
    y_true: np.ndarray, y_prob: np.ndarray, out_path: Path
) -> None:
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
        color="steelblue",
        label="Logistic regression",
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Mean predicted confidence in bin")
    ax.set_ylabel("Empirical accuracy in bin")
    ax.set_title(f"Reliability diagram (ECE = {ece:.3f})")
    ax.legend(loc="upper left")
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_per_class_f1(
    y_true: np.ndarray, y_pred: np.ndarray, out_path: Path
) -> None:
    """Horizontal bar chart of per-class F1 with the worst class in red."""
    stats = per_class_precision_recall_f1(y_true, y_pred, n_classes=10)
    f1 = stats["f1"]
    worst = int(np.argmin(f1))
    colors = ["firebrick" if i == worst else "steelblue" for i in range(10)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(10), f1, color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.set_title("Logistic regression — per-class F1 (lowest in red)")
    for i, v in enumerate(f1):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def _to_jsonable(obj):
    """Recursively convert numpy arrays / scalars for json.dump."""
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


def main() -> None:
    """Run verification, nested CV, and emit metrics + figures."""
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
    logger.info(
        "Loaded features: X=%s y=%s", feature_matrix.shape, labels.shape
    )

    logger.info("Step 1/2: verifying from-scratch LR against sklearn")
    verify_against_sklearn(feature_matrix, labels, seed=0)

    logger.info("Step 2/2: nested cross-validation on the full dataset")
    start = time.perf_counter()
    results = nested_cv(
        X=feature_matrix,
        y=labels,
        model_factory=MultinomialLogisticRegression,
        hyperparam_grid=HP_GRID,
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

    # Persist results as JSON (arrays → lists).
    json_payload = {
        "wall_clock_seconds": round(elapsed, 2),
        "hyperparam_grid": HP_GRID,
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
    json_path = metrics_dir / "03_logreg_results.json"
    json_path.write_text(json.dumps(_to_jsonable(json_payload), indent=2))
    logger.info("Wrote %s", json_path)

    # Persist predictions for downstream comparisons.
    pred_path = metrics_dir / "03_logreg_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_true=results["all_y_true"],
        y_pred=results["all_y_pred"],
        y_prob=results["all_y_prob"],
    )
    logger.info("Wrote %s", pred_path)

    # Plots.
    plot_confusion(
        results["all_y_true"], results["all_y_pred"],
        figures_dir / "03_logreg_confusion.png",
    )
    plot_hp_selection(
        results["outer_fold_results"], figures_dir / "03_logreg_hp_selection.png"
    )
    plot_reliability(
        results["all_y_true"], results["all_y_prob"],
        figures_dir / "03_logreg_reliability.png",
    )
    plot_per_class_f1(
        results["all_y_true"], results["all_y_pred"],
        figures_dir / "03_logreg_per_class_f1.png",
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
