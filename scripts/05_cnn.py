"""Deep-ensemble CNN baseline with nested cross-validation.

Same outer/inner fold splits (seed=42, k=10, inner_k=3) as the logistic
regression and random forest baselines, so the three models can be compared
fold-for-fold. Inner CV picks the learning rate using a *single* CNN per
combination for speed; the outer evaluation then trains the full ensemble
with the selected learning rate.

Run from the project root::

    # Smoke test (32×32, 5 epochs, M=2)
    uv run python scripts/05_cnn.py --fast

    # Full pipeline (64×64, 20 epochs, M=5)
    uv run python scripts/05_cnn.py
"""

from __future__ import annotations

import argparse
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

from galaxy_uq.cv import stratified_kfold_split  # noqa: E402
from galaxy_uq.data import CLASS_NAMES, load_galaxy10  # noqa: E402
from galaxy_uq.metrics import (  # noqa: E402
    accuracy,
    confusion_matrix,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
    per_class_precision_recall_f1,
    reliability_diagram_data,
)
from galaxy_uq.models.cnn import DeepEnsemble, get_device  # noqa: E402
from galaxy_uq.preprocessing import prepare_images_for_cnn  # noqa: E402

logger = logging.getLogger("05_cnn")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}

LR_GRID: tuple[float, ...] = (0.0003, 0.001, 0.003)
HP_GRID: list[dict] = [{"lr": lr, "weight_decay": 1e-4} for lr in LR_GRID]


def parse_args() -> argparse.Namespace:
    """Command-line interface; ``--fast`` overrides the heavy defaults."""
    p = argparse.ArgumentParser(description="Galaxy10 deep-ensemble nested CV")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--img-size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--ensemble-members", type=int, default=5)
    p.add_argument(
        "--out-dir", type=Path, default=Path("results") / "figures",
        help="directory for figures",
    )
    p.add_argument(
        "--fast", action="store_true",
        help="smoke-test mode: img_size=32, epochs=5, ensemble_members=2",
    )
    return p.parse_args()


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out_path: Path) -> None:
    """Row-normalised confusion matrix annotated with percentages."""
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
    ax.set_title("Deep ensemble CNN — row-normalised confusion matrix")
    for i in range(10):
        for j in range(10):
            v = cm_norm[i, j]
            color = "white" if v > 0.5 else "black"
            ax.text(j, i, f"{v * 100:.0f}%", ha="center", va="center",
                    color=color, fontsize=7)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_hp_selection(fold_records: list[dict], out_path: Path) -> None:
    """Bar chart of learning-rate selections across outer folds."""
    counts = {lr: 0 for lr in LR_GRID}
    for r in fold_records:
        counts[r["best_hyperparams"]["lr"]] += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    xs = [str(lr) for lr in LR_GRID]
    ys = [counts[lr] for lr in LR_GRID]
    ax.bar(xs, ys, color="purple")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.05, str(y), ha="center", va="bottom")
    ax.set_xlabel("learning rate")
    ax.set_ylabel("Outer folds selecting value")
    ax.set_title("Deep ensemble HP selection — check middle is most common")
    ax.set_ylim(0, max(ys) + 1.5)
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
        color="purple",
        label="Deep ensemble",
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
    """Horizontal bar chart of per-class F1 with the worst class highlighted."""
    stats = per_class_precision_recall_f1(y_true, y_pred, n_classes=10)
    f1 = stats["f1"]
    worst = int(np.argmin(f1))
    colors = ["firebrick" if i == worst else "purple" for i in range(10)]
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.barh(range(10), f1, color=colors)
    ax.set_yticks(range(10))
    ax.set_yticklabels(CLASS_NAMES)
    ax.invert_yaxis()
    ax.set_xlabel("F1 score")
    ax.set_xlim(0, 1)
    ax.set_title("Deep ensemble CNN — per-class F1 (lowest in red)")
    for i, v in enumerate(f1):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=9)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_training_curves(histories: list[dict], out_path: Path) -> None:
    """Overlay train (solid) and val (dashed) loss curves for each member."""
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for m, h in enumerate(histories):
        epochs = np.arange(1, len(h["train_loss"]) + 1)
        color = cmap(m % 10)
        ax.plot(epochs, h["train_loss"], color=color, linestyle="-",
                label=f"member {m} train")
        ax.plot(epochs, h["val_loss"], color=color, linestyle="--",
                label=f"member {m} val")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-entropy loss")
    ax.set_title("Ensemble training curves (fold 0)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_ensemble_disagreement(
    y_true: np.ndarray, y_pred: np.ndarray, y_prob_mean: np.ndarray, out_path: Path
) -> None:
    """Predictive entropy histograms split by correctness."""
    eps = 1e-12
    p = np.clip(y_prob_mean, eps, 1.0)
    entropy = -np.sum(p * np.log(p), axis=1)
    correct = y_pred == y_true
    fig, ax = plt.subplots(figsize=(9, 6))
    bins = np.linspace(0, max(entropy.max(), 1e-3), 40)
    ax.hist(
        entropy[correct], bins=bins, color="seagreen", alpha=0.5,
        label=f"Correct (n={int(correct.sum())}, μ={entropy[correct].mean():.3f})",
    )
    ax.hist(
        entropy[~correct], bins=bins, color="firebrick", alpha=0.5,
        label=f"Incorrect (n={int((~correct).sum())}, μ={entropy[~correct].mean():.3f})",
    )
    ax.set_xlabel("Predictive entropy (nats)")
    ax.set_ylabel("Count")
    ax.set_title("Ensemble uncertainty — correct vs incorrect predictions")
    ax.legend()
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def _to_jsonable(obj):
    """Recursively convert numpy / Path values for json.dump."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


def log_deltas(agg: dict) -> None:
    """Log ΔAcc/F1/ECE/NLL versus previously-saved LR and RF aggregates."""
    for prev_label, prev_path in (
        ("LR", "results/metrics/03_logreg_results.json"),
        ("RF", "results/metrics/04_rf_results.json"),
    ):
        path = Path(prev_path)
        if not path.exists():
            logger.warning("%s results missing at %s; skipping deltas", prev_label, path)
            continue
        prev_agg = json.loads(path.read_text())["aggregate_metrics"]
        logger.info("vs %s:", prev_label)
        for k, label in (("accuracy", "accuracy"), ("macro_f1", "macro F1"),
                         ("ece", "ECE"), ("nll", "NLL")):
            d = agg[k]["mean"] - prev_agg[k]["mean"]
            sign = "+" if d >= 0 else ""
            logger.info(
                "  %-9s %s=%.4f  CNN=%.4f  Δ=%s%.4f",
                label, prev_label, prev_agg[k]["mean"], agg[k]["mean"], sign, d,
            )


def main() -> None:
    """End-to-end nested CV with deep ensembles."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    if args.fast:
        args.img_size = 32
        args.epochs = 5
        args.ensemble_members = 2
        logger.info("FAST mode: img_size=32, epochs=5, ensemble_members=2")

    figures_dir: Path = args.out_dir
    metrics_dir = Path("results") / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Torch device: %s", device)

    images, labels, _ = load_galaxy10()
    logger.info("Preprocessing images to %dx%d...", args.img_size, args.img_size)
    X = prepare_images_for_cnn(images, target_size=args.img_size)
    y = labels.astype(np.int64)
    logger.info("Image tensor: %s (%.1f MB)", X.shape, X.nbytes / 1e6)
    # Free the raw uint8 tensor; we don't need it again.
    del images

    outer_folds = stratified_kfold_split(y, k=10, seed=args.seed)
    fold_records: list[dict] = []
    start = time.perf_counter()

    for i, (outer_train, outer_test) in enumerate(outer_folds):
        logger.info(
            "=== Outer fold %d/%d (train=%d, test=%d) ===",
            i + 1, 10, outer_train.size, outer_test.size,
        )
        X_outer_train = X[outer_train]
        y_outer_train = y[outer_train]
        X_outer_test = X[outer_test]
        y_outer_test = y[outer_test]

        # Inner CV — single CNN per HP for speed.
        inner_folds = stratified_kfold_split(
            y_outer_train, k=3, seed=args.seed + i
        )
        inner_scores: dict[str, list[float]] = {
            f"lr={lr}": [] for lr in LR_GRID
        }
        for hp in HP_GRID:
            key = f"lr={hp['lr']}"
            for j, (inner_train, inner_val) in enumerate(inner_folds):
                X_in_tr = X_outer_train[inner_train]
                y_in_tr = y_outer_train[inner_train]
                X_in_val = X_outer_train[inner_val]
                y_in_val = y_outer_train[inner_val]
                logger.info(
                    "  inner %d/%d hp=%s training single CNN (n=%d)",
                    j + 1, 3, key, X_in_tr.shape[0],
                )
                model = DeepEnsemble(
                    n_members=1, epochs=args.epochs, seed=args.seed, **hp
                )
                model.fit(X_in_tr, y_in_tr)
                y_in_pred = model.predict(X_in_val)
                score = macro_f1(y_in_val, y_in_pred, n_classes=10)
                inner_scores[key].append(score)
                logger.info("    macro_f1=%.4f", score)
                del model
                del X_in_tr, y_in_tr, X_in_val, y_in_val

        mean_scores = {k: float(np.mean(v)) for k, v in inner_scores.items()}
        best_key = max(mean_scores, key=mean_scores.get)
        best_lr = float(best_key.split("=")[1])
        best_hp = {"lr": best_lr, "weight_decay": 1e-4}
        logger.info(
            "  best HP=%s (inner mean macro_f1=%.4f)",
            best_hp, mean_scores[best_key],
        )

        # Outer retraining: full ensemble.
        logger.info(
            "  training ensemble of %d members on full outer train (n=%d)",
            args.ensemble_members, X_outer_train.shape[0],
        )
        ensemble = DeepEnsemble(
            n_members=args.ensemble_members,
            epochs=args.epochs,
            seed=args.seed,
            **best_hp,
        )
        ensemble.fit(X_outer_train, y_outer_train)
        y_prob_per_member = ensemble.predict_proba_per_member(X_outer_test)
        y_prob = y_prob_per_member.mean(axis=0)
        y_pred = np.argmax(y_prob, axis=1)

        fold_acc = accuracy(y_outer_test, y_pred)
        fold_f1 = macro_f1(y_outer_test, y_pred, 10)
        fold_ece = expected_calibration_error(y_outer_test, y_prob)
        fold_nll = negative_log_likelihood(y_outer_test, y_prob)
        logger.info(
            "  outer fold %d: acc=%.4f macro_f1=%.4f ece=%.4f nll=%.4f",
            i + 1, fold_acc, fold_f1, fold_ece, fold_nll,
        )

        fold_records.append({
            "fold_idx": i,
            "best_hyperparams": best_hp,
            "y_test": y_outer_test,
            "y_pred": y_pred,
            "y_prob": y_prob,
            "y_prob_per_member": y_prob_per_member,
            "test_indices": outer_test,
            "histories": ensemble.histories,
            "inner_scores": inner_scores,
            "fold_metrics": {
                "accuracy": fold_acc,
                "macro_f1": fold_f1,
                "ece": fold_ece,
                "nll": fold_nll,
            },
        })
        # Free memory before the next fold.
        del ensemble, X_outer_train, y_outer_train, X_outer_test, y_outer_test

    elapsed = time.perf_counter() - start
    logger.info("Nested CV finished in %.1f s (%.1f min)", elapsed, elapsed / 60.0)

    # Coverage assertion: every sample is in exactly one outer test fold.
    all_test = np.concatenate([r["test_indices"] for r in fold_records])
    expected = np.arange(y.size)
    if not np.array_equal(np.sort(all_test), expected):
        missing = np.setdiff1d(expected, all_test)
        dup = all_test.size - np.unique(all_test).size
        raise AssertionError(
            f"outer test coverage broken (missing={missing.size}, duplicates={dup})"
        )

    order = np.argsort(all_test)
    all_y_true = np.concatenate([r["y_test"] for r in fold_records])[order]
    all_y_pred = np.concatenate([r["y_pred"] for r in fold_records])[order]
    all_y_prob = np.concatenate([r["y_prob"] for r in fold_records], axis=0)[order]
    all_y_prob_per_member = np.concatenate(
        [r["y_prob_per_member"] for r in fold_records], axis=1
    )[:, order, :]

    def _agg(key: str) -> dict:
        values = [r["fold_metrics"][key] for r in fold_records]
        return {"mean": float(np.mean(values)), "std": float(np.std(values))}

    agg = {
        "accuracy": _agg("accuracy"),
        "macro_f1": _agg("macro_f1"),
        "ece": _agg("ece"),
        "nll": _agg("nll"),
    }
    logger.info(
        "Aggregate: accuracy=%.4f±%.4f macro_f1=%.4f±%.4f ECE=%.4f NLL=%.4f",
        agg["accuracy"]["mean"], agg["accuracy"]["std"],
        agg["macro_f1"]["mean"], agg["macro_f1"]["std"],
        agg["ece"]["mean"], agg["nll"]["mean"],
    )
    log_deltas(agg)

    # Persist results.
    json_payload = {
        "wall_clock_seconds": round(elapsed, 2),
        "args": vars(args),
        "hyperparam_grid": HP_GRID,
        "aggregate_metrics": agg,
        "outer_fold_results": [
            {
                "fold_idx": r["fold_idx"],
                "best_hyperparams": r["best_hyperparams"],
                "test_indices": r["test_indices"],
                "inner_scores": r["inner_scores"],
                "fold_metrics": r["fold_metrics"],
                "histories": r["histories"],
            }
            for r in fold_records
        ],
    }
    json_path = metrics_dir / "05_cnn_results.json"
    json_path.write_text(json.dumps(_to_jsonable(json_payload), indent=2))
    logger.info("Wrote %s", json_path)

    pred_path = metrics_dir / "05_cnn_predictions.npz"
    np.savez_compressed(
        pred_path,
        y_true=all_y_true,
        y_pred=all_y_pred,
        y_prob=all_y_prob,
        y_prob_per_member=all_y_prob_per_member,
    )
    logger.info("Wrote %s", pred_path)

    plot_confusion(
        all_y_true, all_y_pred, figures_dir / "05_cnn_confusion.png"
    )
    plot_hp_selection(fold_records, figures_dir / "05_cnn_hp_selection.png")
    plot_reliability(
        all_y_true, all_y_prob, figures_dir / "05_cnn_reliability.png"
    )
    plot_per_class_f1(
        all_y_true, all_y_pred, figures_dir / "05_cnn_per_class_f1.png"
    )
    plot_training_curves(
        fold_records[0]["histories"],
        figures_dir / "05_cnn_training_curves.png",
    )
    plot_ensemble_disagreement(
        all_y_true, all_y_pred, all_y_prob,
        figures_dir / "05_cnn_ensemble_disagreement.png",
    )

    final_acc = accuracy(all_y_true, all_y_pred)
    final_f1 = macro_f1(all_y_true, all_y_pred, 10)
    final_ece = expected_calibration_error(all_y_true, all_y_prob)
    final_nll = negative_log_likelihood(all_y_true, all_y_prob)
    logger.info(
        "Concatenated holdout metrics: acc=%.4f macro_f1=%.4f ECE=%.4f NLL=%.4f",
        final_acc, final_f1, final_ece, final_nll,
    )


if __name__ == "__main__":
    main()
