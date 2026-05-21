"""Qualitative gallery of confident CNN failures.

Identifies the three most-confused class pairs from the deep-ensemble
confusion matrix and shows the six most confidently-wrong predictions for
each pair as a 3×6 image grid. "Confidently wrong" — high model confidence
but incorrect prediction — is more diagnostic than uniformly-distributed
errors, because it spotlights cases where the learned representation
fails systematically.

Run from the project root::

    uv run python scripts/05e_failure_analysis.py
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

from galaxy_uq.data import CLASS_NAMES, load_galaxy10  # noqa: E402
from galaxy_uq.metrics import confusion_matrix  # noqa: E402

logger = logging.getLogger("05e_failure_analysis")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}
_EPS = 1e-12
N_PAIRS = 3
N_EXAMPLES_PER_PAIR = 6


def top_confused_pairs(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int, n_pairs: int,
) -> list[tuple[int, int, int]]:
    """Return the ``n_pairs`` unordered class pairs with the most confusion.

    Each entry is ``(class_a, class_b, total_count)`` with
    ``total_count = CM[a, b] + CM[b, a]`` and ``a < b``.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    pair_counts: list[tuple[int, int, int]] = []
    for a in range(n_classes):
        for b in range(a + 1, n_classes):
            pair_counts.append((a, b, int(cm[a, b] + cm[b, a])))
    pair_counts.sort(key=lambda t: t[2], reverse=True)
    return pair_counts[:n_pairs]


def confident_wrong_for_pair(
    a: int, b: int,
    y_true: np.ndarray, y_pred: np.ndarray, confidence: np.ndarray,
    k: int,
) -> np.ndarray:
    """Indices of the top-``k`` confident wrong predictions for pair ``{a, b}``.

    "Wrong for pair ``{a, b}``" means either ``(true=a, pred=b)`` or
    ``(true=b, pred=a)``; ties broken by global sample index.
    """
    mask = ((y_true == a) & (y_pred == b)) | ((y_true == b) & (y_pred == a))
    pair_idx = np.flatnonzero(mask)
    if pair_idx.size == 0:
        return pair_idx
    order = np.argsort(-confidence[pair_idx])
    return pair_idx[order[:k]]


def main() -> None:
    """Build the confused-pair gallery from saved predictions + raw images."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    figures_dir = Path("results") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path("results") / "metrics" / "05_cnn_predictions.npz"
    data = np.load(pred_path)
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob = data["y_prob"]
    logger.info("Loaded %d predictions from %s", y_true.size, pred_path)

    confidence = y_prob.max(axis=1)
    safe = np.clip(y_prob, _EPS, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1)

    pairs = top_confused_pairs(y_true, y_pred, n_classes=10, n_pairs=N_PAIRS)
    logger.info("Top %d confused class pairs (unordered):", N_PAIRS)
    for a, b, count in pairs:
        logger.info(
            "  %d (%s)  ↔  %d (%s)   total=%d",
            a, CLASS_NAMES[a], b, CLASS_NAMES[b], count,
        )

    selections: list[tuple[tuple[int, int], np.ndarray]] = []
    for a, b, _ in pairs:
        idx = confident_wrong_for_pair(
            a, b, y_true, y_pred, confidence, N_EXAMPLES_PER_PAIR
        )
        if idx.size < N_EXAMPLES_PER_PAIR:
            logger.warning(
                "  pair {%d,%d}: only %d confident wrong examples (wanted %d)",
                a, b, idx.size, N_EXAMPLES_PER_PAIR,
            )
        selections.append(((a, b), idx))

    needed = np.concatenate([idx for _, idx in selections if idx.size > 0])
    if needed.size == 0:
        raise RuntimeError("no confident-wrong samples found for any pair")
    logger.info("Loading raw Galaxy10 images to fetch %d examples...", needed.size)
    images, raw_labels, _ = load_galaxy10()
    # Predictions are saved in global sample-index order, so y_true[i] aligns
    # with images[i]. Sanity-check the alignment with one assertion.
    if not np.array_equal(raw_labels.astype(np.int64), y_true.astype(np.int64)):
        raise RuntimeError(
            "label alignment mismatch between predictions and raw HDF5 — "
            "the predictions are no longer in global sample order"
        )

    fig, axes = plt.subplots(
        N_PAIRS, N_EXAMPLES_PER_PAIR,
        figsize=(N_EXAMPLES_PER_PAIR * 2.3, N_PAIRS * 3.0),
    )
    if N_PAIRS == 1:
        axes = axes.reshape(1, -1)

    for row, ((a, b), idx) in enumerate(selections):
        for col in range(N_EXAMPLES_PER_PAIR):
            ax = axes[row, col]
            if col >= idx.size:
                ax.axis("off")
                continue
            global_idx = int(idx[col])
            t = int(y_true[global_idx])
            p = int(y_pred[global_idx])
            c = float(confidence[global_idx])
            h = float(entropy[global_idx])
            ax.imshow(images[global_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_title(
                f"true {t}: {CLASS_NAMES[t]}\n"
                f"pred {p}: {CLASS_NAMES[p]}\n"
                f"conf={c:.2f}  H={h:.2f}",
                fontsize=8,
            )
        # Row label on the leftmost panel only.
        axes[row, 0].set_ylabel(
            f"{CLASS_NAMES[a]}\n↔\n{CLASS_NAMES[b]}",
            rotation=0, ha="right", va="center", fontsize=10, labelpad=30,
        )

    fig.suptitle(
        "Confident CNN failures — top 6 high-confidence errors for the "
        f"{N_PAIRS} most-confused class pairs",
        y=1.005,
    )
    out_path = figures_dir / "05e_failure_gallery.png"
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
