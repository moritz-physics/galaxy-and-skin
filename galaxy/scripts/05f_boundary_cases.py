"""Boundary cases — galaxies straddling gross morphological superclasses.

A within-family confusion (e.g. Barred ↔ Unbarred Tight Spiral) is a
labelling-edge problem. A *cross-superclass* confusion — where the
ensemble's top-1 and top-2 sit in fundamentally different morphological
families (smooth / spiral / edge-on / merging) — is much more interesting:
the model is genuinely uncertain about the kind of galaxy it sees. This
script surfaces those cases.

Run from the project root::

    uv run python scripts/05f_boundary_cases.py
"""

from __future__ import annotations

import logging
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.data import CLASS_NAMES, load_galaxy10  # noqa: E402

logger = logging.getLogger("05f_boundary_cases")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}
_EPS = 1e-12

N_GRID_ROWS = 4
N_GRID_COLS = 6
N_EXAMPLES = N_GRID_ROWS * N_GRID_COLS
HIGH_ENTROPY_THRESHOLD = 1.0  # nats

SUPERCLASSES: dict[str, set[int]] = {
    "SMOOTH":  {0, 2, 3, 4},
    "SPIRAL":  {5, 6, 7},
    "EDGE_ON": {8, 9},
    "MERGING": {1},
}


def build_superclass_map(n_classes: int = 10) -> np.ndarray:
    """Lookup table mapping each class id to its superclass *name*."""
    mapping = np.empty(n_classes, dtype=object)
    for name, members in SUPERCLASSES.items():
        for cls in members:
            mapping[cls] = name
    if any(m is None for m in mapping):
        missing = [i for i, m in enumerate(mapping) if m is None]
        raise ValueError(f"classes {missing} not assigned to any superclass")
    return mapping


def main() -> None:
    """Identify cross-superclass uncertain cases and render a 4×6 gallery."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    figures_dir = Path("results") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    pred_path = Path("results") / "metrics" / "05_cnn_predictions.npz"
    data = np.load(pred_path)
    y_true = data["y_true"]
    y_prob = data["y_prob"]
    if "y_prob_per_member" not in data.files:
        raise RuntimeError(
            f"{pred_path} is missing 'y_prob_per_member' — rerun task 5"
        )
    y_prob_per_member = data["y_prob_per_member"]
    n = y_true.size
    logger.info(
        "Loaded predictions: N=%d, K=%d, M=%d",
        n, y_prob.shape[1], y_prob_per_member.shape[0],
    )

    # Predictive entropy on the ensemble mean.
    safe = np.clip(y_prob, _EPS, 1.0)
    entropy = -np.sum(safe * np.log(safe), axis=1)

    # Top-2 predicted classes per sample. argsort descending, take first two.
    order = np.argsort(-y_prob, axis=1)
    top1 = order[:, 0]
    top2 = order[:, 1]
    top1_prob = y_prob[np.arange(n), top1]
    top2_prob = y_prob[np.arange(n), top2]

    super_of = build_superclass_map(n_classes=y_prob.shape[1])
    super_top1 = super_of[top1]
    super_top2 = super_of[top2]
    cross_boundary = super_top1 != super_top2

    # Aggregate stats: total cross-boundary + high-entropy cases and a
    # breakdown by unordered superclass pair.
    high_unc = entropy > HIGH_ENTROPY_THRESHOLD
    qualifying = cross_boundary & high_unc
    n_qualifying = int(qualifying.sum())
    logger.info(
        "Cross-boundary AND H>%.2f nats: %d / %d (%.1f%%)",
        HIGH_ENTROPY_THRESHOLD, n_qualifying, n, 100.0 * n_qualifying / n,
    )

    pair_counts: Counter[tuple[str, str]] = Counter()
    cb_indices = np.flatnonzero(cross_boundary)
    for i in cb_indices:
        pair = tuple(sorted((super_top1[i], super_top2[i])))
        pair_counts[pair] += 1
    logger.info("Cross-boundary breakdown by superclass pair (any entropy):")
    for pair, count in sorted(pair_counts.items(), key=lambda kv: kv[1], reverse=True):
        logger.info("  %s ↔ %s: %d cases", pair[0], pair[1], count)

    # Pick the N_EXAMPLES highest-entropy cross-boundary cases.
    cb_pool = np.flatnonzero(cross_boundary)
    if cb_pool.size == 0:
        raise RuntimeError("no cross-boundary uncertain cases found")
    if cb_pool.size < N_EXAMPLES:
        logger.warning(
            "only %d cross-boundary cases available (wanted %d)",
            cb_pool.size, N_EXAMPLES,
        )
    sorted_by_entropy = cb_pool[np.argsort(-entropy[cb_pool])]
    selected = sorted_by_entropy[:N_EXAMPLES]

    logger.info("Loading raw Galaxy10 images...")
    images, raw_labels, _ = load_galaxy10()
    if not np.array_equal(raw_labels.astype(np.int64), y_true.astype(np.int64)):
        raise RuntimeError(
            "predictions and raw labels disagree — sample order has drifted"
        )

    fig, axes = plt.subplots(
        N_GRID_ROWS, N_GRID_COLS,
        figsize=(N_GRID_COLS * 2.4, N_GRID_ROWS * 3.0),
    )
    for cell_idx in range(N_GRID_ROWS * N_GRID_COLS):
        row = cell_idx // N_GRID_COLS
        col = cell_idx % N_GRID_COLS
        ax = axes[row, col]
        if cell_idx >= selected.size:
            ax.axis("off")
            continue
        i = int(selected[cell_idx])
        ax.imshow(images[i])
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(
            f"true: {CLASS_NAMES[int(y_true[i])]}\n"
            f"top1: {CLASS_NAMES[int(top1[i])]} "
            f"({top1_prob[i]:.0%}) [{super_top1[i]}]\n"
            f"top2: {CLASS_NAMES[int(top2[i])]} "
            f"({top2_prob[i]:.0%}) [{super_top2[i]}]\n"
            f"H={entropy[i]:.2f} nats",
            fontsize=7,
        )

    fig.suptitle(
        "Galaxies where the ensemble is uncertain across gross morphological "
        "boundaries (smooth / spiral / edge-on / merging)",
        y=1.005, fontsize=12,
    )
    out_path = figures_dir / "05f_boundary_cases.png"
    fig.tight_layout()
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
