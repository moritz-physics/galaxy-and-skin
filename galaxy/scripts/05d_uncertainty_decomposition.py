"""Decomposition of deep-ensemble predictive uncertainty.

Splits the ensemble's predictive entropy ``H[p̄]`` into the data-inherent
(aleatoric) component ``E_m[H[p_m]]`` and the model-disagreement
(epistemic) component ``MI = H[p̄] − E_m[H[p_m]]``. The three panels show
the joint behaviour and how each component varies by class.

Run from the project root::

    uv run python scripts/05d_uncertainty_decomposition.py
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

logger = logging.getLogger("05d_uncertainty")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}
_EPS = 1e-12


def entropy_rows(probs: np.ndarray) -> np.ndarray:
    """Shannon entropy of each row of a probability matrix, in nats."""
    safe = np.clip(probs, _EPS, 1.0)
    return -np.sum(safe * np.log(safe), axis=-1)


def main() -> None:
    """Load per-member CNN probs, compute H / E_H / MI, emit one figure."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )
    figures_dir = Path("results") / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    data = np.load(Path("results") / "metrics" / "05_cnn_predictions.npz")
    y_true = data["y_true"]
    y_pred = data["y_pred"]
    y_prob_per_member = data["y_prob_per_member"]  # (M, N, K)
    M, N, K = y_prob_per_member.shape
    logger.info("Loaded per-member probs: M=%d members, N=%d samples, K=%d classes",
                M, N, K)

    p_bar = y_prob_per_member.mean(axis=0)               # (N, K)
    H = entropy_rows(p_bar)                              # predictive entropy
    member_H = entropy_rows(y_prob_per_member)           # (M, N) per-member entropies
    E_H = member_H.mean(axis=0)                          # aleatoric
    MI = H - E_H                                          # epistemic
    # Numerical guard: MI is mathematically non-negative; tiny negatives can
    # appear from the clipping in entropy_rows. Floor at zero for clean plots.
    MI = np.clip(MI, 0.0, None)
    correct = y_pred == y_true

    logger.info(
        "Predictive entropy   μ=%.3f σ=%.3f  (correct μ=%.3f, incorrect μ=%.3f)",
        H.mean(), H.std(), H[correct].mean(), H[~correct].mean(),
    )
    logger.info(
        "Aleatoric (E_H)      μ=%.3f σ=%.3f", E_H.mean(), E_H.std(),
    )
    logger.info(
        "Epistemic (MI)       μ=%.4f σ=%.4f", MI.mean(), MI.std(),
    )

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    # Panel 1 — scatter H vs MI, coloured by correctness.
    ax = axes[0]
    ax.scatter(
        H[correct], MI[correct],
        s=6, alpha=0.25, color="seagreen", label=f"correct (n={int(correct.sum())})",
    )
    ax.scatter(
        H[~correct], MI[~correct],
        s=6, alpha=0.25, color="firebrick",
        label=f"incorrect (n={int((~correct).sum())})",
    )
    ax.set_xlabel("Predictive entropy H[p̄] (nats)")
    ax.set_ylabel("Mutual information MI = H[p̄] − E_m[H[p_m]] (nats)")
    ax.set_title("Total vs epistemic uncertainty")
    leg = ax.legend(loc="upper left")
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    # Panel 2 — per-class MI boxplot (epistemic).
    ax = axes[1]
    per_class_mi = [MI[y_true == c] for c in range(10)]
    bp = ax.boxplot(
        per_class_mi,
        tick_labels=[str(c) for c in range(10)],
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.6)
    ax.set_xlabel("Class id")
    ax.set_ylabel("Epistemic uncertainty (MI, nats)")
    ax.set_title("Per-class epistemic uncertainty (model disagreement)")
    class_labels = [f"{i}: {CLASS_NAMES[i]}" for i in range(10)]
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(class_labels, rotation=40, ha="right", fontsize=8)

    # Panel 3 — per-class aleatoric (E_H) boxplot.
    ax = axes[2]
    per_class_ah = [E_H[y_true == c] for c in range(10)]
    bp = ax.boxplot(
        per_class_ah,
        tick_labels=[str(c) for c in range(10)],
        showfliers=False,
        patch_artist=True,
    )
    for patch in bp["boxes"]:
        patch.set_facecolor("goldenrod")
        patch.set_alpha(0.6)
    ax.set_xlabel("Class id")
    ax.set_ylabel("Aleatoric uncertainty E_m[H[p_m]] (nats)")
    ax.set_title("Per-class aleatoric uncertainty (inherent ambiguity)")
    ax.set_xticks(range(1, 11))
    ax.set_xticklabels(class_labels, rotation=40, ha="right", fontsize=8)

    fig.suptitle(
        "Deep ensemble uncertainty decomposition  (M=%d members, N=%d samples)"
        % (M, N), y=1.02,
    )
    out_path = figures_dir / "05d_uncertainty_decomposition.png"
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)
    logger.info("Wrote %s", out_path)

    # Per-class table for the log.
    header = f"{'cls':>3} {'name':<25} {'μ MI':>9} {'μ E_H':>9} {'n':>6}"
    sep = "-" * len(header)
    logger.info(sep); logger.info(header); logger.info(sep)
    for c in range(10):
        mask = y_true == c
        logger.info(
            "%3d %-25s %9.4f %9.4f %6d",
            c, CLASS_NAMES[c], MI[mask].mean(), E_H[mask].mean(), int(mask.sum()),
        )
    logger.info(sep)


if __name__ == "__main__":
    main()
