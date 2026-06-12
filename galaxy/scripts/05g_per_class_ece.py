"""Per-class ECE for the deep-ensemble CNN.

Loads the stored predictions from script 05 and computes ECE separately for
the samples whose ground-truth label is each of the ten morphology classes,
together with the per-class top-bin over-confidence (mean confidence minus
mean accuracy on the highest confidence bin). Writes a JSON summary and a
small bar plot referenced in the report.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.metrics import expected_calibration_error  # noqa: E402

CLASS_NAMES = [
    "Disturbed", "Merging", "Round Smooth", "In-between Round",
    "Cigar", "Barred Spiral", "Tight Spiral", "Loose Spiral",
    "Edge-on no Bulge", "Edge-on w/ Bulge",
]


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    pred_path = Path("results/metrics/05_cnn_predictions.npz")
    arr = np.load(pred_path)
    y_true = arr["y_true"]
    y_prob = arr["y_prob"]

    per_class = []
    for c, name in enumerate(CLASS_NAMES):
        mask = y_true == c
        n = int(mask.sum())
        if n == 0:
            continue
        ece_c = expected_calibration_error(y_true[mask], y_prob[mask], n_bins=15)
        conf = y_prob[mask].max(axis=1)
        correct = (y_prob[mask].argmax(axis=1) == y_true[mask]).astype(np.float64)
        top_mask = conf >= 14.0 / 15.0
        if top_mask.any():
            top_gap = float(conf[top_mask].mean() - correct[top_mask].mean())
            top_n = int(top_mask.sum())
        else:
            top_gap, top_n = float("nan"), 0
        per_class.append(
            {
                "class": c, "name": name, "n": n,
                "ece": ece_c,
                "mean_confidence": float(conf.mean()),
                "accuracy": float(correct.mean()),
                "top_bin_gap": top_gap, "top_bin_n": top_n,
            }
        )
        logging.info(
            "%-20s n=%4d  ECE=%.3f  acc=%.3f  conf=%.3f  top-bin gap=%.3f (n=%d)",
            name, n, ece_c, correct.mean(), conf.mean(), top_gap, top_n,
        )

    out_json = Path("results/metrics/05g_per_class_ece.json")
    out_json.write_text(json.dumps(per_class, indent=2))
    logging.info("Wrote %s", out_json)

    fig, ax = plt.subplots(figsize=(7.5, 3.2))
    eces = [d["ece"] for d in per_class]
    ns = [d["n"] for d in per_class]
    xs = np.arange(len(per_class))
    bars = ax.bar(xs, eces, color="#5B7DB1")
    for x, e, n in zip(xs, eces, ns):
        ax.text(x, e + 0.005, f"n={n}", ha="center", fontsize=8)
    ax.set_xticks(xs)
    ax.set_xticklabels([d["name"] for d in per_class], rotation=35, ha="right",
                       fontsize=8)
    ax.set_ylabel("Per-class ECE")
    ax.set_title("Deep-ensemble CNN: per-class calibration error")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    out_png = Path("results/figures/05g_per_class_ece.png")
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_png)


if __name__ == "__main__":
    main()
