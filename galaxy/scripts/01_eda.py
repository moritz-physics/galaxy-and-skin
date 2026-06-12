"""Exploratory data analysis for the Galaxy10 DECaLS dataset.

Loads the dataset, logs ground-truth integrity checks, produces five
diagnostic figures and writes a JSON summary of the dataset.

Run from the project root::

    uv run python scripts/01_eda.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; scripts never call plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Make the `src/` package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.data import CLASS_NAMES, load_galaxy10  # noqa: E402

logger = logging.getLogger("01_eda")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}


def plot_class_distribution(
    labels: np.ndarray, counts: np.ndarray, out_dir: Path
) -> None:
    """Save a bar chart of sample counts per class.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer class labels (unused beyond documentation of intent).
    counts : numpy.ndarray
        Per-class sample counts, indexed by class id.
    out_dir : Path
        Directory in which to save the figure.
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(CLASS_NAMES))
    bars = ax.bar(x, counts, color="steelblue")
    ax.set_xticks(x)
    ax.set_xticklabels(CLASS_NAMES, rotation=30, ha="right")
    ax.set_ylabel("Number of images")
    ax.set_title("Galaxy10 DECaLS — class distribution")
    for bar, count in zip(bars, counts):
        ax.annotate(
            str(int(count)),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )
    fig.savefig(out_dir / "01_class_distribution.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_sample_grid(
    images: np.ndarray, labels: np.ndarray, rng: np.random.Generator, out_dir: Path
) -> None:
    """Save a 10x8 grid of random sample images, one class per row.

    Parameters
    ----------
    images : numpy.ndarray
        Image tensor of shape ``(N, 256, 256, 3)``.
    labels : numpy.ndarray
        Integer class labels of shape ``(N,)``.
    rng : numpy.random.Generator
        Seeded random generator used to pick samples.
    out_dir : Path
        Directory in which to save the figure.
    """
    n_cols = 8
    fig, axes = plt.subplots(10, n_cols, figsize=(2 * n_cols, 2 * 10))
    for class_id in range(10):
        class_indices = np.flatnonzero(labels == class_id)
        chosen = rng.choice(class_indices, size=n_cols, replace=False)
        for col in range(n_cols):
            ax = axes[class_id, col]
            ax.imshow(images[chosen[col]])
            ax.set_xticks([])
            ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(
                    f"{class_id}: {CLASS_NAMES[class_id]}",
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=9,
                )
    fig.suptitle("Galaxy10 DECaLS — random samples per class", y=0.99)
    fig.savefig(out_dir / "01_sample_grid.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_mean_per_class(
    images: np.ndarray, labels: np.ndarray, out_dir: Path
) -> None:
    """Save a 2x5 grid of per-class pixel-wise mean images.

    The mean is computed in ``float32`` and displayed as ``uint8``. This is
    the primary correctness check: round/edge-on/cigar classes should yield
    visually distinct mean morphologies.

    Parameters
    ----------
    images : numpy.ndarray
        Image tensor of shape ``(N, 256, 256, 3)``.
    labels : numpy.ndarray
        Integer class labels of shape ``(N,)``.
    out_dir : Path
        Directory in which to save the figure.
    """
    fig, axes = plt.subplots(2, 5, figsize=(15, 6.5))
    for class_id in range(10):
        ax = axes[class_id // 5, class_id % 5]
        class_indices = np.flatnonzero(labels == class_id)
        mean_img = images[class_indices].astype(np.float32).mean(axis=0)
        ax.imshow(mean_img.astype(np.uint8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"{class_id}: {CLASS_NAMES[class_id]}", fontsize=9)
    fig.suptitle("Galaxy10 DECaLS — per-class mean image", y=1.0)
    fig.savefig(out_dir / "01_mean_per_class.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_pixel_intensity_hist(
    images: np.ndarray, labels: np.ndarray, out_dir: Path
) -> None:
    """Save overlaid step histograms of mean per-image brightness per class.

    Parameters
    ----------
    images : numpy.ndarray
        Image tensor of shape ``(N, 256, 256, 3)``.
    labels : numpy.ndarray
        Integer class labels of shape ``(N,)``.
    out_dir : Path
        Directory in which to save the figure.
    """
    brightness = images.astype(np.float32).reshape(images.shape[0], -1).mean(axis=1)
    bins = np.linspace(0.0, float(brightness.max()), 60)
    fig, ax = plt.subplots(figsize=(10, 6))
    cmap = plt.get_cmap("tab10")
    for class_id in range(10):
        values = brightness[labels == class_id]
        ax.hist(
            values,
            bins=bins,
            histtype="step",
            density=True,
            linewidth=1.5,
            color=cmap(class_id),
            label=f"{class_id}: {CLASS_NAMES[class_id]}",
        )
    ax.set_xlabel("Mean per-image brightness (0-255)")
    ax.set_ylabel("Density")
    ax.set_title("Galaxy10 DECaLS — per-image brightness by class")
    ax.legend(fontsize=8)
    fig.savefig(out_dir / "01_pixel_intensity_hist.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_metadata_distributions(meta: dict, out_dir: Path) -> None:
    """Save histograms of redshift, RA and Dec, titled with their medians.

    Parameters
    ----------
    meta : dict
        Metadata mapping with ``redshift``, ``ra`` and ``dec`` arrays.
    out_dir : Path
        Directory in which to save the figure.
    """
    fields: tuple[tuple[str, str], ...] = (
        ("redshift", "Redshift"),
        ("ra", "RA (deg)"),
        ("dec", "Dec (deg)"),
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, (key, label) in zip(axes, fields):
        values = meta[key]
        finite = values[np.isfinite(values)]
        ax.hist(finite, bins=60, color="steelblue")
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        median = float(np.median(finite)) if finite.size else float("nan")
        ax.set_title(f"{label} — median = {median:.4g}")
    fig.suptitle("Galaxy10 DECaLS — metadata distributions", y=1.02)
    fig.savefig(out_dir / "01_metadata_distributions.png", **SAVE_KWARGS)
    plt.close(fig)


def main() -> None:
    """Parse arguments, run the EDA pipeline and write outputs."""
    parser = argparse.ArgumentParser(description="Galaxy10 DECaLS EDA")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "figures",
        help="directory for output figures",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    rng = np.random.default_rng(args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    images, labels, meta = load_galaxy10()

    # Ground-truth checks.
    logger.info("Image count: %d", images.shape[0])
    logger.info("Image dtype: %s, shape: %s", images.dtype, images.shape)
    logger.info(
        "Label dtype: %s, range: [%d, %d]",
        labels.dtype,
        int(labels.min()),
        int(labels.max()),
    )
    for key in ("ra", "dec", "redshift"):
        n_nan = int(np.isnan(meta[key]).sum())
        logger.info("NaN count in %s: %d", key, n_nan)

    counts = np.bincount(labels, minlength=10)
    for class_id in range(10):
        logger.info(
            "Class %d (%s): %d images",
            class_id,
            CLASS_NAMES[class_id],
            int(counts[class_id]),
        )

    # Figures.
    logger.info("Generating figures in %s", out_dir)
    plot_class_distribution(labels, counts, out_dir)
    plot_sample_grid(images, labels, rng, out_dir)
    plot_mean_per_class(images, labels, out_dir)
    plot_pixel_intensity_hist(images, labels, out_dir)
    plot_metadata_distributions(meta, out_dir)

    # JSON summary.
    redshift = meta["redshift"]
    redshift_finite = redshift[np.isfinite(redshift)]
    summary: dict = {
        "total_count": int(images.shape[0]),
        "per_class_counts": {
            CLASS_NAMES[i]: int(counts[i]) for i in range(10)
        },
        "images": {"dtype": str(images.dtype), "shape": list(images.shape)},
        "labels": {"dtype": str(labels.dtype), "shape": list(labels.shape)},
        "redshift": {
            "min": float(np.min(redshift_finite)),
            "median": float(np.median(redshift_finite)),
            "max": float(np.max(redshift_finite)),
        },
        "pxscale_unique": np.unique(meta["pxscale"]).astype(float).tolist(),
        "nan_counts": {
            key: int(np.isnan(meta[key]).sum()) for key in ("ra", "dec", "redshift")
        },
    }
    summary_path = metrics_dir / "01_eda_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
