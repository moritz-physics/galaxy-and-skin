"""Feature extraction pipeline and correctness plots for Galaxy10 DECaLS.

Extracts the full handcrafted feature matrix for all 17,736 images, saves it
to ``data/processed/features.npz``, and produces four diagnostic figures plus
a JSON summary.

Run from the project root::

    uv run python scripts/02_features.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend; scripts never call plt.show()
import matplotlib.pyplot as plt
import numpy as np

# Make the `src/` package importable when run as a plain script.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.augment import (  # noqa: E402
    additive_noise,
    augment_train,
    gaussian_blur,
    random_occlusion,
    random_rotation,
)
from galaxy_uq.data import (  # noqa: E402
    CLASS_NAMES,
    load_galaxy10,
    stratified_subset,
)
from galaxy_uq.features import (  # noqa: E402
    GROUP_SIZES,
    extract_all_features,
)

logger = logging.getLogger("02_features")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}

# Features highlighted in the per-class boxplot grid, with expected ordering.
_BOXPLOT_FEATURES: tuple[tuple[str, str], ...] = (
    ("cas_ellipticity", "Ellipticity — expect edge-on (8,9) > round smooth (2,3)"),
    ("cas_concentration", "Concentration — expect round smooth (2,3) > spirals (5,6,7)"),
    ("cas_asymmetry", "Asymmetry — expect disturbed/merging (0,1) > smooth"),
    ("gini", "Gini — expect smooth classes higher"),
    ("hu_0", "Hu moment 0 (log) — expect broad class separation"),
    ("photometric_gray_mean", "Gray mean — control, expect NO clean separation"),
)


def standardize(matrix: np.ndarray) -> np.ndarray:
    """Standardise columns to zero mean and unit variance.

    Parameters
    ----------
    matrix : numpy.ndarray
        Feature matrix of shape ``(N, D)``.

    Returns
    -------
    numpy.ndarray
        Standardised matrix; columns with zero variance are left centred only.
    """
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return (matrix - mean) / std


def pca(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute PCA scores and explained-variance ratios via SVD.

    Parameters
    ----------
    matrix : numpy.ndarray
        Standardised feature matrix of shape ``(N, D)``.

    Returns
    -------
    scores : numpy.ndarray
        Projected scores of shape ``(N, D)``.
    explained_variance_ratio : numpy.ndarray
        Fraction of variance carried by each component, length ``D``.
    """
    centered = matrix - matrix.mean(axis=0)
    u, s, _ = np.linalg.svd(centered, full_matrices=False)
    scores = u * s
    explained = s**2 / np.sum(s**2)
    return scores, explained


def plot_pca_2d(
    scores: np.ndarray,
    explained: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
) -> None:
    """Save a 2D PCA scatter coloured by class.

    Parameters
    ----------
    scores : numpy.ndarray
        PCA scores of shape ``(N, D)``.
    explained : numpy.ndarray
        Explained-variance ratios.
    labels : numpy.ndarray
        Integer class labels.
    out_dir : Path
        Directory in which to save the figure.
    """
    fig, ax = plt.subplots(figsize=(9, 7))
    cmap = plt.get_cmap("tab10")
    for class_id in range(10):
        mask = labels == class_id
        ax.scatter(
            scores[mask, 0],
            scores[mask, 1],
            s=5,
            alpha=0.3,
            color=cmap(class_id),
            label=f"{class_id}: {CLASS_NAMES[class_id]}",
        )
    ax.set_xlabel(f"PC1 ({explained[0] * 100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({explained[1] * 100:.1f}% variance)")
    ax.set_title("Galaxy10 DECaLS — 2D PCA of handcrafted features")
    ax.legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, markerscale=2.0
    )
    fig.savefig(out_dir / "02_pca_2d.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_pca_3d_pairs(
    scores: np.ndarray,
    explained: np.ndarray,
    labels: np.ndarray,
    out_dir: Path,
) -> None:
    """Save a 1x3 grid of PCA pair scatters: (PC1,PC2), (PC1,PC3), (PC2,PC3).

    Parameters
    ----------
    scores : numpy.ndarray
        PCA scores of shape ``(N, D)``.
    explained : numpy.ndarray
        Explained-variance ratios.
    labels : numpy.ndarray
        Integer class labels.
    out_dir : Path
        Directory in which to save the figure.
    """
    pairs = ((0, 1), (0, 2), (1, 2))
    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    cmap = plt.get_cmap("tab10")
    for ax, (i, j) in zip(axes, pairs):
        for class_id in range(10):
            mask = labels == class_id
            ax.scatter(
                scores[mask, i],
                scores[mask, j],
                s=5,
                alpha=0.3,
                color=cmap(class_id),
                label=f"{class_id}: {CLASS_NAMES[class_id]}",
            )
        ax.set_xlabel(f"PC{i + 1} ({explained[i] * 100:.1f}% variance)")
        ax.set_ylabel(f"PC{j + 1} ({explained[j] * 100:.1f}% variance)")
        ax.set_title(f"PC{i + 1} vs PC{j + 1}")
    axes[-1].legend(
        bbox_to_anchor=(1.02, 1.0), loc="upper left", fontsize=8, markerscale=2.0
    )
    fig.suptitle(
        "Galaxy10 DECaLS — PCA pair scatters of handcrafted features", y=1.02
    )
    fig.savefig(out_dir / "02_pca_3d_pairs.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_key_features_by_class(
    matrix: np.ndarray,
    labels: np.ndarray,
    feature_names: list[str],
    out_dir: Path,
) -> None:
    """Save a 2x3 grid of per-class boxplots for six diagnostic features.

    Parameters
    ----------
    matrix : numpy.ndarray
        Raw (unstandardised) feature matrix of shape ``(N, D)``.
    labels : numpy.ndarray
        Integer class labels.
    feature_names : list of str
        Feature names aligned with the columns of ``matrix``.
    out_dir : Path
        Directory in which to save the figure.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    for ax, (name, title) in zip(axes.ravel(), _BOXPLOT_FEATURES):
        col = feature_names.index(name)
        per_class = [matrix[labels == c, col] for c in range(10)]
        ax.boxplot(
            per_class,
            tick_labels=[str(c) for c in range(10)],
            showfliers=False,
        )
        ax.set_xlabel("Class id")
        ax.set_ylabel(name)
        ax.set_title(title, fontsize=10)
    fig.suptitle(
        "Galaxy10 DECaLS — key features by class (correctness check)", y=1.01
    )
    fig.savefig(out_dir / "02_key_features_by_class.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_feature_correlation(
    matrix: np.ndarray, out_dir: Path
) -> None:
    """Save a heatmap of absolute Pearson correlations between features.

    Parameters
    ----------
    matrix : numpy.ndarray
        Feature matrix of shape ``(N, D)``.
    out_dir : Path
        Directory in which to save the figure.
    """
    # A constant feature has zero variance; corrcoef yields NaN for it, which
    # we map to 0 correlation.
    with np.errstate(invalid="ignore", divide="ignore"):
        corr = np.corrcoef(matrix, rowvar=False)
    corr = np.abs(np.nan_to_num(corr, nan=0.0))
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(corr, cmap="coolwarm", vmin=0.0, vmax=1.0)
    fig.colorbar(im, ax=ax, label="|Pearson correlation|")

    # Draw group boundaries using the (already grouped) feature ordering.
    boundary = 0
    for size in list(GROUP_SIZES.values())[:-1]:
        boundary += size
        ax.axhline(boundary - 0.5, color="black", linewidth=0.8)
        ax.axvline(boundary - 0.5, color="black", linewidth=0.8)

    # Centre a tick label within each group.
    centers: list[float] = []
    start = 0
    for size in GROUP_SIZES.values():
        centers.append(start + size / 2.0 - 0.5)
        start += size
    ax.set_xticks(centers)
    ax.set_yticks(centers)
    ax.set_xticklabels(list(GROUP_SIZES.keys()), rotation=30, ha="right")
    ax.set_yticklabels(list(GROUP_SIZES.keys()))
    ax.set_title("Galaxy10 DECaLS — absolute feature correlation (grouped)")
    fig.savefig(out_dir / "02_feature_correlation.png", **SAVE_KWARGS)
    plt.close(fig)


def plot_augmentation_grid(
    images: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    out_dir: Path,
) -> None:
    """Save a 2x4 grid demonstrating the augmentation transforms.

    Parameters
    ----------
    images : numpy.ndarray
        Image tensor of shape ``(N, 256, 256, 3)``.
    labels : numpy.ndarray
        Integer class labels.
    rng : numpy.random.Generator
        Seeded random generator used to pick the example and drive transforms.
    out_dir : Path
        Directory in which to save the figure.
    """
    class5 = np.flatnonzero(labels == 5)
    base = images[int(rng.choice(class5))]
    panels: list[tuple[str, np.ndarray]] = [
        ("original", base),
        ("random_rotation", random_rotation(base, rng)),
        ("h-flip", np.fliplr(base)),
        ("v-flip", np.flipud(base)),
        ("gaussian_blur (sigma=3)", gaussian_blur(base, 3.0)),
        ("additive_noise (sigma=25)", additive_noise(base, 25.0, rng)),
        ("random_occlusion (size=64)", random_occlusion(base, 64, rng)),
        ("augment_train", augment_train(base, rng)),
    ]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img)
        ax.set_title(title, fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(
        "Galaxy10 DECaLS — augmentation transforms (Barred Spiral example)",
        y=1.0,
    )
    fig.savefig(out_dir / "02_augmentation_grid.png", **SAVE_KWARGS)
    plt.close(fig)


def main() -> None:
    """Run the feature-extraction pipeline and write all outputs."""
    parser = argparse.ArgumentParser(description="Galaxy10 DECaLS feature extraction")
    parser.add_argument("--seed", type=int, default=0, help="random seed")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results") / "figures",
        help="directory for output figures",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help=(
            "if > 0, run on a balanced stratified subset of roughly this many "
            "images (for fast smoke tests); 0 means use all images"
        ),
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s"
    )

    rng = np.random.default_rng(args.seed)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path("data") / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir = Path("results") / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    images, labels, _ = load_galaxy10()

    # Optional balanced subset for fast smoke tests.
    if args.limit > 0:
        n_per_class = max(1, args.limit // 10)
        subset = stratified_subset(labels, n_per_class, seed=args.seed)
        images = images[subset]
        labels = labels[subset]
        logger.info(
            "Running on a stratified subset: %d images (%d per class)",
            images.shape[0],
            n_per_class,
        )

    # Feature extraction over all images.
    logger.info("Extracting features for %d images...", images.shape[0])
    start = time.perf_counter()
    feature_matrix, labels, feature_names = extract_all_features(images, labels)
    elapsed = time.perf_counter() - start
    logger.info(
        "Feature extraction took %.1f s (%.1f min) for D=%d features",
        elapsed,
        elapsed / 60.0,
        len(feature_names),
    )

    # Integrity check: no NaN / Inf.
    nan_mask = np.isnan(feature_matrix)
    inf_mask = np.isinf(feature_matrix)
    n_nan = int(nan_mask.sum())
    n_inf = int(inf_mask.sum())
    if n_nan or n_inf:
        bad_cols = np.unique(
            np.concatenate(
                [np.where(nan_mask.any(0))[0], np.where(inf_mask.any(0))[0]]
            )
        )
        for col in bad_cols:
            logger.error("Non-finite values in feature '%s'", feature_names[col])
        raise ValueError(
            f"feature matrix has {n_nan} NaN and {n_inf} Inf values"
        )
    logger.info("Feature matrix is finite (no NaN, no Inf)")

    # Save the feature matrix. Subset runs use a distinct file so they never
    # clobber the full feature matrix that later tasks depend on.
    features_name = "features_subset.npz" if args.limit > 0 else "features.npz"
    features_path = processed_dir / features_name
    np.savez_compressed(
        features_path,
        features=feature_matrix,
        labels=labels,
        feature_names=np.asarray(feature_names),
    )
    logger.info("Saved features to %s", features_path)

    # Correctness plots.
    logger.info("Generating figures in %s", out_dir)
    # For PCA visualisation only, clip each feature to its [1st, 99th]
    # percentile so a handful of extreme outliers don't dominate the
    # projection. The saved features.npz remains the raw matrix; any
    # downstream model is responsible for its own preprocessing.
    lo = np.percentile(feature_matrix, 1, axis=0)
    hi = np.percentile(feature_matrix, 99, axis=0)
    clipped = np.clip(feature_matrix, lo, hi)
    standardized = standardize(clipped)
    scores, explained = pca(standardized)
    plot_pca_2d(scores, explained, labels, out_dir)
    plot_pca_3d_pairs(scores, explained, labels, out_dir)
    plot_key_features_by_class(feature_matrix, labels, feature_names, out_dir)
    plot_feature_correlation(feature_matrix, out_dir)
    plot_augmentation_grid(images, labels, rng, out_dir)

    # JSON summary.
    summary: dict = {
        "total_feature_count": len(feature_names),
        "per_group_feature_count": dict(GROUP_SIZES),
        "extraction_wall_clock_seconds": round(elapsed, 2),
        "nan_count": n_nan,
        "inf_count": n_inf,
        "pca_explained_variance_first_10": [
            float(v) for v in explained[:10]
        ],
    }
    summary_path = metrics_dir / "02_feature_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    logger.info("Wrote summary to %s", summary_path)


if __name__ == "__main__":
    main()
