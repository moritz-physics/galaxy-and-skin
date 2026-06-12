"""Data loading utilities for the Galaxy10 DECaLS dataset.

This module exposes the canonical class names, the on-disk location of the
HDF5 dataset, a loader that performs integrity checks, and a helper for
building balanced subsets used during prototyping.
"""

from __future__ import annotations

import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger(__name__)

CLASS_NAMES: tuple[str, ...] = (
    "Disturbed",
    "Merging",
    "Round Smooth",
    "In-between Round Smooth",
    "Cigar Shaped Smooth",
    "Barred Spiral",
    "Unbarred Tight Spiral",
    "Unbarred Loose Spiral",
    "Edge-on without Bulge",
    "Edge-on with Bulge",
)

# Path to the raw HDF5 file, relative to the project root (the assumed cwd).
DATA_PATH: Path = Path("data") / "raw" / "Galaxy10_DECals.h5"

# Expected shape of the image tensor; used for the integrity assertion.
_EXPECTED_SHAPE: tuple[int, int, int, int] = (17736, 256, 256, 3)


def load_galaxy10(path: Path = DATA_PATH) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load the Galaxy10 DECaLS dataset from its HDF5 file.

    Parameters
    ----------
    path : Path, optional
        Location of the ``Galaxy10_DECals.h5`` file. Defaults to
        :data:`DATA_PATH`.

    Returns
    -------
    images : numpy.ndarray
        Image tensor of shape ``(17736, 256, 256, 3)`` and dtype ``uint8``.
    labels : numpy.ndarray
        Integer class labels of shape ``(17736,)`` and dtype ``int64``.
    meta : dict
        Mapping with keys ``ra``, ``dec``, ``redshift`` and ``pxscale``, each
        a ``float64`` array.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the loaded data fails any of the integrity checks (image shape,
        image dtype, or the set of label values).
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Galaxy10 HDF5 file not found at: {path}")

    logger.info("Loading Galaxy10 DECaLS from %s", path)
    with h5py.File(path, "r") as f:
        images = np.asarray(f["images"])
        labels = np.asarray(f["ans"]).astype(np.int64)
        meta: dict = {
            "ra": np.asarray(f["ra"]).astype(np.float64),
            "dec": np.asarray(f["dec"]).astype(np.float64),
            "redshift": np.asarray(f["redshift"]).astype(np.float64),
            "pxscale": np.asarray(f["pxscale"]).astype(np.float64),
        }

    try:
        assert images.shape == _EXPECTED_SHAPE, (
            f"unexpected image shape {images.shape}, expected {_EXPECTED_SHAPE}"
        )
        assert images.dtype == np.uint8, (
            f"unexpected image dtype {images.dtype}, expected uint8"
        )
        assert set(np.unique(labels).tolist()) == set(range(10)), (
            f"unexpected label set {sorted(np.unique(labels).tolist())}, "
            "expected 0..9"
        )
    except AssertionError as exc:
        raise ValueError(f"Galaxy10 integrity check failed: {exc}") from exc

    logger.info(
        "Loaded %d images (%s) and %d labels", len(images), images.dtype, len(labels)
    )
    return images, labels, meta


def stratified_subset(
    labels: np.ndarray, n_per_class: int, seed: int = 0
) -> np.ndarray:
    """Return indices of a class-balanced subset of the dataset.

    For each of the ten classes, ``n_per_class`` indices are drawn uniformly
    at random without replacement.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer class labels of shape ``(N,)``.
    n_per_class : int
        Number of samples to draw from each class.
    seed : int, optional
        Seed for the random number generator. Defaults to ``0``.

    Returns
    -------
    numpy.ndarray
        Sorted array of selected indices into ``labels`` of length
        ``10 * n_per_class``.

    Raises
    ------
    ValueError
        If a class has fewer than ``n_per_class`` samples available.
    """
    rng = np.random.default_rng(seed)
    selected: list[np.ndarray] = []
    for class_id in range(10):
        class_indices = np.flatnonzero(labels == class_id)
        if class_indices.size < n_per_class:
            raise ValueError(
                f"class {class_id} has only {class_indices.size} samples, "
                f"cannot draw {n_per_class}"
            )
        chosen = rng.choice(class_indices, size=n_per_class, replace=False)
        selected.append(chosen)
    return np.sort(np.concatenate(selected))
