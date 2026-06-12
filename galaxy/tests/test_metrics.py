"""Unit tests for metrics and CV splitters.

Run with::

    uv run pytest tests/
"""

from __future__ import annotations

import numpy as np
import pytest

from galaxy_uq.cv import stratified_kfold_split
from galaxy_uq.metrics import confusion_matrix, expected_calibration_error


# ---------------------------------------------------------------------------
# confusion_matrix
# ---------------------------------------------------------------------------


def test_confusion_matrix_perfect_predictions() -> None:
    """Perfect predictions produce a diagonal count matrix."""
    n_classes = 4
    y_true = np.array([0, 1, 2, 3, 0, 1, 2, 3], dtype=np.int64)
    y_pred = y_true.copy()
    cm = confusion_matrix(y_true, y_pred, n_classes)
    expected = np.diag([2, 2, 2, 2]).astype(np.int64)
    np.testing.assert_array_equal(cm, expected)
    assert int(np.diag(cm).sum()) == y_true.size


def test_confusion_matrix_all_wrong_zero_diagonal() -> None:
    """When no prediction matches the truth the diagonal must be all zero."""
    n_classes = 3
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([1, 2, 2, 0, 0, 1], dtype=np.int64)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    np.testing.assert_array_equal(np.diag(cm), np.zeros(n_classes, dtype=np.int64))
    assert int(cm.sum()) == y_true.size


def test_confusion_matrix_known_three_class_example() -> None:
    """Hand-computed 3-class confusion matrix."""
    n_classes = 3
    y_true = np.array([0, 0, 1, 1, 2, 2], dtype=np.int64)
    y_pred = np.array([0, 1, 1, 2, 2, 0], dtype=np.int64)
    # Rows = true class, cols = predicted class.
    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
            [1, 0, 1],
        ],
        dtype=np.int64,
    )
    cm = confusion_matrix(y_true, y_pred, n_classes)
    np.testing.assert_array_equal(cm, expected)


# ---------------------------------------------------------------------------
# expected_calibration_error
# ---------------------------------------------------------------------------


def test_ece_perfect_calibration_is_zero() -> None:
    """When confidence equals accuracy in every populated bin, ECE = 0."""
    # 20 samples all predicted class 0 with confidence 1.0 and all correct:
    # the only populated bin has acc=1 and conf=1, gap=0.
    n = 20
    y_true = np.zeros(n, dtype=np.int64)
    y_prob = np.zeros((n, 2), dtype=np.float64)
    y_prob[:, 0] = 1.0
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    assert ece == pytest.approx(0.0, abs=1e-12)


def test_ece_worst_case_all_confident_all_wrong() -> None:
    """All samples predicted with conf=1 but all wrong → ECE = 1."""
    n = 10
    y_true = np.zeros(n, dtype=np.int64)
    y_prob = np.zeros((n, 2), dtype=np.float64)
    y_prob[:, 1] = 1.0  # always predicts class 1, truth is class 0
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    assert ece == pytest.approx(1.0, abs=1e-12)


def test_ece_known_small_example() -> None:
    """10 samples, max-confidences split 0.9/0.1, all correct.

    Confidence is the max softmax probability, so the low-confidence half is
    constructed as a uniform 10-class distribution (max = 0.1), and the
    high-confidence half puts 0.9 on the true class and 0.1/9 on the rest.

    With 15 equal-width bins on [0, 1] (bin width = 1/15 ≈ 0.0667):
      - confidence 0.9 falls into bin index 13 (edges [13/15, 14/15)).
      - confidence 0.1 falls into bin index 1  (edges [1/15, 2/15)).
    Each bin has 5 samples, all correct (argmax = class 0 for both halves):
      bin 1:  |acc - conf| = |1.0 - 0.1| = 0.9, weight 5/10 = 0.5 → 0.45
      bin 13: |acc - conf| = |1.0 - 0.9| = 0.1, weight 5/10 = 0.5 → 0.05
    Total ECE = 0.50.
    """
    n_classes = 10
    y_prob = np.zeros((10, n_classes), dtype=np.float64)
    # High-confidence half: 0.9 on class 0, 0.1/9 elsewhere → max = 0.9.
    y_prob[:5, 0] = 0.9
    y_prob[:5, 1:] = 0.1 / 9.0
    # Low-confidence half: uniform 0.1 on all 10 classes → max = 0.1.
    # np.argmax breaks ties at the first index, so the prediction is class 0.
    y_prob[5:, :] = 1.0 / n_classes
    y_true = np.zeros(10, dtype=np.int64)  # all correct
    ece = expected_calibration_error(y_true, y_prob, n_bins=15)
    assert ece == pytest.approx(0.5, abs=1e-6)


# ---------------------------------------------------------------------------
# stratified_kfold_split
# ---------------------------------------------------------------------------


def test_stratified_kfold_coverage() -> None:
    """Every sample appears in exactly one test fold."""
    rng = np.random.default_rng(0)
    labels = rng.integers(0, 5, size=500)
    k = 5
    folds = stratified_kfold_split(labels, k=k, seed=42)
    all_test = np.concatenate([test_idx for _, test_idx in folds])
    assert all_test.size == labels.size
    np.testing.assert_array_equal(np.sort(all_test), np.arange(labels.size))


def test_stratified_kfold_test_fold_sizes_balanced() -> None:
    """Test fold sizes differ by at most one sample.

    Uses a perfectly balanced dataset (class size divisible by ``k``), so
    every class contributes an exactly equal chunk to every fold and the
    per-class remainder effects cancel out.
    """
    n_classes = 4
    per_class = 50  # 50 % 5 == 0 → no remainder per class
    labels = np.repeat(np.arange(n_classes), per_class)
    k = 5
    folds = stratified_kfold_split(labels, k=k, seed=7)
    sizes = np.array([test_idx.size for _, test_idx in folds])
    assert sizes.max() - sizes.min() <= 1


def test_stratified_kfold_preserves_class_proportions() -> None:
    """Class proportions in each test fold roughly match the overall mix."""
    n_classes = 4
    per_class = 100
    labels = np.repeat(np.arange(n_classes), per_class)  # perfectly balanced
    expected_prop = 1.0 / n_classes
    k = 5
    folds = stratified_kfold_split(labels, k=k, seed=123)
    for _, test_idx in folds:
        counts = np.bincount(labels[test_idx], minlength=n_classes)
        props = counts / counts.sum()
        # 10 % tolerance: each class proportion is within 0.1 of expected.
        assert np.all(np.abs(props - expected_prop) < 0.1)
