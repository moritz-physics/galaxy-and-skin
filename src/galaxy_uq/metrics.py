"""Classification metrics implemented from scratch.

All functions operate on integer class labels in ``range(n_classes)`` and / or
probability matrices of shape ``(N, K)``. They are intentionally
sklearn-free so the project can claim a from-scratch implementation; sklearn
is only used by the verification script to cross-check these against a
reference implementation.
"""

from __future__ import annotations

import numpy as np

_PROB_EPS: float = 1e-12


def accuracy(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Fraction of predictions equal to the ground truth.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Integer label arrays of equal length.

    Returns
    -------
    float
        Accuracy in ``[0, 1]``.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    if y_true.size == 0:
        return 0.0
    return float(np.mean(y_true == y_pred))


def confusion_matrix(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> np.ndarray:
    """Count matrix with rows = true classes, cols = predicted classes.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Integer label arrays of equal length, values in ``range(n_classes)``.
    n_classes : int
        Number of classes ``K``.

    Returns
    -------
    numpy.ndarray
        Integer matrix of shape ``(n_classes, n_classes)``.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if y_true.shape != y_pred.shape:
        raise ValueError(
            f"shape mismatch: y_true {y_true.shape} vs y_pred {y_pred.shape}"
        )
    # np.bincount on the flattened index t*K + p is the fastest way to build
    # a confusion matrix without a Python loop.
    flat = y_true * n_classes + y_pred
    counts = np.bincount(flat, minlength=n_classes * n_classes)
    return counts.reshape(n_classes, n_classes).astype(np.int64)


def per_class_precision_recall_f1(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> dict:
    """Per-class precision, recall, F1 and support.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Integer label arrays.
    n_classes : int
        Number of classes.

    Returns
    -------
    dict
        Mapping with arrays of length ``n_classes`` under keys
        ``"precision"``, ``"recall"``, ``"f1"`` and ``"support"``.
    """
    cm = confusion_matrix(y_true, y_pred, n_classes)
    tp = np.diag(cm).astype(np.float64)
    predicted_positives = cm.sum(axis=0).astype(np.float64)
    actual_positives = cm.sum(axis=1).astype(np.float64)

    with np.errstate(invalid="ignore", divide="ignore"):
        precision = np.where(predicted_positives > 0, tp / predicted_positives, 0.0)
        recall = np.where(actual_positives > 0, tp / actual_positives, 0.0)
        denom = precision + recall
        f1 = np.where(denom > 0, 2.0 * precision * recall / denom, 0.0)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": actual_positives.astype(np.int64),
    }


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Unweighted mean of per-class F1.

    Parameters
    ----------
    y_true, y_pred : numpy.ndarray
        Integer label arrays.
    n_classes : int
        Number of classes.

    Returns
    -------
    float
        Macro-averaged F1 in ``[0, 1]``.
    """
    return float(per_class_precision_recall_f1(y_true, y_pred, n_classes)["f1"].mean())


def expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> float:
    """Expected Calibration Error following Guo et al. (2017).

    Each sample's predicted class confidence is binned into ``n_bins`` equal
    width bins over ``[0, 1]``. The ECE is the weighted mean absolute gap
    between bin accuracy and bin confidence.

    Parameters
    ----------
    y_true : numpy.ndarray
        Integer labels of shape ``(N,)``.
    y_prob : numpy.ndarray
        Probability matrix of shape ``(N, K)``.
    n_bins : int, optional
        Number of equal-width confidence bins.

    Returns
    -------
    float
        ECE in ``[0, 1]``.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    n = y_true.shape[0]
    if n == 0:
        return 0.0

    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    # np.digitize with right=True puts values at the upper edge into the same
    # bin as values strictly inside it; we additionally clip the bin index
    # so confidence == 1.0 lands in the final bin rather than out of range.
    bin_idx = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, n_bins - 1)

    ece = 0.0
    for b in range(n_bins):
        mask = bin_idx == b
        if not np.any(mask):
            continue
        bin_acc = correct[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += (mask.sum() / n) * abs(bin_acc - bin_conf)
    return float(ece)


def negative_log_likelihood(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Mean per-sample negative log-likelihood.

    Parameters
    ----------
    y_true : numpy.ndarray
        Integer labels of shape ``(N,)``.
    y_prob : numpy.ndarray
        Probability matrix of shape ``(N, K)``.

    Returns
    -------
    float
        Mean NLL averaged over samples.
    """
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    clipped = np.clip(y_prob, _PROB_EPS, 1.0 - _PROB_EPS)
    chosen = clipped[np.arange(y_true.size), y_true]
    return float(-np.mean(np.log(chosen)))


def reliability_diagram_data(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> dict:
    """Per-bin accuracy / confidence / count for a reliability diagram.

    Parameters
    ----------
    y_true : numpy.ndarray
        Integer labels of shape ``(N,)``.
    y_prob : numpy.ndarray
        Probability matrix of shape ``(N, K)``.
    n_bins : int, optional
        Number of equal-width confidence bins.

    Returns
    -------
    dict
        ``bin_midpoints`` (length ``n_bins``), ``bin_accuracies``,
        ``bin_confidences``, and ``bin_counts``. Empty bins have NaN accuracy
        and NaN confidence so the plot can skip them.
    """
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    confidences = y_prob.max(axis=1)
    predictions = y_prob.argmax(axis=1)
    correct = (predictions == y_true).astype(np.float64)

    edges = np.linspace(0.0, 1.0, n_bins + 1)
    midpoints = 0.5 * (edges[:-1] + edges[1:])
    bin_idx = np.clip(np.digitize(confidences, edges[1:-1], right=False), 0, n_bins - 1)

    bin_acc = np.full(n_bins, np.nan)
    bin_conf = np.full(n_bins, np.nan)
    bin_counts = np.zeros(n_bins, dtype=np.int64)
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(mask.sum())
        bin_counts[b] = count
        if count > 0:
            bin_acc[b] = correct[mask].mean()
            bin_conf[b] = confidences[mask].mean()

    return {
        "bin_midpoints": midpoints,
        "bin_accuracies": bin_acc,
        "bin_confidences": bin_conf,
        "bin_counts": bin_counts,
    }
