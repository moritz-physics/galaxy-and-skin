"""Stratified k-fold and nested cross-validation, from scratch.

The nested loop is the workhorse for all three models in this project. It
selects hyperparameters on an inner CV split of the outer training fold and
evaluates on the held-out outer fold, so reported metrics are not biased by
hyperparameter tuning. Standardisation is refit per fold to avoid leaking
test-set statistics into training.
"""

from __future__ import annotations

import logging
from typing import Callable

import numpy as np

from galaxy_uq.metrics import (
    accuracy,
    expected_calibration_error,
    macro_f1,
    negative_log_likelihood,
)

logger = logging.getLogger(__name__)


def stratified_kfold_split(
    labels: np.ndarray, k: int, seed: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build ``k`` stratified train/test index pairs.

    Each class's indices are shuffled and split into ``k`` near-equal chunks
    using ``np.array_split``; chunk ``i`` becomes the test set for fold
    ``i`` and the remaining chunks form the training set. If a class has
    fewer than ``k`` samples some folds will have zero test instances of
    that class — those folds simply omit it.

    Parameters
    ----------
    labels : numpy.ndarray
        Integer labels of shape ``(N,)``.
    k : int
        Number of folds.
    seed : int
        Seed for the per-call shuffle.

    Returns
    -------
    list of (numpy.ndarray, numpy.ndarray)
        For each fold a ``(train_indices, test_indices)`` pair sorted in
        ascending order.
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    labels = np.asarray(labels)
    rng = np.random.default_rng(seed)

    classes = np.unique(labels)
    test_per_fold: list[list[np.ndarray]] = [[] for _ in range(k)]
    for c in classes:
        idx = np.flatnonzero(labels == c)
        rng.shuffle(idx)
        # array_split distributes the remainder across the first chunks,
        # which keeps fold sizes within one of each other even when the
        # class count is not divisible by k.
        chunks = np.array_split(idx, k)
        for fold_idx, chunk in enumerate(chunks):
            test_per_fold[fold_idx].append(chunk)

    all_indices = np.arange(labels.size)
    folds: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_idx in range(k):
        test_idx = np.sort(np.concatenate(test_per_fold[fold_idx]))
        mask = np.ones(labels.size, dtype=bool)
        mask[test_idx] = False
        train_idx = all_indices[mask]
        folds.append((train_idx, test_idx))
    return folds


def _standardize_fit(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return per-column mean and safe std for use with :func:`_standardize_apply`."""
    mean = matrix.mean(axis=0)
    std = matrix.std(axis=0)
    std = np.where(std < 1e-12, 1.0, std)
    return mean, std


def _standardize_apply(
    matrix: np.ndarray, mean: np.ndarray, std: np.ndarray
) -> np.ndarray:
    """Apply a previously-fit standardisation."""
    return (matrix - mean) / std


def _hp_key(hp: dict) -> str:
    """Stable, JSON-friendly string key for a hyperparameter dict."""
    return "|".join(f"{k}={hp[k]}" for k in sorted(hp))


def nested_cv(
    X: np.ndarray,
    y: np.ndarray,
    model_factory: Callable,
    hyperparam_grid: list[dict],
    outer_k: int = 10,
    inner_k: int = 3,
    seed: int = 0,
    n_classes: int = 10,
) -> dict:
    """Run nested stratified cross-validation with per-fold standardisation.

    Parameters
    ----------
    X : numpy.ndarray
        Feature matrix of shape ``(N, D)``.
    y : numpy.ndarray
        Integer label array of shape ``(N,)``.
    model_factory : Callable
        Called as ``model_factory(**hyperparams)`` and must return an object
        exposing ``fit``, ``predict`` and ``predict_proba``.
    hyperparam_grid : list of dict
        Hyperparameter combinations to evaluate in the inner loop.
    outer_k : int, optional
        Number of outer folds. Defaults to ``10``.
    inner_k : int, optional
        Number of inner folds. Defaults to ``3``.
    seed : int, optional
        Base seed; outer and inner splits derive deterministic seeds from it.
    n_classes : int, optional
        Number of classes used by metric helpers. Defaults to ``10``.

    Returns
    -------
    dict
        See module docstring of :mod:`galaxy_uq.cv` and the project task
        spec for the full schema.
    """
    X = np.asarray(X)
    y = np.asarray(y)
    if X.shape[0] != y.shape[0]:
        raise ValueError("X and y must have matching first dimension")
    if len(hyperparam_grid) == 0:
        raise ValueError("hyperparam_grid must contain at least one combination")

    outer_folds = stratified_kfold_split(y, outer_k, seed=seed)

    outer_fold_results: list[dict] = []
    fold_accuracies: list[float] = []
    fold_macro_f1s: list[float] = []
    fold_eces: list[float] = []
    fold_nlls: list[float] = []
    seen_test_indices: list[np.ndarray] = []

    for fold_idx, (outer_train, outer_test) in enumerate(outer_folds):
        logger.info(
            "Outer fold %d/%d: %d train, %d test",
            fold_idx + 1,
            outer_k,
            outer_train.size,
            outer_test.size,
        )

        X_outer_train = X[outer_train]
        y_outer_train = y[outer_train]
        X_outer_test = X[outer_test]
        y_outer_test = y[outer_test]

        # Inner CV operates only on the outer training partition; outer_test
        # is never touched until the final evaluation below.
        inner_folds = stratified_kfold_split(
            y_outer_train, inner_k, seed=seed + 1000 * (fold_idx + 1)
        )

        inner_scores: dict[str, list[float]] = {
            _hp_key(hp): [] for hp in hyperparam_grid
        }
        for hp in hyperparam_grid:
            key = _hp_key(hp)
            for inner_fold_idx, (inner_train, inner_val) in enumerate(inner_folds):
                X_in_train = X_outer_train[inner_train]
                y_in_train = y_outer_train[inner_train]
                X_in_val = X_outer_train[inner_val]
                y_in_val = y_outer_train[inner_val]

                mean, std = _standardize_fit(X_in_train)
                X_in_train_s = _standardize_apply(X_in_train, mean, std)
                X_in_val_s = _standardize_apply(X_in_val, mean, std)

                model = model_factory(**hp)
                model.fit(X_in_train_s, y_in_train)
                y_in_pred = model.predict(X_in_val_s)
                score = macro_f1(y_in_val, y_in_pred, n_classes)
                inner_scores[key].append(score)
                logger.debug(
                    "    inner fold %d/%d hp=%s macro_f1=%.4f",
                    inner_fold_idx + 1,
                    inner_k,
                    key,
                    score,
                )

        mean_inner = {k: float(np.mean(v)) for k, v in inner_scores.items()}
        best_key = max(mean_inner, key=mean_inner.get)
        best_hp = next(hp for hp in hyperparam_grid if _hp_key(hp) == best_key)
        logger.info(
            "  best HP=%s (inner mean macro_f1=%.4f)",
            best_key,
            mean_inner[best_key],
        )

        # Retrain on the FULL outer-train partition with the chosen HPs.
        mean, std = _standardize_fit(X_outer_train)
        X_outer_train_s = _standardize_apply(X_outer_train, mean, std)
        X_outer_test_s = _standardize_apply(X_outer_test, mean, std)
        final_model = model_factory(**best_hp)
        final_model.fit(X_outer_train_s, y_outer_train)
        y_pred = final_model.predict(X_outer_test_s)
        y_prob = final_model.predict_proba(X_outer_test_s)

        fold_acc = accuracy(y_outer_test, y_pred)
        fold_f1 = macro_f1(y_outer_test, y_pred, n_classes)
        fold_ece = expected_calibration_error(y_outer_test, y_prob)
        fold_nll = negative_log_likelihood(y_outer_test, y_prob)
        logger.info(
            "  outer fold %d: acc=%.4f macro_f1=%.4f ece=%.4f nll=%.4f",
            fold_idx + 1,
            fold_acc,
            fold_f1,
            fold_ece,
            fold_nll,
        )

        fold_accuracies.append(fold_acc)
        fold_macro_f1s.append(fold_f1)
        fold_eces.append(fold_ece)
        fold_nlls.append(fold_nll)

        outer_fold_results.append(
            {
                "fold_idx": fold_idx,
                "best_hyperparams": dict(best_hp),
                "y_test": y_outer_test,
                "y_pred": y_pred,
                "y_prob": y_prob,
                "test_indices": outer_test,
                "inner_scores": {k: list(v) for k, v in inner_scores.items()},
            }
        )
        seen_test_indices.append(outer_test)

    # Coverage assertion: each sample appears in exactly one outer test fold.
    concatenated_test = np.concatenate(seen_test_indices)
    expected = np.arange(y.size)
    if not np.array_equal(np.sort(concatenated_test), expected):
        missing = np.setdiff1d(expected, concatenated_test)
        duplicated = concatenated_test.size - np.unique(concatenated_test).size
        raise AssertionError(
            "outer test folds do not cover the dataset exactly once "
            f"(missing={missing.size}, duplicates={duplicated})"
        )

    # Concatenate predictions in the order of growing test index so they
    # align with the original dataset row order.
    order = np.argsort(concatenated_test)
    all_y_true = np.concatenate([r["y_test"] for r in outer_fold_results])[order]
    all_y_pred = np.concatenate([r["y_pred"] for r in outer_fold_results])[order]
    all_y_prob = np.concatenate([r["y_prob"] for r in outer_fold_results], axis=0)[
        order
    ]

    aggregate = {
        "accuracy": {
            "mean": float(np.mean(fold_accuracies)),
            "std": float(np.std(fold_accuracies)),
        },
        "macro_f1": {
            "mean": float(np.mean(fold_macro_f1s)),
            "std": float(np.std(fold_macro_f1s)),
        },
        "ece": {
            "mean": float(np.mean(fold_eces)),
            "std": float(np.std(fold_eces)),
        },
        "nll": {
            "mean": float(np.mean(fold_nlls)),
            "std": float(np.std(fold_nlls)),
        },
    }
    logger.info(
        "Aggregate: acc=%.4f±%.4f f1=%.4f±%.4f ece=%.4f±%.4f nll=%.4f±%.4f",
        aggregate["accuracy"]["mean"],
        aggregate["accuracy"]["std"],
        aggregate["macro_f1"]["mean"],
        aggregate["macro_f1"]["std"],
        aggregate["ece"]["mean"],
        aggregate["ece"]["std"],
        aggregate["nll"]["mean"],
        aggregate["nll"]["std"],
    )

    return {
        "outer_fold_results": outer_fold_results,
        "aggregate_metrics": aggregate,
        "all_y_true": all_y_true,
        "all_y_pred": all_y_pred,
        "all_y_prob": all_y_prob,
    }
