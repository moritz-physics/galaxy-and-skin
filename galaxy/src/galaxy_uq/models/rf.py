"""Random Forest wrapper matching the nested-CV model interface.

The class is a thin shim around :class:`sklearn.ensemble.RandomForestClassifier`
so that :func:`galaxy_uq.cv.nested_cv` — which expects ``fit``, ``predict`` and
``predict_proba`` — can drive it identically to our from-scratch logistic
regression.
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    """Sklearn Random Forest with the project's model-factory signature.

    Parameters
    ----------
    n_estimators : int, optional
        Number of trees. Defaults to ``100``.
    max_depth : int or None, optional
        Maximum tree depth; ``None`` lets trees grow until all leaves are
        pure or contain ``min_samples_leaf`` samples. Defaults to ``None``.
    min_samples_leaf : int, optional
        Minimum samples per leaf. Defaults to ``1``.
    class_weight : str or None, optional
        Passed through to sklearn. Defaults to ``"balanced"`` so rare
        classes (e.g. Cigar Shaped Smooth) get adequate splitting attention.
    seed : int, optional
        Seed forwarded to sklearn's ``random_state``. Defaults to ``0``.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int | None = None,
        min_samples_leaf: int = 1,
        class_weight: str | None = "balanced",
        seed: int = 0,
    ) -> None:
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_leaf = int(min_samples_leaf)
        self.class_weight = class_weight
        self.seed = int(seed)
        # n_jobs=-1 keeps fold runs honest about wall-clock by exploiting all
        # cores; the algorithm itself is unchanged.
        self._model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            class_weight=self.class_weight,
            random_state=self.seed,
            n_jobs=-1,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RandomForestModel":
        """Fit the underlying sklearn forest.

        Parameters
        ----------
        X : numpy.ndarray
            Training matrix of shape ``(N, D)``.
        y : numpy.ndarray
            Integer class labels of shape ``(N,)``.

        Returns
        -------
        RandomForestModel
            ``self``.
        """
        self._model.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return class labels."""
        return self._model.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(N, K)`` probability matrix."""
        return self._model.predict_proba(X)

    @property
    def feature_importances_(self) -> np.ndarray:
        """Mean impurity-based feature importances from the fitted forest."""
        return self._model.feature_importances_
