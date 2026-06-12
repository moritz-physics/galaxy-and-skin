"""Multinomial logistic regression in pure NumPy.

Implements softmax regression with L2 regularisation, fit by full-batch
gradient descent. The bias is implemented by prepending a column of ones
to the design matrix and excluding the bias row from the L2 penalty.
"""

from __future__ import annotations

import numpy as np


def _softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable row-wise softmax."""
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


class MultinomialLogisticRegression:
    """Softmax regression with L2 regularisation, fit by gradient descent.

    Parameters
    ----------
    C : float, optional
        Inverse regularisation strength, mirroring sklearn semantics:
        larger ``C`` means weaker penalty. Defaults to ``1.0``.
    lr : float, optional
        Gradient-descent learning rate. Defaults to ``0.01``.
    max_iter : int, optional
        Maximum number of gradient steps. Defaults to ``1000``.
    tol : float, optional
        Convergence threshold on the Frobenius norm of the gradient. Defaults
        to ``1e-6``.

    Attributes
    ----------
    W_ : numpy.ndarray
        Weight matrix of shape ``(D + 1, K)`` where row 0 is the bias.
    classes_ : numpy.ndarray
        Sorted unique class labels seen during fit.
    n_iter_ : int
        Number of gradient-descent iterations actually run.
    """

    def __init__(
        self,
        C: float = 1.0,
        lr: float = 0.01,
        max_iter: int = 1000,
        tol: float = 1e-6,
    ) -> None:
        if C <= 0:
            raise ValueError(f"C must be positive, got {C}")
        if lr <= 0:
            raise ValueError(f"lr must be positive, got {lr}")
        self.C = float(C)
        self.lr = float(lr)
        self.max_iter = int(max_iter)
        self.tol = float(tol)

        self.W_: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.n_iter_: int = 0

    @staticmethod
    def _add_bias(X: np.ndarray) -> np.ndarray:
        return np.hstack([np.ones((X.shape[0], 1), dtype=X.dtype), X])

    def fit(self, X: np.ndarray, y: np.ndarray) -> "MultinomialLogisticRegression":
        """Fit the model with gradient descent.

        Parameters
        ----------
        X : numpy.ndarray
            Training matrix of shape ``(N, D)``.
        y : numpy.ndarray
            Integer class labels of shape ``(N,)``.

        Returns
        -------
        MultinomialLogisticRegression
            ``self``, fitted in place.
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y, dtype=np.int64)
        n, _ = X.shape

        self.classes_ = np.unique(y)
        k = self.classes_.size
        # Map labels to dense column indices; lets the class handle non-zero-
        # based label sets even though Galaxy10 already uses 0..K-1.
        class_to_idx = {c: i for i, c in enumerate(self.classes_)}
        y_idx = np.array([class_to_idx[c] for c in y], dtype=np.int64)
        Y = np.zeros((n, k), dtype=np.float64)
        Y[np.arange(n), y_idx] = 1.0

        Xb = self._add_bias(X)  # (N, D+1)
        d_plus_1 = Xb.shape[1]
        W = np.zeros((d_plus_1, k), dtype=np.float64)

        # Penalty coefficient. We use sklearn's C semantics: sklearn's loss is
        # ``0.5 ||W||² + C · Σ CE``, which when normalised by N is equivalent
        # to ``mean(CE) + (1/(2NC)) ||W||²``. With that normalisation the
        # gradient is ``(1/N) [Xᵀ(P − Y) + (1/C) W]`` — i.e. the bracketed term
        # matches the spec's "(1/C) W" regularisation gradient, divided by N
        # along with the data term. Without the 1/N on the penalty, our
        # optimum is N× more regularised than sklearn at the same C.
        reg_coef = 1.0 / self.C

        self.n_iter_ = 0
        for it in range(1, self.max_iter + 1):
            logits = Xb @ W
            probs = _softmax(logits)
            grad = Xb.T @ (probs - Y)
            grad[1:] += reg_coef * W[1:]
            grad /= n

            grad_norm = float(np.linalg.norm(grad))
            self.n_iter_ = it
            if grad_norm < self.tol:
                break
            W -= self.lr * grad

        self.W_ = W
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return ``(N, K)`` class probabilities via softmax.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix of shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            Probability matrix of shape ``(N, K)``.
        """
        if self.W_ is None:
            raise RuntimeError("model is not fit; call fit() first")
        X = np.asarray(X, dtype=np.float64)
        return _softmax(self._add_bias(X) @ self.W_)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels.

        Parameters
        ----------
        X : numpy.ndarray
            Input matrix of shape ``(N, D)``.

        Returns
        -------
        numpy.ndarray
            Integer class labels of shape ``(N,)``.
        """
        if self.classes_ is None:
            raise RuntimeError("model is not fit; call fit() first")
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]
