"""Small CNN classifier and deep-ensemble wrapper.

The architecture is a four-block VGG-style CNN designed to fit on an Apple M1
class machine in a couple of hours when bundled into a five-model ensemble.

:class:`DeepEnsemble` follows Lakshminarayanan et al., *Simple and Scalable
Predictive Uncertainty Estimation using Deep Ensembles* (NeurIPS 2017): the
ensemble is built by training independent networks from different random
initialisations on the same data, with class weights computed from the
training labels and image-space augmentation (per-sample horizontal flip,
batch-uniform 90° rotation) applied during training only.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from galaxy_uq.cv import stratified_kfold_split

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """Return the best available device: MPS on Apple Silicon else CPU."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class GalaxyCNN(nn.Module):
    """Four-block convolutional classifier; returns raw logits.

    Parameters
    ----------
    n_classes : int, optional
        Number of output classes. Defaults to ``10``.
    dropout : float, optional
        Dropout probability applied in the classifier head. Defaults to
        ``0.3``.
    """

    def __init__(self, n_classes: int = 10, dropout: float = 0.3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(2),
        )
        # Final feature map: 128 × 2 × 2 = 512 flat.
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(128 * 4, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, n_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def _augment_batch(x: torch.Tensor) -> torch.Tensor:
    """Per-sample horizontal flip + batch-uniform random 90° rotation.

    Galaxies have no canonical orientation so multiples of 90° are exact
    label-preserving augmentations. Per-sample flip is implemented by
    masking; rotation is applied once per batch to keep things fast while
    still exposing the network to a different orientation each step.
    """
    mask = torch.rand(x.shape[0], device=x.device) < 0.5
    flipped = torch.flip(x, dims=[3])
    x = torch.where(mask[:, None, None, None], flipped, x)
    k = int(torch.randint(0, 4, (1,)).item())
    if k > 0:
        x = torch.rot90(x, k=k, dims=[2, 3])
    return x


def _stratified_val_split(
    y: np.ndarray, val_fraction: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Take a single stratified hold-out from training data.

    Implemented by reusing :func:`stratified_kfold_split` with ``k = 1 /
    val_fraction`` and pulling the first fold; this guarantees identical
    fold construction logic and stratification rules as the outer/inner CV
    splits, so the ensemble's internal "early-stopping val" matches the
    project's stratification convention.
    """
    k = int(round(1.0 / val_fraction))
    folds = stratified_kfold_split(y, k=k, seed=seed)
    train_idx, val_idx = folds[0]
    return train_idx, val_idx


def _compute_class_weights(y: np.ndarray, n_classes: int) -> torch.Tensor:
    """Inverse-frequency weights normalised to sum to ``n_classes``."""
    counts = np.bincount(y, minlength=n_classes).astype(np.float64)
    counts = np.where(counts < 1, 1.0, counts)
    inv = 1.0 / counts
    weights = inv * (n_classes / inv.sum())
    return torch.tensor(weights, dtype=torch.float32)


class DeepEnsemble:
    """A deep ensemble of :class:`GalaxyCNN` networks.

    Parameters
    ----------
    n_members : int, optional
        Number of independently-initialised CNNs to train. Defaults to ``5``.
    lr : float, optional
        AdamW learning rate. Defaults to ``0.005``.
    weight_decay : float, optional
        AdamW weight decay. Defaults to ``1e-4``.
    epochs : int, optional
        Maximum epochs per member. Defaults to ``20``.
    batch_size : int, optional
        Mini-batch size. Defaults to ``128``.
    dropout : float, optional
        Dropout probability passed to each network. Defaults to ``0.3``.
    n_classes : int, optional
        Output class count. Defaults to ``10``.
    seed : int, optional
        Base seed; member ``m`` is seeded with ``seed + m``. Defaults to ``0``.
    """

    def __init__(
        self,
        n_members: int = 5,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        epochs: int = 20,
        batch_size: int = 256,
        dropout: float = 0.3,
        n_classes: int = 10,
        seed: int = 0,
    ) -> None:
        self.n_members = int(n_members)
        self.lr = float(lr)
        self.weight_decay = float(weight_decay)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.dropout = float(dropout)
        self.n_classes = int(n_classes)
        self.seed = int(seed)

        self.device: torch.device = get_device()
        self.members: list[GalaxyCNN] = []
        self.histories: list[dict] = []

    def _train_member(
        self,
        X_train_t: torch.Tensor,
        y_train_t: torch.Tensor,
        X_val_t: torch.Tensor,
        y_val_t: torch.Tensor,
        class_weights_t: torch.Tensor,
        member_seed: int,
    ) -> tuple[GalaxyCNN, dict]:
        """Train a single CNN with early stopping; return the model + history."""
        torch.manual_seed(member_seed)
        model = GalaxyCNN(n_classes=self.n_classes, dropout=self.dropout).to(
            self.device
        )
        opt = torch.optim.AdamW(
            model.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        loss_fn = nn.CrossEntropyLoss(weight=class_weights_t.to(self.device))

        n_train = X_train_t.shape[0]
        history: dict = {"train_loss": [], "val_loss": []}
        best_val = float("inf")
        best_state: dict | None = None
        patience = 5
        patience_counter = 0

        for _ in range(self.epochs):
            model.train()
            perm = torch.randperm(n_train)
            total = 0.0
            n_batches = 0
            for i in range(0, n_train, self.batch_size):
                idx = perm[i : i + self.batch_size]
                xb = X_train_t[idx].to(self.device, non_blocking=True)
                yb = y_train_t[idx].to(self.device, non_blocking=True)
                xb = _augment_batch(xb)
                opt.zero_grad()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                loss.backward()
                opt.step()
                total += float(loss.item())
                n_batches += 1
            train_loss = total / max(n_batches, 1)

            model.eval()
            total_val = 0.0
            n_val_batches = 0
            with torch.no_grad():
                for i in range(0, X_val_t.shape[0], self.batch_size):
                    xb = X_val_t[i : i + self.batch_size].to(
                        self.device, non_blocking=True
                    )
                    yb = y_val_t[i : i + self.batch_size].to(
                        self.device, non_blocking=True
                    )
                    logits = model(xb)
                    total_val += float(loss_fn(logits, yb).item())
                    n_val_batches += 1
            val_loss = total_val / max(n_val_batches, 1)

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss < best_val - 1e-6:
                best_val = val_loss
                best_state = {
                    k: v.detach().cpu().clone() for k, v in model.state_dict().items()
                }
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break

        if best_state is not None:
            model.load_state_dict(best_state)
        return model, history

    def fit(self, X: np.ndarray, y: np.ndarray) -> "DeepEnsemble":
        """Train ``n_members`` independent CNNs from different seeds.

        Parameters
        ----------
        X : numpy.ndarray
            Image tensor of shape ``(N, 3, H, W)`` in ``float32 ∈ [0, 1]``.
        y : numpy.ndarray
            Integer class labels of shape ``(N,)``.

        Returns
        -------
        DeepEnsemble
            ``self``.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.int64)

        train_idx, val_idx = _stratified_val_split(
            y, val_fraction=0.1, seed=self.seed
        )
        X_train = X[train_idx]
        y_train = y[train_idx]
        X_val = X[val_idx]
        y_val = y[val_idx]
        class_weights = _compute_class_weights(y_train, self.n_classes)

        # Tensors stay on CPU; batches are moved to device individually so
        # peak device memory tracks one batch, not the full fold.
        X_train_t = torch.from_numpy(X_train)
        y_train_t = torch.from_numpy(y_train)
        X_val_t = torch.from_numpy(X_val)
        y_val_t = torch.from_numpy(y_val)

        self.members = []
        self.histories = []
        for m in range(self.n_members):
            logger.info(
                "    training member %d/%d (seed=%d)",
                m + 1, self.n_members, self.seed + m,
            )
            model, history = self._train_member(
                X_train_t, y_train_t, X_val_t, y_val_t, class_weights,
                member_seed=self.seed + m,
            )
            self.members.append(model)
            self.histories.append(history)
        return self

    def _predict_all_members(self, X: np.ndarray) -> np.ndarray:
        """Stack of ``(M, N, K)`` per-member softmax probabilities."""
        X = np.asarray(X, dtype=np.float32)
        X_t = torch.from_numpy(X)
        all_probs: list[np.ndarray] = []
        for model in self.members:
            model.eval()
            batches: list[np.ndarray] = []
            with torch.no_grad():
                for i in range(0, X_t.shape[0], self.batch_size):
                    xb = X_t[i : i + self.batch_size].to(
                        self.device, non_blocking=True
                    )
                    p = F.softmax(model(xb), dim=1).cpu().numpy()
                    batches.append(p)
            all_probs.append(np.concatenate(batches, axis=0))
        return np.stack(all_probs, axis=0)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Mean of per-member softmax probabilities, shape ``(N, K)``."""
        return self._predict_all_members(X).mean(axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Argmax of :meth:`predict_proba`, shape ``(N,)``."""
        return np.argmax(self.predict_proba(X), axis=1)

    def predict_proba_per_member(self, X: np.ndarray) -> np.ndarray:
        """Stack of per-member probabilities, shape ``(M, N, K)``."""
        return self._predict_all_members(X)
