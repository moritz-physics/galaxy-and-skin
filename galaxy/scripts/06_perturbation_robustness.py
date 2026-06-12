"""Perturbation robustness of the deep-ensemble CNN.

Inference-only experiment: a single deep ensemble is trained once on the
non-holdout fraction of Galaxy10, then perturbations of four types (Gaussian
blur, additive noise, central occlusion, rotation) are applied at five
severity levels to the holdout images. The ensemble's accuracy, ECE, NLL
and mean predictive entropy are tracked as severity increases. The
logistic-regression and random-forest baselines appear as horizontal
reference lines at their severity-zero values, taken from their saved
nested-CV predictions filtered to the holdout indices.

Run from the project root::

    uv run python scripts/06_perturbation_robustness.py

The trained ensemble is cached to ``results/metrics/perturbation_model.pt``;
re-runs skip training and reload from that file.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter, rotate

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.cv import stratified_kfold_split  # noqa: E402
from galaxy_uq.data import load_galaxy10  # noqa: E402
from galaxy_uq.metrics import (  # noqa: E402
    accuracy,
    expected_calibration_error,
    negative_log_likelihood,
)
from galaxy_uq.models.cnn import DeepEnsemble, GalaxyCNN, get_device  # noqa: E402
from galaxy_uq.preprocessing import prepare_images_for_cnn  # noqa: E402

logger = logging.getLogger("06_perturbation")

SAVE_KWARGS: dict = {"dpi": 150, "bbox_inches": "tight"}

SEED: int = 42
IMG_SIZE: int = 64
ENSEMBLE_MEMBERS: int = 5
EPOCHS: int = 20
BEST_LR: float = 0.001
BEST_WD: float = 1e-4


def parse_args() -> argparse.Namespace:
    """Command-line interface; ``--fast`` overrides the heavy defaults."""
    p = argparse.ArgumentParser(
        description="Galaxy10 perturbation robustness experiment"
    )
    p.add_argument(
        "--fast", action="store_true",
        help="smoke-test mode: img_size=32, epochs=5, ensemble_members=2; "
             "uses a separate cache file perturbation_model_fast.pt",
    )
    return p.parse_args()

BLUR_SIGMAS: tuple[float, ...] = (0.0, 1.0, 2.0, 4.0, 8.0)
NOISE_SIGMAS: tuple[float, ...] = (0.0, 10.0, 25.0, 50.0, 100.0)
OCCLUSION_SIZES: tuple[int, ...] = (0, 32, 64, 96, 128)
ROTATION_ANGLES: tuple[float, ...] = (0.0, 15.0, 45.0, 90.0, 180.0)


def gaussian_blur(image: np.ndarray, sigma: float) -> np.ndarray:
    """Per-channel isotropic Gaussian blur of a uint8 image.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(H, W, 3)`` and dtype ``uint8``.
    sigma : float
        Standard deviation of the Gaussian kernel in pixels. ``0`` returns
        the image unchanged.

    Returns
    -------
    numpy.ndarray
        Blurred uint8 image of the same shape.
    """
    if sigma == 0.0:
        return image
    out = np.empty_like(image)
    for c in range(image.shape[2]):
        out[..., c] = gaussian_filter(image[..., c], sigma=sigma)
    return out


def additive_noise(
    image: np.ndarray, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Add zero-mean Gaussian noise to a uint8 image and clip to ``[0, 255]``.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(H, W, 3)`` and dtype ``uint8``.
    sigma : float
        Noise standard deviation on the ``0-255`` pixel intensity scale.
    rng : numpy.random.Generator
        Random source for reproducibility.

    Returns
    -------
    numpy.ndarray
        Noisy uint8 image of the same shape.
    """
    if sigma == 0.0:
        return image
    noisy = image.astype(np.float32) + rng.normal(
        0.0, sigma, size=image.shape
    ).astype(np.float32)
    return np.clip(noisy, 0.0, 255.0).astype(np.uint8)


def random_occlusion(image: np.ndarray, size: int) -> np.ndarray:
    """Black-square occlusion centred on the image.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(H, W, 3)`` and dtype ``uint8``.
    size : int
        Side length of the square occlusion in pixels. ``0`` returns the
        image unchanged.

    Returns
    -------
    numpy.ndarray
        Occluded uint8 image.
    """
    if size == 0:
        return image
    h, w = image.shape[:2]
    cy, cx = h // 2, w // 2
    half = size // 2
    y0, y1 = max(0, cy - half), min(h, cy + half)
    x0, x1 = max(0, cx - half), min(w, cx + half)
    out = image.copy()
    out[y0:y1, x0:x1, :] = 0
    return out


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    """Rotate a uint8 image by ``angle`` degrees with zero padding.

    Parameters
    ----------
    image : numpy.ndarray
        Image of shape ``(H, W, 3)`` and dtype ``uint8``.
    angle : float
        Rotation angle in degrees, counter-clockwise.

    Returns
    -------
    numpy.ndarray
        Rotated uint8 image of the same shape.
    """
    if angle == 0.0:
        return image
    rotated = rotate(
        image, angle=angle, axes=(0, 1), reshape=False,
        mode="constant", cval=0, order=1,
    )
    return np.clip(rotated, 0, 255).astype(np.uint8)


def apply_perturbation(
    images: np.ndarray, name: str, severity: float, rng: np.random.Generator
) -> np.ndarray:
    """Apply one perturbation to a batch of uint8 images."""
    out = np.empty_like(images)
    for i in range(images.shape[0]):
        if name == "blur":
            out[i] = gaussian_blur(images[i], float(severity))
        elif name == "noise":
            out[i] = additive_noise(images[i], float(severity), rng)
        elif name == "occlusion":
            out[i] = random_occlusion(images[i], int(severity))
        elif name == "rotation":
            out[i] = rotate_image(images[i], float(severity))
        else:
            raise ValueError(f"unknown perturbation: {name}")
    return out


def ensemble_predict(
    members: list[GalaxyCNN], X: np.ndarray, device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Mean softmax probabilities over ensemble members.

    Parameters
    ----------
    members : list of GalaxyCNN
        Trained ensemble members.
    X : numpy.ndarray
        Image tensor of shape ``(N, 3, H, W)`` in ``float32 ∈ [0, 1]``.
    device : torch.device
        Device used for the forward pass.
    batch_size : int, optional
        Mini-batch size. Defaults to ``256``.

    Returns
    -------
    numpy.ndarray
        Probability matrix of shape ``(N, 10)``.
    """
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    n = X_t.shape[0]
    accum = np.zeros((n, 10), dtype=np.float64)
    for model in members:
        model.eval()
        outs: list[np.ndarray] = []
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = X_t[i : i + batch_size].to(device, non_blocking=True)
                p = F.softmax(model(xb), dim=1).cpu().numpy()
                outs.append(p)
        accum += np.concatenate(outs, axis=0)
    return (accum / len(members)).astype(np.float64)


def predictive_entropy(probs: np.ndarray) -> float:
    """Mean Shannon entropy (nats) of a probability matrix."""
    p = np.clip(probs, 1e-12, 1.0)
    return float(np.mean(-np.sum(p * np.log(p), axis=1)))


def evaluate_probs(y_true: np.ndarray, probs: np.ndarray) -> dict:
    """Accuracy, ECE, NLL and mean predictive entropy for one prediction set."""
    y_pred = np.argmax(probs, axis=1)
    return {
        "acc": accuracy(y_true, y_pred),
        "ece": expected_calibration_error(y_true, probs, n_bins=15),
        "nll": negative_log_likelihood(y_true, probs),
        "entropy": predictive_entropy(probs),
    }


def train_or_load_ensemble(
    images_raw: np.ndarray, labels: np.ndarray, holdout_idx: np.ndarray,
    model_path: Path,
) -> list[GalaxyCNN]:
    """Train (or reload from disk) the ensemble used for perturbation eval.

    Caches the trained members to ``model_path``; subsequent runs skip
    training and reload directly.
    """
    device = get_device()
    if model_path.exists():
        logger.info("Loading cached ensemble from %s", model_path)
        members: list[GalaxyCNN] = torch.load(
            model_path, map_location=device, weights_only=False,
        )
        for m in members:
            m.to(device).eval()
        return members

    logger.info("Cached ensemble not found; training fresh ensemble.")
    train_mask = np.ones(labels.size, dtype=bool)
    train_mask[holdout_idx] = False
    train_idx = np.flatnonzero(train_mask)
    logger.info(
        "Training partition: %d samples; holdout: %d samples",
        train_idx.size, holdout_idx.size,
    )

    X_train_raw = images_raw[train_idx]
    y_train = labels[train_idx].astype(np.int64)
    logger.info("Preprocessing training images to %dx%d...", IMG_SIZE, IMG_SIZE)
    X_train = prepare_images_for_cnn(X_train_raw, target_size=IMG_SIZE)
    del X_train_raw

    ensemble = DeepEnsemble(
        n_members=ENSEMBLE_MEMBERS,
        lr=BEST_LR,
        weight_decay=BEST_WD,
        epochs=EPOCHS,
        seed=SEED,
    )
    t0 = time.perf_counter()
    ensemble.fit(X_train, y_train)
    logger.info(
        "Ensemble trained in %.1f min", (time.perf_counter() - t0) / 60.0
    )

    model_path.parent.mkdir(parents=True, exist_ok=True)
    cpu_members = [m.cpu() for m in ensemble.members]
    torch.save(cpu_members, model_path)
    logger.info("Saved ensemble to %s", model_path)
    for m in cpu_members:
        m.to(device).eval()
    return cpu_members


def _baseline_metrics_from_saved(
    pred_path: Path, holdout_idx: np.ndarray, y_holdout: np.ndarray
) -> dict:
    """Compute the four metrics for a saved baseline on the holdout subset."""
    arr = np.load(pred_path)
    probs = arr["y_prob"][holdout_idx]
    y_true_saved = arr["y_true"][holdout_idx]
    if not np.array_equal(y_true_saved, y_holdout):
        raise AssertionError(f"label mismatch for {pred_path}")
    return evaluate_probs(y_holdout, probs)


def plot_per_perturbation(
    pert_name: str, x_label: str, severities: tuple, cnn_curve: list[dict],
    lr_base: dict, rf_base: dict, out_path: Path,
) -> None:
    """2x2 panel: acc, ECE, NLL, entropy vs severity for one perturbation."""
    metrics_order = [
        ("acc", "Accuracy"),
        ("ece", "ECE"),
        ("nll", "NLL"),
        ("entropy", "Mean predictive entropy (nats)"),
    ]
    xs = np.asarray(severities, dtype=np.float64)
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    cnn_acc = np.array([s["acc"] for s in cnn_curve])
    baseline_min = min(lr_base["acc"], rf_base["acc"])
    below = np.where(cnn_acc < baseline_min)[0]
    cross_sev: float | None = float(xs[below[0]]) if below.size else None

    for ax, (key, title) in zip(axes.flat, metrics_order):
        ys = [s[key] for s in cnn_curve]
        ax.plot(xs, ys, color="purple", marker="o", linewidth=2,
                label="CNN ensemble")
        ax.axhline(lr_base[key], color="tab:blue", linestyle="--",
                   label="LR (severity=0)")
        ax.axhline(rf_base[key], color="seagreen", linestyle="--",
                   label="RF (severity=0)")
        if cross_sev is not None and key == "acc":
            ax.axvline(
                cross_sev, color="firebrick", linestyle=":", alpha=0.7,
                label=f"CNN<baseline @ {cross_sev:g}",
            )
        ax.set_xlabel(x_label)
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Perturbation robustness — {pert_name}", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def plot_summary(
    results: dict, baselines: dict, configs: list[tuple], out_path: Path
) -> None:
    """2x4 compact summary: rows accuracy & ECE, columns over perturbations."""
    fig, axes = plt.subplots(2, 4, figsize=(18, 8))
    for col, (name, x_label, severities) in enumerate(configs):
        xs = np.asarray(severities, dtype=np.float64)
        curve = results[name]
        for row, key in enumerate(("acc", "ece")):
            ax = axes[row, col]
            ys = [curve[s][key] for s in severities]
            ax.plot(xs, ys, color="purple", marker="o", linewidth=2, label="CNN")
            ax.axhline(baselines["LR"][key], color="tab:blue",
                       linestyle="--", label="LR")
            ax.axhline(baselines["RF"][key], color="seagreen",
                       linestyle="--", label="RF")
            ax.set_xlabel(x_label)
            ax.set_ylabel("Accuracy" if key == "acc" else "ECE")
            ax.set_title(f"{name} — {'Accuracy' if key == 'acc' else 'ECE'}")
            ax.grid(True, alpha=0.3)
            if row == 0 and col == 0:
                ax.legend(fontsize=8)
    fig.suptitle("Perturbation robustness — summary", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, **SAVE_KWARGS)
    plt.close(fig)


def main() -> None:
    """Train (or load) the ensemble, run the perturbation grid, save outputs."""
    args = parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    global IMG_SIZE, EPOCHS, ENSEMBLE_MEMBERS
    if args.fast:
        IMG_SIZE = 32
        EPOCHS = 5
        ENSEMBLE_MEMBERS = 2
        logger.info(
            "FAST mode: img_size=32, epochs=5, ensemble_members=2"
        )

    figures_dir = Path("results") / "figures"
    metrics_dir = Path("results") / "metrics"
    figures_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)

    device = get_device()
    logger.info("Torch device: %s", device)

    images_raw, labels, _ = load_galaxy10()
    labels = labels.astype(np.int64)

    # Sanity check that the three baselines were evaluated on identical y_true.
    lr_pred_path = metrics_dir / "03_logreg_predictions.npz"
    rf_pred_path = metrics_dir / "04_rf_predictions.npz"
    cnn_pred_path = metrics_dir / "05_cnn_predictions.npz"
    lr_arr = np.load(lr_pred_path)
    rf_arr = np.load(rf_pred_path)
    cnn_arr = np.load(cnn_pred_path)
    if not (
        np.array_equal(lr_arr["y_true"], rf_arr["y_true"])
        and np.array_equal(lr_arr["y_true"], cnn_arr["y_true"])
    ):
        raise AssertionError("baseline y_true arrays disagree")
    logger.info("Baseline label arrays match across LR, RF, CNN.")

    # The 10% stratified holdout is the first fold's test split.
    folds = stratified_kfold_split(labels, k=10, seed=SEED)
    _, holdout_idx = folds[0]
    y_holdout = labels[holdout_idx]
    logger.info("Holdout size: %d", holdout_idx.size)

    model_path = metrics_dir / (
        "perturbation_model_fast.pt" if args.fast else "perturbation_model.pt"
    )
    members = train_or_load_ensemble(images_raw, labels, holdout_idx, model_path)

    baselines = {
        "LR": _baseline_metrics_from_saved(lr_pred_path, holdout_idx, y_holdout),
        "RF": _baseline_metrics_from_saved(rf_pred_path, holdout_idx, y_holdout),
    }
    logger.info("LR baseline (holdout): %s", baselines["LR"])
    logger.info("RF baseline (holdout): %s", baselines["RF"])

    holdout_raw = images_raw[holdout_idx]
    # Free the full 11.6 GB uint8 tensor; the holdout subset is now retained.
    del images_raw

    configs: list[tuple] = [
        ("blur", "Gaussian blur σ (pixels)", BLUR_SIGMAS),
        ("noise", "Additive noise σ (pixel intensity)", NOISE_SIGMAS),
        ("occlusion", "Occlusion size (pixels)", OCCLUSION_SIZES),
        ("rotation", "Rotation angle (degrees)", ROTATION_ANGLES),
    ]
    rng = np.random.default_rng(SEED)
    results: dict = {}
    t_eval = time.perf_counter()
    for name, _x_label, severities in configs:
        results[name] = {}
        for sev in severities:
            t0 = time.perf_counter()
            perturbed = apply_perturbation(holdout_raw, name, sev, rng)
            X = prepare_images_for_cnn(perturbed, target_size=IMG_SIZE)
            probs = ensemble_predict(members, X, device)
            metrics = evaluate_probs(y_holdout, probs)
            results[name][sev] = metrics
            logger.info(
                "%-9s severity=%-6s acc=%.4f ece=%.4f nll=%.4f H=%.4f (%.1fs)",
                name, str(sev), metrics["acc"], metrics["ece"],
                metrics["nll"], metrics["entropy"], time.perf_counter() - t0,
            )
    logger.info(
        "All perturbation evaluations done in %.1f min",
        (time.perf_counter() - t_eval) / 60.0,
    )

    for name, x_label, severities in configs:
        curve = [results[name][s] for s in severities]
        plot_per_perturbation(
            name, x_label, severities, curve,
            baselines["LR"], baselines["RF"],
            figures_dir / f"06_{name}.png",
        )
        logger.info("Wrote %s", figures_dir / f"06_{name}.png")

    plot_summary(results, baselines, configs, figures_dir / "06_summary.png")
    logger.info("Wrote %s", figures_dir / "06_summary.png")

    json_payload = {
        "seed": SEED,
        "img_size": IMG_SIZE,
        "ensemble_members": ENSEMBLE_MEMBERS,
        "epochs": EPOCHS,
        "best_hp": {"lr": BEST_LR, "weight_decay": BEST_WD},
        "holdout_size": int(holdout_idx.size),
        "baselines": baselines,
        "perturbations": {
            name: {str(s): results[name][s] for s in severities}
            for name, _x_label, severities in configs
        },
    }
    json_path = metrics_dir / "06_perturbation_results.json"
    json_path.write_text(json.dumps(json_payload, indent=2))
    logger.info("Wrote %s", json_path)

    logger.info("")
    logger.info("Perturbation    | Severity | Acc   | ECE   | Entropy")
    logger.info("----------------|----------|-------|-------|--------")
    for name, _x_label, severities in configs:
        for s in severities:
            m = results[name][s]
            logger.info(
                "%-15s | %-8s | %.3f | %.3f | %.3f",
                name, str(s), m["acc"], m["ece"], m["entropy"],
            )

    logger.info("")
    baseline_min_acc = min(baselines["LR"]["acc"], baselines["RF"]["acc"])
    for name, _x_label, severities in configs:
        cross: float | None = None
        for s in severities:
            if results[name][s]["acc"] < baseline_min_acc:
                cross = float(s)
                break
        if cross is not None:
            logger.info(
                "CNN accuracy crosses LR/RF baseline at severity %g for perturbation %s",
                cross, name,
            )
        else:
            logger.info(
                "CNN accuracy stays above LR/RF baseline across all severities for %s",
                name,
            )


if __name__ == "__main__":
    main()
