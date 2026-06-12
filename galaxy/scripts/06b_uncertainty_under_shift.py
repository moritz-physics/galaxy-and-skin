"""Post-hoc analysis of the cached perturbation ensemble.

Closes three gaps that the report previously flagged as limitations:
    (i)   Expected calibration error under shift *with* a temperature
          rescaling, where ``T*`` is fit on clean-data predictions.
    (ii)  Per-perturbation decomposition of the deep-ensemble predictive
          entropy into an aleatoric component (mean per-member entropy)
          and an epistemic component (Jensen gap).
    (iii) Mean ensemble max-softmax confidence under shift, to make the
          "confidently wrong" finding numerical rather than visual.

The cached members at ``results/metrics/perturbation_model_fast.pt`` are
loaded once; for each (family, severity) we re-apply the perturbation,
run the same per-member forward pass, and record the per-member softmaxes.

Run from the project root::

    uv run python scripts/06b_uncertainty_under_shift.py
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from galaxy_uq.cv import stratified_kfold_split  # noqa: E402
from galaxy_uq.data import load_galaxy10  # noqa: E402
from galaxy_uq.metrics import expected_calibration_error  # noqa: E402
from galaxy_uq.models.cnn import GalaxyCNN, get_device  # noqa: E402
from galaxy_uq.preprocessing import prepare_images_for_cnn  # noqa: E402

# Re-using the perturbation primitives from the main script.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from importlib import import_module  # noqa: E402

_p = import_module("06_perturbation_robustness")

apply_perturbation = _p.apply_perturbation
BLUR_SIGMAS = _p.BLUR_SIGMAS
NOISE_SIGMAS = _p.NOISE_SIGMAS
OCCLUSION_SIZES = _p.OCCLUSION_SIZES
ROTATION_ANGLES = _p.ROTATION_ANGLES

SEED = 42
IMG_SIZE = 32  # cached model was trained at this resolution
_EPS = 1e-12


def per_member_softmax(
    members: list[GalaxyCNN], X: np.ndarray, device: torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Return softmaxes of shape ``(M, N, 10)`` for every ensemble member."""
    X_t = torch.from_numpy(np.asarray(X, dtype=np.float32))
    n = X_t.shape[0]
    out = np.zeros((len(members), n, 10), dtype=np.float64)
    for j, model in enumerate(members):
        model.eval()
        with torch.no_grad():
            for i in range(0, n, batch_size):
                xb = X_t[i : i + batch_size].to(device, non_blocking=True)
                p = F.softmax(model(xb), dim=1).cpu().numpy()
                out[j, i : i + xb.shape[0]] = p
    return out


def fit_temperature(probs: np.ndarray, y_true: np.ndarray) -> float:
    """Bounded scalar search for ``T`` minimising clean-data NLL."""
    from scipy.optimize import minimize_scalar

    log_p = np.log(np.clip(probs, _EPS, 1.0))

    def nll(T: float) -> float:
        scaled = log_p / T
        scaled = scaled - scaled.max(axis=1, keepdims=True)
        exp = np.exp(scaled)
        soft = exp / exp.sum(axis=1, keepdims=True)
        chosen = soft[np.arange(y_true.size), y_true]
        return float(-np.mean(np.log(np.clip(chosen, _EPS, 1.0))))

    res = minimize_scalar(nll, bounds=(0.1, 10.0), method="bounded")
    return float(res.x)


def apply_temperature(probs: np.ndarray, T: float) -> np.ndarray:
    log_p = np.log(np.clip(probs, _EPS, 1.0)) / T
    log_p = log_p - log_p.max(axis=1, keepdims=True)
    exp = np.exp(log_p)
    return exp / exp.sum(axis=1, keepdims=True)


def shannon_entropy(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, _EPS, 1.0)
    return -np.sum(p * np.log(p), axis=-1)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    metrics_dir = Path("results/metrics")
    figures_dir = Path("results/figures")
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_path = metrics_dir / "perturbation_model_fast.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"{model_path} not found - run 06_perturbation_robustness.py --fast first."
        )

    device = get_device()
    members: list[GalaxyCNN] = torch.load(
        model_path, map_location=device, weights_only=False
    )
    for m in members:
        m.to(device).eval()
    M = len(members)
    logging.info("Loaded %d-member ensemble on %s", M, device)

    images_raw, labels, _ = load_galaxy10()
    labels = labels.astype(np.int64)
    folds = stratified_kfold_split(labels, k=10, seed=SEED)
    _, holdout_idx = folds[0]
    y_hold = labels[holdout_idx]
    holdout_raw = images_raw[holdout_idx]
    del images_raw

    # Clean-data per-member softmaxes; mean -> ensemble probs used to fit T*.
    X_clean = prepare_images_for_cnn(holdout_raw, target_size=IMG_SIZE)
    clean_perM = per_member_softmax(members, X_clean, device)
    clean_probs = clean_perM.mean(axis=0)
    T_star = fit_temperature(clean_probs, y_hold)
    logging.info("Fitted T* on clean holdout: %.3f", T_star)

    configs = [
        ("blur", BLUR_SIGMAS),
        ("noise", NOISE_SIGMAS),
        ("occlusion", OCCLUSION_SIZES),
        ("rotation", ROTATION_ANGLES),
    ]
    rng = np.random.default_rng(SEED)

    summary: dict = {"T_star": T_star, "img_size": IMG_SIZE, "M": M, "perturbations": {}}
    for name, severities in configs:
        summary["perturbations"][name] = []
        for sev in severities:
            perturbed = apply_perturbation(holdout_raw, name, sev, rng)
            X = prepare_images_for_cnn(perturbed, target_size=IMG_SIZE)
            perM = per_member_softmax(members, X, device)  # (M, N, 10)
            ensemble_probs = perM.mean(axis=0)

            # Calibration: raw and temperature-scaled.
            ece_raw = expected_calibration_error(y_hold, ensemble_probs, n_bins=15)
            ensemble_probs_T = apply_temperature(ensemble_probs, T_star)
            ece_T = expected_calibration_error(y_hold, ensemble_probs_T, n_bins=15)

            # Entropy decomposition (Depeweg 2018; Smith & Gal 2018).
            #   total      = H[E[p]]                  -- entropy of ensemble mean
            #   aleatoric  = E[H[p_m]]                -- mean per-member entropy
            #   epistemic  = total - aleatoric        -- non-negative
            total_H = float(np.mean(shannon_entropy(ensemble_probs)))
            ale_H = float(np.mean(shannon_entropy(perM)))  # mean over M*N then over axis 1 above; equivalent
            epi_H = total_H - ale_H

            # Top-class confidence (mean and on wrong predictions only).
            preds = ensemble_probs.argmax(axis=1)
            top_conf = float(ensemble_probs.max(axis=1).mean())
            wrong_mask = preds != y_hold
            wrong_conf = (
                float(ensemble_probs.max(axis=1)[wrong_mask].mean())
                if wrong_mask.any() else float("nan")
            )
            acc = float((preds == y_hold).mean())

            summary["perturbations"][name].append(
                {
                    "severity": float(sev),
                    "acc": acc,
                    "ece_raw": ece_raw,
                    "ece_T_scaled": ece_T,
                    "entropy_total": total_H,
                    "entropy_aleatoric": ale_H,
                    "entropy_epistemic": epi_H,
                    "top_confidence_mean": top_conf,
                    "top_confidence_on_wrong": wrong_conf,
                }
            )
            logging.info(
                "%-9s sev=%-6s acc=%.3f ECE_raw=%.3f ECE_T=%.3f H=%.3f ale=%.3f epi=%.3f conf_wrong=%.3f",
                name, str(sev), acc, ece_raw, ece_T, total_H, ale_H, epi_H, wrong_conf,
            )

    out_json = metrics_dir / "06b_uncertainty_under_shift.json"
    out_json.write_text(json.dumps(summary, indent=2))
    logging.info("Wrote %s", out_json)

    # Plot: ECE raw vs T-scaled, and aleatoric/epistemic stack, per family.
    fig, axes = plt.subplots(2, 4, figsize=(14, 6), sharex=False)
    for col, (name, severities) in enumerate(configs):
        rows = summary["perturbations"][name]
        x = [r["severity"] for r in rows]

        ax_e = axes[0, col]
        ax_e.plot(x, [r["ece_raw"] for r in rows], "o-", label="ECE (raw)", color="#C44536")
        ax_e.plot(x, [r["ece_T_scaled"] for r in rows], "s--",
                  label=f"ECE (T*={T_star:.2f})", color="#197278")
        ax_e.set_title(f"{name}: calibration")
        ax_e.set_xlabel("severity")
        ax_e.set_ylabel("ECE")
        ax_e.grid(True, alpha=0.3)
        if col == 0:
            ax_e.legend(fontsize=8)

        ax_h = axes[1, col]
        ale = np.array([r["entropy_aleatoric"] for r in rows])
        epi = np.array([r["entropy_epistemic"] for r in rows])
        ax_h.bar(np.arange(len(x)), ale, label="aleatoric", color="#5B7DB1")
        ax_h.bar(np.arange(len(x)), epi, bottom=ale, label="epistemic", color="#E4B363")
        ax_h.set_xticks(np.arange(len(x)))
        ax_h.set_xticklabels([str(v) for v in x], fontsize=8)
        ax_h.set_title(f"{name}: entropy decomposition")
        ax_h.set_xlabel("severity")
        ax_h.set_ylabel("entropy (nats)")
        if col == 0:
            ax_h.legend(fontsize=8)
        ax_h.grid(True, axis="y", alpha=0.3)

    fig.suptitle(
        f"Post-hoc: ECE with/without T*={T_star:.2f} (top) and aleatoric/epistemic decomposition (bottom)",
        fontsize=12,
    )
    fig.tight_layout()
    out_png = figures_dir / "06b_uncertainty_under_shift.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logging.info("Wrote %s", out_png)


if __name__ == "__main__":
    main()
