"""Offline calibration & robustness analysis of the trained skin model.

Runs entirely on the local machine (Apple GPU / CPU) against the checkpoint and
validation set already in ``skin/results`` and ``skin/data`` — no Colab needed.
It answers the question this whole repo is about, but for the skin model: are the
predicted probabilities *calibrated*, and how does confidence behave when the
input is *perturbed*?

Produces, in ``skin/results/figures/``:
  - ``per_class_f1.png``            per-class precision / recall / F1
  - ``confusion_matrix.png``        row-normalised confusion matrix
  - ``reliability.png``             reliability diagram + Expected Calibration Error
  - ``calibration_temperature.png`` reliability before/after temperature scaling
  - ``uncertainty_split.png``       predictive entropy, correct vs. incorrect
  - ``confident_errors.png``        the most confident *wrong* predictions
  - ``robustness.png``              balanced accuracy & mean uncertainty under shift

and a metrics summary at ``skin/results/analysis_metrics.json`` that now also
carries a bootstrap 95% CI on balanced accuracy, the fitted temperature and
before/after ECE, and a tally of the confident-error class pairs (see
CHANGELOG.md, Tier-1 rigor pass).

Run:  cd skin && uv run python 04_analysis.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find skin_model.py from any CWD
from skin_model import IMAGENET_MEAN, IMAGENET_STD, load_trained_model, normalized_entropy

SKIN = Path(__file__).resolve().parent
VAL_DIR = SKIN / "data" / "val"
RESULTS = SKIN / "results"
FIG_DIR = RESULTS / "figures"
BATCH_SIZE = 32

# Which checkpoint to analyse. Defaults to the canonical results/ model, but set
# SKIN_MODEL_DIR to point at any checkpoint folder (e.g. an archived best model)
# while figures/metrics still write to the canonical results/ location. Example:
#   SKIN_MODEL_DIR=results/archive_efficientnet_b3 uv run python 04_analysis.py
MODEL_DIR = Path(os.environ.get("SKIN_MODEL_DIR", RESULTS))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, CLASSES, CFG = load_trained_model(MODEL_DIR, device)
IMG_SIZE = CFG["img_size"]
SHORT = [c.replace("_", " ").replace("-like lesions", "") for c in CLASSES]


# --- preprocessing -----------------------------------------------------------
# Base pipeline produces a [0,1] CHW tensor; a perturbation is applied to that
# tensor, then ImageNet normalisation. This lets robustness corruptions act on
# pixels in a natural [0,1] range, exactly as a real degraded photo would.
def make_loader(perturb=None) -> DataLoader:
    steps = [
        transforms.Resize(round(IMG_SIZE * 256 / 224)),
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
    ]
    if perturb is not None:
        steps.append(transforms.Lambda(perturb))
    steps.append(transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD))
    ds = datasets.ImageFolder(VAL_DIR, transform=transforms.Compose(steps))
    assert ds.classes == CLASSES, "val folder classes don't match the trained model"
    # num_workers=0: the val set is tiny, and worker processes can't pickle the
    # perturbation lambdas anyway.
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)


@torch.no_grad()
def run_inference(loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (probs [N, C], targets [N])."""
    model.eval()
    all_probs, all_targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        all_probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        all_targets.append(y.numpy())
    return np.concatenate(all_probs), np.concatenate(all_targets)


@torch.no_grad()
def run_inference_logits(loader) -> tuple[np.ndarray, np.ndarray]:
    """Return (logits [N, C], targets [N]). Temperature scaling needs the raw
    logits, not the softmax probabilities, so this is the clean-pass variant."""
    model.eval()
    all_logits, all_targets = [], []
    for x, y in loader:
        all_logits.append(model(x.to(device)).cpu().numpy())
        all_targets.append(y.numpy())
    return np.concatenate(all_logits), np.concatenate(all_targets)


# --- perturbations (act on a [0,1] CHW tensor) -------------------------------
def gaussian_noise(std):
    return lambda t: (t + torch.randn_like(t) * std).clamp(0, 1)


def gaussian_blur(sigma):
    k = 2 * int(np.ceil(2 * sigma)) + 1  # odd kernel covering ~±2σ
    return lambda t: TF.gaussian_blur(t, kernel_size=k, sigma=sigma)


def brightness(factor):
    return lambda t: TF.adjust_brightness(t, factor).clamp(0, 1)


# severity 0 = clean; 1..4 increasing corruption, shared x-axis across types
PERTURBATIONS = {
    "Gaussian noise": (gaussian_noise, [0.05, 0.10, 0.18, 0.30]),
    "Blur": (gaussian_blur, [0.6, 1.2, 2.2, 3.5]),
    "Darken": (brightness, [0.8, 0.6, 0.4, 0.25]),
}


# --- calibration -------------------------------------------------------------
def expected_calibration_error(probs, targets, n_bins=15):
    """Standard top-label ECE: |confidence − accuracy| averaged over confidence
    bins, weighted by bin population. Also returns per-bin (conf, acc, count)."""
    conf = probs.max(axis=1)
    pred = probs.argmax(axis=1)
    correct = (pred == targets).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    ece, bins = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        count = int(mask.sum())
        if count == 0:
            bins.append((0.5 * (lo + hi), np.nan, 0))
            continue
        bin_conf = conf[mask].mean()
        bin_acc = correct[mask].mean()
        ece += count / len(conf) * abs(bin_conf - bin_acc)
        bins.append((bin_conf, bin_acc, count))
    return ece, bins


# --- Tier-1 rigor helpers (added 2026-06-17, see CHANGELOG.md) ---------------
def bootstrap_balanced_accuracy_ci(targets, preds, n_boot=2000, alpha=0.05, seed=0):
    """Percentile bootstrap 95% CI for balanced accuracy.

    A single val split gives one point estimate; resampling the val set with
    replacement and recomputing the metric many times shows how much of that
    number is sampling noise. Two models whose CIs overlap are not meaningfully
    different — this is the guard against over-reading a 0.5% gap."""
    rng = np.random.default_rng(seed)
    n = len(targets)
    stats = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        stats[b] = balanced_accuracy_score(targets[idx], preds[idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi), float(stats.std())


def fit_temperature(logits, targets, max_iter=200):
    """Fit a single scalar temperature T that minimises NLL of softmax(logits / T)
    (Guo et al., 2017). T > 1 softens over-confident logits. Accuracy is unchanged
    (argmax is scale-invariant); only the *calibration* of the probabilities moves.
    Must be fit on a calibration split that is NOT used to report the final ECE."""
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(targets, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.05, max_iter=max_iter)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lt / T.clamp_min(1e-3), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp_min(1e-3).item())


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: {CFG['model']} @ {IMG_SIZE}px on {device}")

    # Clean pass returns logits (temperature scaling needs them); probs derived.
    logits, targets = run_inference_logits(make_loader())
    probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
    preds = probs.argmax(axis=1)
    bal_acc = balanced_accuracy_score(targets, preds)
    acc = float((preds == targets).mean())
    ci_lo, ci_hi, ci_std = bootstrap_balanced_accuracy_ci(targets, preds)
    print(f"Validation: {len(targets)} images  acc {acc:.3f}  "
          f"balanced acc {bal_acc:.3f}  (95% CI [{ci_lo:.3f}, {ci_hi:.3f}])")

    # 1) Per-class precision / recall / F1 -----------------------------------
    prec, rec, f1, support = precision_recall_fscore_support(
        targets, preds, labels=range(len(CLASSES)), zero_division=0
    )
    order = np.argsort(f1)
    y = np.arange(len(CLASSES))
    fig, ax = plt.subplots(figsize=(8, 5))
    h = 0.26
    ax.barh(y + h, prec[order], height=h, label="Precision", color="#6366f1")
    ax.barh(y, rec[order], height=h, label="Recall", color="#10b981")
    ax.barh(y - h, f1[order], height=h, label="F1", color="#f59e0b")
    ax.set_yticks(y, [SHORT[i] for i in order], fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Score")
    ax.set_title(f"Per-class performance (val) — balanced acc {bal_acc:.3f}")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "per_class_f1.png", dpi=150)
    plt.close(fig)

    # 1b) Confusion matrix (row-normalised) ----------------------------------
    cm = confusion_matrix(targets, preds, labels=range(len(CLASSES)), normalize="true")
    fig, ax = plt.subplots(figsize=(7.5, 6.5))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES)), SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES)), SHORT, fontsize=8)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black", fontsize=7)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix (val, row-normalised) — bal-acc {bal_acc:.3f}")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "confusion_matrix.png", dpi=150)
    plt.close(fig)

    # 2) Reliability diagram + ECE -------------------------------------------
    ece, bins = expected_calibration_error(probs, targets)
    print(f"Expected Calibration Error: {ece:.3f}")
    bin_centers = [b[0] for b in bins]
    bin_accs = [b[1] for b in bins]
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Perfect calibration")
    valid = [(c, a) for c, a in zip(bin_centers, bin_accs) if not np.isnan(a)]
    if valid:
        xs, ys = zip(*valid)
        ax.plot(xs, ys, marker="o", color="#6366f1", label="Model")
        ax.bar(xs, ys, width=1 / len(bins) * 0.9, alpha=0.18, color="#6366f1")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (mean predicted probability)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Reliability diagram — ECE {ece:.3f}")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "reliability.png", dpi=150)
    plt.close(fig)

    # 2b) Temperature scaling (post-hoc calibration) -------------------------
    # Fit T on a calibration half of the val set, measure ECE on the held-out
    # half *before* and *after* dividing the logits by T. Splitting matters: a T
    # fit and evaluated on the same data would report an optimistically low ECE.
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(targets))
    half = len(perm) // 2
    cal_idx, test_idx = perm[:half], perm[half:]
    temperature = fit_temperature(logits[cal_idx], targets[cal_idx])
    probs_test = probs[test_idx]
    probs_test_cal = torch.softmax(
        torch.tensor(logits[test_idx]) / temperature, dim=1).numpy()
    ece_test_raw, bins_raw = expected_calibration_error(probs_test, targets[test_idx])
    ece_test_cal, bins_cal = expected_calibration_error(probs_test_cal, targets[test_idx])
    print(f"Temperature scaling: T={temperature:.3f}  "
          f"ECE {ece_test_raw:.3f} -> {ece_test_cal:.3f} (held-out half)")

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af", label="Perfect calibration")
    for bins, color, lbl in [(bins_raw, "#ef4444", f"Before (ECE {ece_test_raw:.3f})"),
                             (bins_cal, "#10b981", f"After  (ECE {ece_test_cal:.3f})")]:
        pts = [(c, a) for c, a, _ in bins if not np.isnan(a)]
        if pts:
            xs, ys = zip(*pts)
            ax.plot(xs, ys, marker="o", color=color, label=lbl)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Confidence (mean predicted probability)")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Temperature scaling (T={temperature:.2f}) — held-out half")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "calibration_temperature.png", dpi=150)
    plt.close(fig)

    # 3) Predictive entropy, correct vs incorrect ----------------------------
    ent = np.array([normalized_entropy(p) for p in probs])
    correct_mask = preds == targets
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(ent[correct_mask], bins=20, range=(0, 1), alpha=0.7,
            color="#10b981", label=f"Correct (n={correct_mask.sum()})", density=True)
    ax.hist(ent[~correct_mask], bins=20, range=(0, 1), alpha=0.7,
            color="#ef4444", label=f"Incorrect (n={(~correct_mask).sum()})", density=True)
    ax.set_xlabel("Normalised predictive entropy (0 = sure, 1 = uniform)")
    ax.set_ylabel("Density")
    ax.set_title("The model is more uncertain when it is wrong")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "uncertainty_split.png", dpi=150)
    plt.close(fig)
    mean_ent_correct = float(ent[correct_mask].mean())
    mean_ent_incorrect = float(ent[~correct_mask].mean()) if (~correct_mask).any() else float("nan")

    # 3b) Confident-and-wrong error analysis ---------------------------------
    # The dangerous failures are the *confident* mistakes, not the unsure ones.
    # Rank wrong predictions by confidence, show the worst as a labelled grid,
    # and tally which true->predicted class pairs dominate the high-confidence
    # errors (conf >= 0.5). ImageFolder with the default order matches the
    # shuffle=False loader, so `samples[i]` is the image behind prediction i.
    from collections import Counter
    from PIL import Image

    base_ds = datasets.ImageFolder(VAL_DIR)  # paths only, same order as inference
    paths = [p for p, _ in base_ds.samples]
    conf = probs.max(axis=1)
    wrong_idx = np.where(~correct_mask)[0]
    worst = wrong_idx[np.argsort(-conf[wrong_idx])]  # most confident wrong first
    hc_pairs = Counter(
        (SHORT[targets[i]], SHORT[preds[i]]) for i in wrong_idx if conf[i] >= 0.5
    )
    confident_errors = [
        {"path": Path(paths[i]).name, "true": CLASSES[targets[i]],
         "pred": CLASSES[preds[i]], "confidence": float(conf[i]),
         "entropy": float(ent[i])}
        for i in worst[:25]
    ]
    print(f"Confident errors: {(conf[wrong_idx] >= 0.5).sum()} wrong with conf>=0.5; "
          f"top confused pairs: {hc_pairs.most_common(3)}")

    k = min(12, len(worst))
    if k:
        cols = 4
        rows = (k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), worst[:k]):
            ax.imshow(Image.open(paths[i]).convert("RGB"))
            ax.set_title(f"true: {SHORT[targets[i]]}\npred: {SHORT[preds[i]]} "
                         f"({conf[i]:.0%})", fontsize=8, color="#b91c1c")
        fig.suptitle("Most confident mistakes (the failures that matter)", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG_DIR / "confident_errors.png", dpi=150)
        plt.close(fig)

    # 4) Robustness under shift ----------------------------------------------
    # Clean baseline at severity 0, then each corruption at severities 1..4.
    torch.manual_seed(0)  # reproducible noise
    robustness = {}
    for name, (fn, params) in PERTURBATIONS.items():
        accs, ents = [bal_acc], [float(ent.mean())]
        for param in params:
            p, t = run_inference(make_loader(fn(param)))
            accs.append(balanced_accuracy_score(t, p.argmax(axis=1)))
            ents.append(float(np.mean([normalized_entropy(row) for row in p])))
        robustness[name] = {"params": params, "bal_acc": accs, "mean_entropy": ents}
        print(f"  {name}: bal-acc {accs[0]:.3f} -> {accs[-1]:.3f}  "
              f"entropy {ents[0]:.3f} -> {ents[-1]:.3f}")

    sev = list(range(5))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = {"Gaussian noise": "#ef4444", "Blur": "#6366f1", "Darken": "#f59e0b"}
    for name, data in robustness.items():
        ax1.plot(sev, data["bal_acc"], marker="o", label=name, color=colors[name])
        ax2.plot(sev, data["mean_entropy"], marker="o", label=name, color=colors[name])
    ax1.set_xlabel("Corruption severity")
    ax1.set_ylabel("Balanced accuracy")
    ax1.set_title("Accuracy degrades under shift")
    ax1.set_xticks(sev)
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)
    ax2.set_xlabel("Corruption severity")
    ax2.set_ylabel("Mean normalised entropy")
    ax2.set_title("Uncertainty should rise to match")
    ax2.set_xticks(sev)
    ax2.grid(alpha=0.3)
    ax2.legend(fontsize=9)
    fig.suptitle("Calibrated uncertainty under perturbation", fontsize=12)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "robustness.png", dpi=150)
    plt.close(fig)

    # --- metrics summary ----------------------------------------------------
    summary = {
        "model": CFG["model"],
        "img_size": IMG_SIZE,
        "n_val": int(len(targets)),
        "accuracy": acc,
        "balanced_accuracy": float(bal_acc),
        "balanced_accuracy_ci95": [ci_lo, ci_hi],
        "balanced_accuracy_boot_std": ci_std,
        "ece": float(ece),
        "temperature_scaling": {
            "temperature": temperature,
            "ece_heldout_before": float(ece_test_raw),
            "ece_heldout_after": float(ece_test_cal),
            "note": "T fit on a 50% calibration split; ECE reported on the held-out 50%",
        },
        "mean_entropy_correct": mean_ent_correct,
        "mean_entropy_incorrect": mean_ent_incorrect,
        "confident_errors": {
            "n_wrong_conf_ge_0.5": int((conf[wrong_idx] >= 0.5).sum()),
            "top_confused_pairs": [
                {"true": t, "pred": p, "count": c} for (t, p), c in hc_pairs.most_common(5)
            ],
            "worst_examples": confident_errors,
        },
        "per_class": {
            CLASSES[i]: {
                "precision": float(prec[i]),
                "recall": float(rec[i]),
                "f1": float(f1[i]),
                "support": int(support[i]),
            }
            for i in range(len(CLASSES))
        },
        "robustness": robustness,
    }
    (RESULTS / "analysis_metrics.json").write_text(json.dumps(summary, indent=2))
    print(f"\nSaved figures to {FIG_DIR} and metrics to "
          f"{RESULTS / 'analysis_metrics.json'}")


if __name__ == "__main__":
    main()
