"""Offline calibration & robustness analysis of the trained skin model.

Runs entirely on the local machine (Apple GPU / CPU) against the checkpoint and
validation set already in ``skin/results`` and ``skin/data`` — no Colab needed.
It answers the question this whole repo is about, but for the skin model: are the
predicted probabilities *calibrated*, and how does confidence behave when the
input is *perturbed*?

Produces, in ``skin/results/figures/``:
  - ``per_class_f1.png``       per-class precision / recall / F1
  - ``confusion_matrix.png``   row-normalised confusion matrix
  - ``reliability.png``        reliability diagram + Expected Calibration Error
  - ``uncertainty_split.png``  predictive entropy, correct vs. incorrect
  - ``robustness.png``         balanced accuracy & mean uncertainty under shift

and a metrics summary at ``skin/results/analysis_metrics.json``.

Run:  cd skin && uv run python 04_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
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

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, CLASSES, CFG = load_trained_model(RESULTS, device)
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


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Model: {CFG['model']} @ {IMG_SIZE}px on {device}")

    probs, targets = run_inference(make_loader())
    preds = probs.argmax(axis=1)
    bal_acc = balanced_accuracy_score(targets, preds)
    acc = float((preds == targets).mean())
    print(f"Validation: {len(targets)} images  acc {acc:.3f}  balanced acc {bal_acc:.3f}")

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
        "ece": float(ece),
        "mean_entropy_correct": mean_ent_correct,
        "mean_entropy_incorrect": mean_ent_incorrect,
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
    print(f"\nSaved 5 figures to {FIG_DIR} and metrics to "
          f"{RESULTS / 'analysis_metrics.json'}")


if __name__ == "__main__":
    main()
