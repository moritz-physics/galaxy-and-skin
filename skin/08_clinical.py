"""Clinically-honest evaluation of the trained skin model (offline).

Balanced accuracy treats every class as equally important. A *clinic* does not:
missing a melanoma (a false negative on a deadly cancer) is far worse than
mistaking one benign lesion for another. This script re-scores the same trained
model and val set through that lens and writes a standalone report.

It produces, in ``skin/results/figures/clinical/``:
  - ``clinical_sensitivity.png``   per-class recall (sensitivity), danger classes flagged
  - ``clinical_melanoma_roc.png``  melanoma-vs-rest ROC with a high-sensitivity operating point
  - ``clinical_referral.png``      "refer vs reassure" screening confusion + threshold sweep
  - ``clinical_missed.png``        true melanomas the model called benign (the dangerous misses)

plus ``skin/results/clinical_metrics.json``. The narrative overview is written to
``skin/CLINICAL_REPORT.md`` by the companion step.

Run:  cd skin && uv run python 08_clinical.py
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
from PIL import Image
from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from torchvision import datasets

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skin_model import load_trained_model, val_transform

SKIN = Path(__file__).resolve().parent
VAL_DIR = SKIN / "data" / "val"
RESULTS = SKIN / "results"
FIG_DIR = RESULTS / "figures"
CLIN_DIR = FIG_DIR / "clinical"

# Clinical risk tiers (see README risk table). "Concerning" = anything a
# screening tool should flag for a dermatologist; benign = safe to reassure.
MALIGNANT = {"melanoma", "basal_cell_carcinoma"}
PREMALIGNANT = {"actinic_keratoses"}
CONCERNING = MALIGNANT | PREMALIGNANT  # should be referred
TARGET_SENSITIVITY = 0.90  # the operating point a screen would be tuned to

# Set SKIN_MODEL_DIR to analyse a checkpoint other than the canonical results/
# one (e.g. results/archive_efficientnet_b3 for the best model).
MODEL_DIR = Path(os.environ.get("SKIN_MODEL_DIR", RESULTS))

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, CLASSES, CFG = load_trained_model(MODEL_DIR, device)
SHORT = [c.replace("_", " ").replace("-like lesions", "") for c in CLASSES]


@torch.no_grad()
def infer(loader):
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        probs.append(torch.softmax(model(x.to(device)), dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def sens_spec_at(scores, positive, thresh):
    """Sensitivity / specificity of flagging `scores >= thresh` as positive."""
    flag = scores >= thresh
    tp = int((flag & positive).sum()); fn = int((~flag & positive).sum())
    tn = int((~flag & ~positive).sum()); fp = int((flag & ~positive).sum())
    sens = tp / (tp + fn) if (tp + fn) else float("nan")
    spec = tn / (tn + fp) if (tn + fp) else float("nan")
    ppv = tp / (tp + fp) if (tp + fp) else float("nan")
    return sens, spec, ppv, dict(tp=tp, fn=fn, tn=tn, fp=fp)


def main() -> None:
    CLIN_DIR.mkdir(parents=True, exist_ok=True)
    base_ds = datasets.ImageFolder(VAL_DIR)
    paths = [p for p, _ in base_ds.samples]
    ds = datasets.ImageFolder(VAL_DIR, transform=val_transform(CFG["img_size"]))
    assert ds.classes == CLASSES, "val classes don't match the trained model"
    probs, targets = infer(DataLoader(ds, batch_size=32, num_workers=0))
    preds = probs.argmax(axis=1)
    cidx = {c: i for i, c in enumerate(CLASSES)}
    print(f"Model: {CFG['model']} @ {CFG['img_size']}px — {len(targets)} val images")

    # 1) per-class sensitivity (recall) -------------------------------------
    cm = confusion_matrix(targets, preds, labels=range(len(CLASSES)))
    recall = np.diag(cm) / cm.sum(axis=1).clip(min=1)
    danger = [c in CONCERNING for c in CLASSES]
    order = np.argsort(recall)
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["#ef4444" if danger[i] else "#9ca3af" for i in order]
    ax.barh([SHORT[i] for i in order], recall[order], color=colors)
    for y_, i in enumerate(order):
        ax.text(recall[i] + 0.01, y_, f"{recall[i]:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("Sensitivity (recall) — fraction of this class correctly caught")
    ax.set_title("Per-class sensitivity (red = malignant / pre-cancerous)")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout(); fig.savefig(CLIN_DIR / "clinical_sensitivity.png", dpi=150); plt.close(fig)

    # 2) melanoma-vs-rest ROC + operating point -----------------------------
    mel = cidx["melanoma"]
    y_mel = (targets == mel).astype(int)
    s_mel = probs[:, mel]
    auc_mel = roc_auc_score(y_mel, s_mel)
    fpr, tpr, thr = roc_curve(y_mel, s_mel)
    # smallest threshold reaching the target sensitivity
    ok = np.where(tpr >= TARGET_SENSITIVITY)[0]
    j = ok[0] if len(ok) else len(tpr) - 1
    op_thr = float(thr[j])
    mel_sens, mel_spec, mel_ppv, mel_cm = sens_spec_at(s_mel, y_mel.astype(bool), op_thr)
    # argmax-based melanoma sensitivity (the default decision rule)
    mel_sens_argmax = float(((preds == mel) & (targets == mel)).sum() / max(1, (targets == mel).sum()))
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, color="#6366f1", label=f"melanoma ROC (AUC {auc_mel:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#9ca3af")
    ax.scatter([fpr[j]], [tpr[j]], color="#ef4444", zorder=5,
               label=f"op @ sens={tpr[j]:.2f}, spec={1 - fpr[j]:.2f}")
    ax.set_xlabel("False positive rate (1 − specificity)")
    ax.set_ylabel("True positive rate (sensitivity)")
    ax.set_title(f"Melanoma detection (one-vs-rest), threshold={op_thr:.2f}")
    ax.legend(loc="lower right", fontsize=9); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(CLIN_DIR / "clinical_melanoma_roc.png", dpi=150); plt.close(fig)

    # 3) referral screen: concerning (malignant/pre-cancerous) vs benign -----
    conc_idx = [cidx[c] for c in CONCERNING]
    y_conc = np.isin(targets, conc_idx)
    s_conc = probs[:, conc_idx].sum(axis=1)            # P(lesion is concerning)
    auc_conc = roc_auc_score(y_conc.astype(int), s_conc)
    # argmax rule
    refer_argmax = np.isin(preds, conc_idx)
    a_sens, a_spec, a_ppv, a_cm = sens_spec_at(refer_argmax.astype(float), y_conc, 0.5)
    # threshold tuned to TARGET_SENSITIVITY
    fpr_c, tpr_c, thr_c = roc_curve(y_conc.astype(int), s_conc)
    okc = np.where(tpr_c >= TARGET_SENSITIVITY)[0]
    jc = okc[0] if len(okc) else len(tpr_c) - 1
    t_sens, t_spec, t_ppv, t_cm = sens_spec_at(s_conc, y_conc, float(thr_c[jc]))
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.6))
    # left: threshold sweep
    axL.plot(thr_c, tpr_c, color="#10b981", label="sensitivity")
    axL.plot(thr_c, 1 - fpr_c, color="#6366f1", label="specificity")
    axL.axhline(TARGET_SENSITIVITY, ls="--", color="#ef4444", lw=0.8)
    axL.axvline(float(thr_c[jc]), ls="--", color="#9ca3af", lw=0.8)
    axL.set_xlabel("Referral threshold on P(concerning)")
    axL.set_ylabel("Rate"); axL.set_title(f"Refer-vs-reassure trade-off (AUC {auc_conc:.3f})")
    axL.legend(fontsize=9); axL.grid(alpha=0.3); axL.set_xlim(0, 1)
    # right: argmax screening confusion (2x2)
    scm = np.array([[a_cm["tn"], a_cm["fp"]], [a_cm["fn"], a_cm["tp"]]])
    im = axR.imshow(scm, cmap="Blues")
    axR.set_xticks([0, 1], ["reassure", "refer"]); axR.set_yticks([0, 1], ["benign", "concerning"])
    for r in range(2):
        for cc in range(2):
            axR.text(cc, r, scm[r, cc], ha="center", va="center",
                     color="white" if scm[r, cc] > scm.max() / 2 else "black")
    axR.set_xlabel("Model decision (argmax)"); axR.set_ylabel("Truth")
    axR.set_title(f"Screen @ argmax — sens {a_sens:.2f}, spec {a_spec:.2f}")
    fig.tight_layout(); fig.savefig(CLIN_DIR / "clinical_referral.png", dpi=150); plt.close(fig)

    # 4) the dangerous misses: true melanomas predicted benign ---------------
    benign_idx = [i for i, c in enumerate(CLASSES) if c not in CONCERNING]
    missed = np.where((targets == mel) & np.isin(preds, benign_idx))[0]
    missed = missed[np.argsort(-probs[missed].max(axis=1))]  # most confident misses first
    k = min(8, len(missed))
    if k:
        cols = 4; rows = (k + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
        for ax in np.atleast_1d(axes).ravel():
            ax.axis("off")
        for ax, i in zip(np.atleast_1d(axes).ravel(), missed[:k]):
            ax.imshow(Image.open(paths[i]).convert("RGB"))
            ax.set_title(f"melanoma called\n{SHORT[preds[i]]} ({probs[i].max():.0%})",
                         fontsize=8, color="#b91c1c")
        fig.suptitle("Missed melanomas (true melanoma predicted benign)", fontsize=12)
        fig.tight_layout(); fig.savefig(CLIN_DIR / "clinical_missed.png", dpi=150); plt.close(fig)

    metrics = {
        "model": CFG["model"], "img_size": CFG["img_size"], "n_val": int(len(targets)),
        "per_class_sensitivity": {CLASSES[i]: float(recall[i]) for i in range(len(CLASSES))},
        "melanoma": {
            "auc": float(auc_mel),
            "sensitivity_argmax": mel_sens_argmax,
            "operating_point": {"target_sensitivity": TARGET_SENSITIVITY, "threshold": op_thr,
                                "sensitivity": mel_sens, "specificity": mel_spec, "ppv": mel_ppv,
                                "counts": mel_cm},
        },
        "referral_screen": {
            "auc": float(auc_conc),
            "argmax": {"sensitivity": a_sens, "specificity": a_spec, "ppv": a_ppv, "counts": a_cm},
            "tuned": {"target_sensitivity": TARGET_SENSITIVITY, "threshold": float(thr_c[jc]),
                      "sensitivity": t_sens, "specificity": t_spec, "ppv": t_ppv, "counts": t_cm},
        },
        "missed_melanomas": {
            "n_true_melanoma": int((targets == mel).sum()),
            "n_called_benign": int(len(missed)),
            "examples": [Path(paths[i]).name for i in missed[:8]],
        },
    }
    (RESULTS / "clinical_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"Melanoma AUC {auc_mel:.3f}; argmax sensitivity {mel_sens_argmax:.2f}")
    print(f"Referral screen @ argmax: sens {a_sens:.2f} spec {a_spec:.2f}; "
          f"@ tuned (sens {TARGET_SENSITIVITY:.0%}): spec {t_spec:.2f}")
    print(f"Missed melanomas (called benign): {len(missed)}/{int((targets == mel).sum())}")
    print(f"Saved 4 clinical figures + clinical_metrics.json")


if __name__ == "__main__":
    main()
