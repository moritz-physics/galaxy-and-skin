"""Log a *rich* evaluation of the trained skin model to Weights & Biases.

This is the answer to "I could have printed that myself." Scalar loss/accuracy
curves, yes -- you can. But these objects you cannot meaningfully print, and they
are where W&B earns its place:

  1. An interactive PREDICTION TABLE: every validation image as a row with its
     thumbnail, true label, predicted label, confidence and entropy. In the
     browser you can *sort and filter* it -- e.g. "show me the cases the model
     got wrong while being >90% confident". For a medical model that single view
     is worth more than any curve.
  2. An interactive CONFUSION MATRIX you can hover/scroll, not a static PNG.
  3. Per-class PR and ROC curves as live panels.
  4. The existing analysis figures (reliability, robustness, ...) collected into
     one shareable media gallery.

It reuses the exact model + local val data loading from 04_analysis.py, so it
runs fully offline on your Mac (MPS/CPU) -- no Kaggle, no retraining.

Run:
    cd skin
    uv run wandb login            # once, if you haven't
    uv run python 07_wandb_eval.py

Then open the run URL it prints. See the bottom of this file's docstring-less
main() for exactly which W&B tab each object lands in.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch
import wandb
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, f1_score
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))
from skin_model import IMAGENET_MEAN, IMAGENET_STD, load_trained_model, normalized_entropy

SKIN = Path(__file__).resolve().parent
VAL_DIR = SKIN / "data" / "val"
RESULTS = SKIN / "results"
FIG_DIR = RESULTS / "figures"
PROJECT = "skin-lesion"
THUMB = 160  # thumbnail size (px) for table images

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, CLASSES, CFG = load_trained_model(RESULTS, device)
IMG_SIZE = CFG["img_size"]
SHORT = [c.replace("_", " ").replace("-like lesions", "") for c in CLASSES]


def build_loader() -> tuple[DataLoader, list[str]]:
    """Val loader (normalised tensors) plus the on-disk path of every image, in
    the same order, so we can attach the original picture to each table row."""
    tf = transforms.Compose(
        [
            transforms.Resize(round(IMG_SIZE * 256 / 224)),
            transforms.CenterCrop(IMG_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    ds = datasets.ImageFolder(VAL_DIR, transform=tf)
    assert ds.classes == CLASSES, "val folder classes don't match the trained model"
    paths = [p for p, _ in ds.samples]
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    return loader, paths


@torch.no_grad()
def infer(loader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, targets = [], []
    for x, y in loader:
        logits = model(x.to(device))
        probs.append(torch.softmax(logits, dim=1).cpu().numpy())
        targets.append(y.numpy())
    return np.concatenate(probs), np.concatenate(targets)


def main() -> None:
    loader, paths = build_loader()
    probs, targets = infer(loader)
    preds = probs.argmax(axis=1)
    conf = probs.max(axis=1)
    ent = np.array([normalized_entropy(p) for p in probs])
    bal_acc = balanced_accuracy_score(targets, preds)
    acc = float((preds == targets).mean())
    print(f"val: {len(targets)} imgs  acc {acc:.3f}  bal-acc {bal_acc:.3f}")

    run = wandb.init(
        project=PROJECT,
        name=f"eval_{CFG['model']}",
        job_type="evaluation",
        config={"model": CFG["model"], "img_size": IMG_SIZE, "n_val": len(targets)},
    )

    # (1) Interactive prediction table -- the centrepiece. One row per image.
    # `correct` and `confidence` are sortable columns, so in the UI you can rank
    # by "wrong & most confident" to find the model's dangerous mistakes.
    table = wandb.Table(
        columns=["image", "true", "pred", "correct", "confidence", "entropy", "p_true"]
    )
    for i, path in enumerate(paths):
        img = Image.open(path).convert("RGB")
        img.thumbnail((THUMB, THUMB))
        table.add_data(
            wandb.Image(img),
            SHORT[targets[i]],
            SHORT[preds[i]],
            bool(preds[i] == targets[i]),
            float(conf[i]),
            float(ent[i]),
            float(probs[i, targets[i]]),
        )
    run.log({"predictions": table})

    # (2) Interactive confusion matrix (hover to read counts; not a flat PNG).
    run.log(
        {
            "confusion_matrix": wandb.plot.confusion_matrix(
                y_true=targets.tolist(), preds=preds.tolist(), class_names=SHORT
            )
        }
    )

    # (3) Per-class PR and ROC curves as live panels.
    run.log(
        {
            "pr_curve": wandb.plot.pr_curve(targets.tolist(), probs, labels=SHORT),
            "roc_curve": wandb.plot.roc_curve(targets.tolist(), probs, labels=SHORT),
        }
    )

    # (4) The analysis PNGs gathered into one media gallery, side by side.
    figures = [
        "calibration/reliability.png", "robustness/robustness.png",
        "calibration/uncertainty_split.png", "performance/per_class_f1.png",
        "interpretability/gradcam.png", "performance/prob_heatmap.png",
    ]
    gallery = [
        wandb.Image(str(FIG_DIR / f), caption=Path(f).stem)
        for f in figures
        if (FIG_DIR / f).exists()
    ]
    if gallery:
        run.log({"analysis_figures": gallery})

    # Headline scalars -> the run's summary / the runs table.
    per_class_f1 = f1_score(targets, preds, labels=range(len(CLASSES)), average=None, zero_division=0)
    run.summary.update(
        {
            "accuracy": acc,
            "balanced_accuracy": float(bal_acc),
            "macro_f1": float(per_class_f1.mean()),
            **{f"f1/{SHORT[i]}": float(per_class_f1[i]) for i in range(len(CLASSES))},
        }
    )
    run.finish()
    print("done -- open the run URL above; see the Tables and Charts sections.")


if __name__ == "__main__":
    main()
