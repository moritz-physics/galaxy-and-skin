"""Offline 'what did the model learn?' visualisations of the trained skin model.

Runs entirely on the local machine (Apple GPU / CPU) against the checkpoint and
validation set already in ``skin/results`` and ``skin/data`` — no Colab/Kaggle. All
inference-only, so it works on whatever model is currently promoted in ``skin/results``.

Produces, in ``skin/results/figures/``:
  - ``training_curves.png``   loss + val balanced-acc across head / fine-tune / cRT phases
  - ``gradcam.png``           Grad-CAM heatmaps: where the model looks, one lesion per class
  - ``feature_tsne.png``      t-SNE of the penultimate 1280-d features, coloured by class
  - ``prob_heatmap.png``      mean predicted probability per true class (soft confusion)

Run:  cd skin && uv run python 05_visualize.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

sys.path.insert(0, str(Path(__file__).resolve().parent))  # find skin_model.py from any CWD
from skin_model import IMAGENET_MEAN, IMAGENET_STD, load_trained_model

SKIN = Path(__file__).resolve().parent
VAL_DIR = SKIN / "data" / "val"
RESULTS = SKIN / "results"
FIG_DIR = RESULTS / "figures"
PERF_DIR = FIG_DIR / "performance"        # training curves, soft-confusion heatmap
INTERP_DIR = FIG_DIR / "interpretability"  # Grad-CAM, t-SNE
for _d in (PERF_DIR, INTERP_DIR):
    _d.mkdir(parents=True, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model, CLASSES, CFG = load_trained_model(RESULTS, device)
IMG_SIZE = CFG["img_size"]
SHORT = [c.replace("_", " ").replace("-like lesions", "") for c in CLASSES]
RESIZE = round(IMG_SIZE * 256 / 224)
norm = transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)

# A 7-colour palette shared by the t-SNE and heatmap so classes read consistently.
PALETTE = plt.get_cmap("tab10").colors[:len(CLASSES)]


def display_tensor(path: Path) -> torch.Tensor:
    """Resize/centre-crop a raw image to a [0,1] CHW tensor (no normalisation) — used
    both as Grad-CAM background and, after normalising, as the model input."""
    img = Image.open(path).convert("RGB")
    t = transforms.Compose([transforms.Resize(RESIZE), transforms.CenterCrop(IMG_SIZE),
                            transforms.ToTensor()])(img)
    return t


# ---------------------------------------------------------------------------
# 1. Training curves — the learning process over epochs, by phase
# ---------------------------------------------------------------------------
def plot_training_curves() -> None:
    log = json.loads((RESULTS / "training_log.json").read_text())
    hist = log["history"]
    if not hist:
        print("training_curves: no history in training_log.json — skipped")
        return
    xs = list(range(1, len(hist) + 1))
    loss = [h["loss"] for h in hist]
    bal = [h["val_bal_acc"] for h in hist]
    phases = [h["phase"] for h in hist]
    pretty = {"head": "head warm-up", "ft": "fine-tune", "crt": "cRT"}
    pcol = {"head": "#a5b4fc", "ft": "#6366f1", "crt": "#10b981"}

    fig, ax1 = plt.subplots(figsize=(10, 5))
    # shade phase spans
    start = 0
    for i in range(len(phases)):
        if i + 1 == len(phases) or phases[i + 1] != phases[i]:
            ax1.axvspan(xs[start] - 0.5, xs[i] + 0.5, color=pcol.get(phases[i], "#ddd"),
                        alpha=0.12)
            ax1.text((xs[start] + xs[i]) / 2, ax1.get_ylim()[1] if False else 0.02,
                     pretty.get(phases[i], phases[i]), ha="center", va="bottom",
                     fontsize=9, color="#374151", transform=ax1.get_xaxis_transform())
            start = i + 1

    ax1.plot(xs, loss, "-o", color="#ef4444", label="train loss", markersize=4)
    ax1.set_xlabel("epoch (running, across all phases)")
    ax1.set_ylabel("train loss", color="#ef4444")
    ax1.tick_params(axis="y", labelcolor="#ef4444")

    ax2 = ax1.twinx()
    ax2.plot(xs, bal, "-s", color="#4f46e5", label="val balanced acc", markersize=4)
    ax2.set_ylabel("val balanced accuracy", color="#4f46e5")
    ax2.tick_params(axis="y", labelcolor="#4f46e5")
    best_i = int(np.argmax(bal))
    ax2.scatter([xs[best_i]], [bal[best_i]], s=140, facecolors="none",
                edgecolors="#16a34a", linewidths=2, zorder=5)
    ax2.annotate(f"best {bal[best_i]:.3f}", (xs[best_i], bal[best_i]),
                 textcoords="offset points", xytext=(6, -14), color="#16a34a", fontsize=9)
    ax1.set_title(f"Learning curve — {CFG['model']} @ {IMG_SIZE}px "
                  f"(best val bal-acc {max(bal):.3f})")
    fig.tight_layout()
    fig.savefig(PERF_DIR / "training_curves.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved training_curves.png")


# ---------------------------------------------------------------------------
# 2. Grad-CAM — where the model looks, one example per class
# ---------------------------------------------------------------------------
def grad_cam(model_cpu, x, class_idx, target_layer):
    """Standard Grad-CAM on a single image (run on CPU for backward-hook stability)."""
    acts, grads = {}, {}
    h1 = target_layer.register_forward_hook(lambda m, i, o: acts.__setitem__("v", o))
    h2 = target_layer.register_full_backward_hook(lambda m, gi, go: grads.__setitem__("v", go[0]))
    model_cpu.zero_grad()
    logits = model_cpu(x)
    if class_idx is None:
        class_idx = int(logits.argmax(1))
    prob = torch.softmax(logits, 1)[0, class_idx].item()
    logits[0, class_idx].backward()
    A, G = acts["v"][0], grads["v"][0]          # [K,h,w]
    w = G.mean(dim=(1, 2))                        # [K]
    cam = torch.relu((w[:, None, None] * A).sum(0))
    cam = cam / (cam.max() + 1e-8)
    cam = F.interpolate(cam[None, None], size=(IMG_SIZE, IMG_SIZE),
                        mode="bilinear", align_corners=False)[0, 0]
    h1.remove(); h2.remove()
    return cam.detach().numpy(), class_idx, prob


def plot_gradcam() -> None:
    ds = datasets.ImageFolder(VAL_DIR)  # .samples = [(path, label), ...], sorted by class
    # one representative image per class
    per_class = {}
    for path, label in ds.samples:
        per_class.setdefault(label, path)
    model_cpu = model.to("cpu").eval()
    # EfficientNet/ConvNeXt expose .features; target the last conv block's output.
    target_layer = model_cpu.features[-1]

    fig, axes = plt.subplots(2, 4, figsize=(14, 7.5))
    axes = axes.ravel()
    for ci in range(len(CLASSES)):
        path = Path(per_class[ci])
        disp = display_tensor(path)                       # [0,1] CHW
        x = norm(disp).unsqueeze(0)                        # normalised, CPU
        cam, pred, prob = grad_cam(model_cpu, x, None, target_layer)
        img = disp.permute(1, 2, 0).numpy()
        heat = cm.jet(cam)[..., :3]
        overlay = 0.55 * img + 0.45 * heat
        ax = axes[ci]
        ax.imshow(np.clip(overlay, 0, 1))
        ok = pred == ci
        ax.set_title(f"{SHORT[ci]}\npred: {SHORT[pred]} ({prob:.0%})",
                     fontsize=9, color="#16a34a" if ok else "#dc2626")
        ax.axis("off")
    for j in range(len(CLASSES), len(axes)):
        axes[j].axis("off")
    fig.suptitle("Grad-CAM — image regions driving the prediction (red = most influential)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(INTERP_DIR / "gradcam.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    model.to(device)  # restore
    print("saved gradcam.png")


# ---------------------------------------------------------------------------
# 3 & 4. Feature embedding (t-SNE) + soft-probability heatmap (shared inference)
# ---------------------------------------------------------------------------
def _last_linear(m):
    """The final nn.Linear (the classifier head) — generic across backbones."""
    last = None
    for mod in m.modules():
        if isinstance(mod, torch.nn.Linear):
            last = mod
    return last


@torch.no_grad()
def collect_features_and_probs():
    ds = datasets.ImageFolder(VAL_DIR, transform=transforms.Compose(
        [transforms.Resize(RESIZE), transforms.CenterCrop(IMG_SIZE),
         transforms.ToTensor(), norm]))
    assert ds.classes == CLASSES, "val folder classes don't match the trained model"
    loader = DataLoader(ds, batch_size=32, shuffle=False, num_workers=0)
    model.eval()
    # Architecture-agnostic penultimate embedding: capture the input to the final
    # Linear via a forward hook (works for EfficientNet, ConvNeXt, ViT, ... alike).
    grabbed = {}
    h = _last_linear(model).register_forward_hook(
        lambda mod, inp, out: grabbed.__setitem__("v", inp[0].detach()))
    feats, probs, targs = [], [], []
    for x, y in loader:
        logits = model(x.to(device))
        feats.append(grabbed["v"].cpu().numpy())
        probs.append(torch.softmax(logits, 1).cpu().numpy())
        targs.append(y.numpy())
    h.remove()
    return np.concatenate(feats), np.concatenate(probs), np.concatenate(targs)


def plot_tsne(feats, targs) -> None:
    n = len(feats)
    emb = TSNE(n_components=2, init="pca", perplexity=min(30, max(5, n // 12)),
               learning_rate="auto", random_state=42).fit_transform(feats)
    fig, ax = plt.subplots(figsize=(8.5, 7))
    for ci in range(len(CLASSES)):
        m = targs == ci
        ax.scatter(emb[m, 0], emb[m, 1], s=18, color=PALETTE[ci], label=SHORT[ci],
                   alpha=0.75, edgecolors="white", linewidths=0.3)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title("Learned feature space (t-SNE of the 1280-d embedding)\n"
                 "tight, separated clusters = the model has learned to tell classes apart")
    ax.legend(loc="best", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(INTERP_DIR / "feature_tsne.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved feature_tsne.png")


def plot_prob_heatmap(probs, targs) -> None:
    # mean predicted probability vector per true class -> [C, C]
    M = np.vstack([probs[targs == ci].mean(0) for ci in range(len(CLASSES))])
    fig, ax = plt.subplots(figsize=(8, 6.5))
    im = ax.imshow(M, cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASSES))); ax.set_xticklabels(SHORT, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(CLASSES))); ax.set_yticklabels(SHORT, fontsize=8)
    ax.set_xlabel("predicted class"); ax.set_ylabel("true class")
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 0.6 else "black")
    fig.colorbar(im, fraction=0.046, pad=0.04, label="mean predicted probability")
    ax.set_title("Mean predicted probability per true class\n"
                 "(diagonal = confidence on correct class; off-diagonal = where it leaks)")
    fig.tight_layout()
    fig.savefig(PERF_DIR / "prob_heatmap.png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved prob_heatmap.png")


def main() -> None:
    print(f"Model: {CFG['model']} @ {IMG_SIZE}px on {device}")
    plot_training_curves()
    plot_gradcam()
    feats, probs, targs = collect_features_and_probs()
    print(f"Collected {len(feats)} val embeddings ({feats.shape[1]}-d)")
    plot_tsne(feats, targs)
    plot_prob_heatmap(probs, targs)
    print(f"\nDone — 4 figures in {FIG_DIR}")


if __name__ == "__main__":
    main()
