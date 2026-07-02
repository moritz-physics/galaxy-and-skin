"""Feature extraction — step 1 of mechanistic interpretability on the skin model.

Not "feature extraction" in the classic transfer-learning sense (chop the head off,
use the embedding). This is the *mechanistic* sense: go channel by channel inside a
chosen conv layer and answer "what concept does this unit detect?", building a labelled
inventory of the network's learned feature dictionary. That inventory is the prerequisite
for everything downstream (feature visualization, circuits, sparse autoencoders).

We hook the final spatial feature map (output of ``model.features`` — the tensor the
classifier reads after global-average-pooling), because those are precisely the features
the skin head relies on. Pass ``--stage`` to probe an earlier block for more localised,
texture-like features instead.

For each channel we characterise it two ways and make them agree:
  1. Class selectivity — does this channel fire mostly on one lesion class?
  2. Max-activating dataset examples — which real image patches drive it hardest?

Outputs (to ``mech-interp/figures/``):
  - ``selectivity_heatmap.png``      top channels × classes, class-tuning of each unit
  - ``max_activating_patches.png``   top selective channels × their strongest patches
  - ``per_class_feature.png``        the most class-selective channel per lesion class
and ``mech-interp/cache/feature_stats.npz`` (activation stats, so re-plotting is cheap)
plus ``mech-interp/figures/feature_summary.json``.

Intended to run where the model lives (Colab GPU). It loads the trained checkpoint and
runs forward passes over the val set, so it is NOT meant for the local Mac.

Run (from skin/, on a GPU box):  uv run python mech-interp/01_feature_extraction.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import datasets, transforms

SKIN = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(SKIN))  # find skin_model.py from any CWD
from skin_model import IMAGENET_MEAN, IMAGENET_STD, load_trained_model, val_transform

VAL_DIR = SKIN / "data" / "val"
RESULTS = SKIN / "results"
HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures"
CACHE_DIR = HERE / "cache"
BATCH_SIZE = 32


def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def get_feature_module(model: torch.nn.Module, stage: int) -> torch.nn.Module:
    """Return the conv block to hook. ``stage=-1`` is the full feature trunk output
    (what the classifier reads); a non-negative index selects one block inside it."""
    features = getattr(model, "features", None)
    if features is None:
        raise ValueError(
            f"{type(model).__name__} has no `.features` trunk; this script targets "
            "torchvision conv nets (EfficientNet/ConvNeXt)."
        )
    return features if stage < 0 else features[stage]


@torch.no_grad()
def collect_activations(model, module, loader, device):
    """Forward the whole val set once, capturing the hooked feature map.

    Returns:
      maxact  [N, C]  per-channel spatial-max activation for each image
      argloc  [N, C]  flattened (h*w) index of that max, for cropping the patch
      labels  [N]     class index per image
      grid_hw (h, w)  spatial size of the feature map
    """
    captured = {}

    def hook(_m, _inp, out):
        captured["a"] = out.detach()

    handle = module.register_forward_hook(hook)
    maxact, argloc, labels, grid_hw = [], [], [], None
    try:
        for x, y in loader:
            model(x.to(device))
            a = captured["a"]  # [B, C, h, w]
            b, c, h, w = a.shape
            grid_hw = (h, w)
            flat = a.reshape(b, c, h * w)
            vmax, vidx = flat.max(dim=2)  # [B, C] each
            maxact.append(vmax.cpu().numpy())
            argloc.append(vidx.cpu().numpy())
            labels.append(y.numpy())
    finally:
        handle.remove()
    return (
        np.concatenate(maxact),
        np.concatenate(argloc),
        np.concatenate(labels),
        grid_hw,
    )


def class_selectivity(maxact, labels, n_classes):
    """Per-channel class tuning.

    classmean [C, n_classes] — mean spatial-max activation of each channel on each class.
    selectivity [C]          — (best_class_mean − mean_of_rest) / (best + rest), in ~[0,1].
    best_class  [C]          — the class each channel is most tuned to.
    """
    c = maxact.shape[1]
    classmean = np.zeros((c, n_classes), dtype=np.float64)
    for k in range(n_classes):
        m = labels == k
        if m.any():
            classmean[:, k] = maxact[m].mean(axis=0)
    best_class = classmean.argmax(axis=1)
    best = classmean.max(axis=1)
    rest = (classmean.sum(axis=1) - best) / max(n_classes - 1, 1)
    selectivity = (best - rest) / (best + rest + 1e-9)
    return classmean, selectivity, best_class


def display_image(path: str, img_size: int) -> np.ndarray:
    """Reload one image with the val geometry (resize + centre-crop), no normalisation,
    as an HWC [0,1] array — used to crop display patches at the model's input scale."""
    t = transforms.Compose(
        [
            transforms.Resize(round(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
        ]
    )
    from PIL import Image

    return t(Image.open(path).convert("RGB")).permute(1, 2, 0).numpy()


def crop_patch(img_hwc, loc_flat, grid_hw, img_size):
    """Crop a context window around the feature map's peak location."""
    gh, gw = grid_hw
    cy, cx = divmod(int(loc_flat), gw)
    cell = img_size / gh
    py, px = int((cy + 0.5) * cell), int((cx + 0.5) * cell)
    half = int(cell * 1.5)  # ~3 grid cells of context
    y0, y1 = max(py - half, 0), min(py + half, img_size)
    x0, x1 = max(px - half, 0), min(px + half, img_size)
    return img_hwc[y0:y1, x0:x1]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", type=int, default=-1,
                    help="feature block to hook (-1 = full trunk output, the head's input)")
    ap.add_argument("--top-channels", type=int, default=12,
                    help="how many most-selective channels to show patches for")
    ap.add_argument("--heatmap-channels", type=int, default=40,
                    help="how many channels in the selectivity heatmap")
    ap.add_argument("--patches", type=int, default=6, help="patches per channel")
    ap.add_argument("--use-cache", action="store_true",
                    help="reuse cached activation stats instead of re-running inference")
    args = ap.parse_args()

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    device = pick_device()

    model, classes, cfg = load_trained_model(RESULTS, device)
    img_size = cfg["img_size"]
    n_classes = len(classes)
    short = [c.replace("_", " ").replace("-like lesions", "") for c in classes]
    print(f"Model: {cfg['model']} @ {img_size}px on {device} — hooking stage {args.stage}")

    ds = datasets.ImageFolder(VAL_DIR, transform=val_transform(img_size))
    assert ds.classes == classes, "val folder classes don't match the trained model"
    samples = ds.samples  # [(path, class_idx)]

    cache = CACHE_DIR / f"feature_stats_stage{args.stage}.npz"
    if args.use_cache and cache.exists():
        d = np.load(cache)
        maxact, argloc, labels, grid_hw = d["maxact"], d["argloc"], d["labels"], tuple(d["grid_hw"])
        print(f"Loaded cached stats {cache.name}")
    else:
        loader = torch.utils.data.DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        module = get_feature_module(model, args.stage)
        maxact, argloc, labels, grid_hw = collect_activations(model, module, loader, device)
        np.savez(cache, maxact=maxact, argloc=argloc, labels=labels, grid_hw=np.array(grid_hw))
        print(f"Collected activations: {maxact.shape[0]} images × {maxact.shape[1]} channels "
              f"at {grid_hw[0]}×{grid_hw[1]} grid; cached to {cache.name}")

    classmean, selectivity, best_class = class_selectivity(maxact, labels, n_classes)
    order = np.argsort(selectivity)[::-1]  # most selective first

    # 1) Selectivity heatmap — top channels × classes (each row normalised to its max) ----
    sel_ch = order[: args.heatmap_channels]
    hm = classmean[sel_ch]
    hm = hm / (hm.max(axis=1, keepdims=True) + 1e-9)
    fig, ax = plt.subplots(figsize=(8, 0.28 * len(sel_ch) + 1.5))
    im = ax.imshow(hm, aspect="auto", cmap="magma", vmin=0, vmax=1)
    ax.set_xticks(range(n_classes), short, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(sel_ch)),
                  [f"ch{c} → {short[best_class[c]]}" for c in sel_ch], fontsize=7)
    ax.set_xlabel("Lesion class")
    ax.set_title(f"Channel class-tuning (top {len(sel_ch)} by selectivity, stage {args.stage})")
    fig.colorbar(im, fraction=0.025, pad=0.02, label="class-mean activation (row-normalised)")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "selectivity_heatmap.png", dpi=150)
    plt.close(fig)

    # 2) Max-activating patches — top selective channels × strongest real examples --------
    top = order[: args.top_channels]
    k = args.patches
    fig, axes = plt.subplots(len(top), k, figsize=(1.5 * k, 1.6 * len(top)))
    axes = np.atleast_2d(axes)
    for r, ch in enumerate(top):
        top_imgs = np.argsort(maxact[:, ch])[::-1][:k]
        for col, idx in enumerate(top_imgs):
            ax = axes[r, col]
            img = display_image(samples[idx][0], img_size)
            patch = crop_patch(img, argloc[idx, ch], grid_hw, img_size)
            ax.imshow(np.clip(patch, 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"ch{ch}\n{short[best_class[ch]]}\nsel {selectivity[ch]:.2f}",
                              fontsize=7, rotation=0, ha="right", va="center", labelpad=28)
    fig.suptitle(f"Max-activating patches per channel (stage {args.stage})", fontsize=12)
    fig.tight_layout(rect=(0.06, 0, 1, 0.98))
    fig.savefig(FIG_DIR / "max_activating_patches.png", dpi=150)
    plt.close(fig)

    # 3) Per-class feature — the single most class-selective channel for each lesion -------
    n_show = 4
    fig, axes = plt.subplots(n_classes, n_show, figsize=(1.5 * n_show, 1.6 * n_classes))
    axes = np.atleast_2d(axes)
    per_class = {}
    for r in range(n_classes):
        ch_for_class = np.where(best_class == r)[0]
        if len(ch_for_class) == 0:
            ch = int(classmean[:, r].argmax())
        else:
            ch = int(ch_for_class[np.argmax(selectivity[ch_for_class])])
        per_class[classes[r]] = {"channel": ch, "selectivity": float(selectivity[ch])}
        top_imgs = np.argsort(maxact[:, ch])[::-1][:n_show]
        for col, idx in enumerate(top_imgs):
            ax = axes[r, col]
            img = display_image(samples[idx][0], img_size)
            ax.imshow(np.clip(crop_patch(img, argloc[idx, ch], grid_hw, img_size), 0, 1))
            ax.set_xticks([]); ax.set_yticks([])
            if col == 0:
                ax.set_ylabel(f"{short[r]}\nch{ch}", fontsize=7, rotation=0,
                              ha="right", va="center", labelpad=24)
    fig.suptitle(f"Most class-selective channel per lesion (stage {args.stage})", fontsize=12)
    fig.tight_layout(rect=(0.08, 0, 1, 0.98))
    fig.savefig(FIG_DIR / "per_class_feature.png", dpi=150)
    plt.close(fig)

    # --- summary ------------------------------------------------------------------------
    summary = {
        "model": cfg["model"],
        "img_size": img_size,
        "stage": args.stage,
        "grid": list(grid_hw),
        "n_channels": int(maxact.shape[1]),
        "n_val": int(len(labels)),
        "top_selective_channels": [
            {"channel": int(c), "best_class": classes[best_class[c]],
             "selectivity": float(selectivity[c])}
            for c in order[: args.top_channels]
        ],
        "per_class_feature": per_class,
    }
    (FIG_DIR / "feature_summary.json").write_text(json.dumps(summary, indent=2))
    print(f"Saved 3 figures + feature_summary.json to {FIG_DIR}")


if __name__ == "__main__":
    main()
