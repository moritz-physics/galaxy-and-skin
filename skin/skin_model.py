"""Shared model helpers for the skin sub-project.

One source of truth for the things the training notebook, the Gradio app, the
analysis script, and the tests must all agree on: how to swap a torchvision
classifier head, how to rebuild a trained model from the saved config, the
validation preprocessing pipeline, and the normalised-entropy uncertainty score.

Kept dependency-light (torch / torchvision only) and `__file__`-relative so the
skin scripts stay runnable from anywhere.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models, transforms

# ImageNet statistics — the pretrained backbones expect inputs normalised this way.
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Head containers across torchvision families, in priority order:
# EfficientNet/ConvNeXt -> .classifier, Swin -> .head, ViT -> .heads, ResNet -> .fc
_HEAD_ATTRS = ["classifier", "head", "heads", "fc"]


def replace_head(model: nn.Module, n_classes: int) -> nn.Module:
    """Swap the final classification layer for a fresh ``n_classes`` Linear.

    Different torchvision families name the head differently, so the last
    ``nn.Linear`` is located generically. This is the same logic used in the
    training notebook; keeping it here means the app rebuilds *exactly* the
    architecture that was trained, whatever the backbone.
    """
    for attr in _HEAD_ATTRS:
        head = getattr(model, attr, None)
        if head is None:
            continue
        if isinstance(head, nn.Linear):
            setattr(model, attr, nn.Linear(head.in_features, n_classes))
            return model
        # head is a Sequential / module: replace the last Linear inside it
        last_linear_name = None
        for name, module in head.named_modules():
            if isinstance(module, nn.Linear):
                last_linear_name = name
        if last_linear_name is not None:
            parent = head
            *path, leaf = last_linear_name.split(".")
            for p in path:
                parent = getattr(parent, p)
            in_features = getattr(parent, leaf).in_features
            setattr(parent, leaf, nn.Linear(in_features, n_classes))
            return model
    raise ValueError(f"Could not find a classifier head on {type(model).__name__}")


def build_model(model_name: str, n_classes: int, weights=None) -> nn.Module:
    """Construct a torchvision model and swap its head to ``n_classes`` outputs."""
    model = models.get_model(model_name, weights=weights)
    return replace_head(model, n_classes)


def val_transform(img_size: int) -> transforms.Compose:
    """Deterministic eval preprocessing: resize (keeping the 256/224 ratio used at
    training time), centre-crop to ``img_size``, tensor, ImageNet-normalise."""
    return transforms.Compose(
        [
            transforms.Resize(round(img_size * 256 / 224)),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def normalized_entropy(probs) -> float:
    """Shannon entropy of a probability vector scaled to [0, 1].

    0 = fully confident (mass on one class), 1 = uniform (maximally unsure).
    Accepts a list or 1-D tensor.
    """
    p = probs.tolist() if isinstance(probs, torch.Tensor) else list(probs)
    n = len(p)
    if n <= 1:
        return 0.0
    entropy = -sum(pi * math.log(pi + 1e-12) for pi in p)
    return entropy / math.log(n)


def load_trained_model(results_dir: Path, device: torch.device | None = None):
    """Rebuild and load the trained model described by ``results_dir``.

    Reads ``model_config.json`` (model name, img size, checkpoint filename) and
    ``classes.json``, constructs the matching architecture, loads the weights,
    and returns ``(model, classes, config)``. Falls back to the legacy
    EfficientNet-B0 layout for checkpoints that predate ``model_config.json``.
    """
    results_dir = Path(results_dir)
    classes = json.loads((results_dir / "classes.json").read_text())

    cfg_path = results_dir / "model_config.json"
    if cfg_path.exists():
        cfg = json.loads(cfg_path.read_text())
    else:
        cfg = {"model": "efficientnet_b0", "img_size": 224, "checkpoint": "effnet_b0_best.pt"}

    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    model = build_model(cfg["model"], len(classes))
    state = torch.load(results_dir / cfg["checkpoint"], map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.to(device).eval()
    return model, classes, cfg
