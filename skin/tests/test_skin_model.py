"""Unit tests for the shared skin model helpers (no GPU / no training needed)."""

import math

import pytest
import torch
import torch.nn as nn
from PIL import Image

from skin_model import (
    build_model,
    normalized_entropy,
    replace_head,
    val_transform,
)


def _last_linear(model: nn.Module) -> nn.Linear:
    return [m for m in model.modules() if isinstance(m, nn.Linear)][-1]


# The whole point of the generic head-swap: it must work across torchvision
# families (.classifier / .head / .heads), not just EfficientNet. weights=None
# keeps this offline and fast.
@pytest.mark.parametrize(
    "model_name",
    ["efficientnet_b0", "efficientnet_v2_s", "convnext_small", "swin_v2_t", "vit_b_16"],
)
def test_replace_head_sets_output_classes(model_name):
    model = build_model(model_name, n_classes=7)
    assert _last_linear(model).out_features == 7


def test_replace_head_preserves_input_features():
    # swapping the head must keep the incoming feature dimension intact
    model = build_model("efficientnet_b0", n_classes=3)
    head = _last_linear(model)
    assert head.in_features == 1280  # EfficientNet-B0 penultimate width
    assert head.out_features == 3


def test_replace_head_on_bare_linear():
    model = nn.Module()
    model.fc = nn.Linear(16, 4)
    replace_head(model, 9)
    assert model.fc.in_features == 16
    assert model.fc.out_features == 9


def test_replace_head_raises_without_head():
    model = nn.Sequential(nn.Conv2d(3, 8, 3))  # no recognised head container
    with pytest.raises(ValueError):
        replace_head(model, 5)


def test_val_transform_output_shape():
    img = Image.new("RGB", (640, 480), color=(120, 80, 200))
    x = val_transform(300)(img)
    assert x.shape == (3, 300, 300)
    assert x.dtype == torch.float32


@pytest.mark.parametrize(
    "probs,expected",
    [
        ([1.0, 0.0, 0.0, 0.0], 0.0),          # certain -> 0
        ([0.25, 0.25, 0.25, 0.25], 1.0),      # uniform -> 1
    ],
)
def test_normalized_entropy_bounds(probs, expected):
    assert normalized_entropy(torch.tensor(probs)) == pytest.approx(expected, abs=1e-6)


def test_normalized_entropy_accepts_list_and_stays_in_unit_interval():
    val = normalized_entropy([0.5, 0.3, 0.2])
    assert 0.0 <= val <= 1.0
    # matches the closed-form Shannon entropy / log(n)
    expected = -sum(p * math.log(p) for p in [0.5, 0.3, 0.2]) / math.log(3)
    assert val == pytest.approx(expected, abs=1e-6)
