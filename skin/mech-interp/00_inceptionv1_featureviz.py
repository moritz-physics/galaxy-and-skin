"""Feature visualization on InceptionV1 — the mechanistic-interpretability warm-up.

Before touching the skin model we learn the technique on the canonical mech-interp
network: InceptionV1 / GoogLeNet (the model behind the famous Distill "Circuits" work).
It is small (6.6M params) and runs comfortably on an M1 CPU.

This is *activation maximization* — the iconic mech-interp output. We synthesise, from
noise, the image that most excites a single channel, answering "what does this unit want
to see?". No dataset is needed: the image is optimised, not retrieved.

The recipe is the standard lucid one, which is what makes the images look like features
rather than adversarial static:
  - Fourier (spectral) image parameterization with 1/f scaling  -> natural frequency content
  - colour decorrelation                                        -> plausible colours
  - transform robustness (jitter / scale / rotate every step)   -> features, not noise
  - gradient ascent (Adam) on a target channel's mean activation

Output: ``mech-interp/figures/inceptionv1/`` — one montage spanning layers shallow→deep,
showing how features grow from simple textures to complex object-parts with depth.

Runs locally:  uv run python mech-interp/00_inceptionv1_featureviz.py
                 [--size 160 --steps 256 --layers inception3b,inception4c,inception4e,inception5b]
"""

from __future__ import annotations

import argparse
import ssl
from pathlib import Path

import certifi  # macOS ships without a usable CA bundle; point torch.hub at certifi's
ssl._create_default_https_context = lambda *a, **k: ssl.create_default_context(cafile=certifi.where())

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import models

HERE = Path(__file__).resolve().parent
FIG_DIR = HERE / "figures" / "inceptionv1"

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

# Lucid's empirical ImageNet colour-correlation (sqrt of the colour covariance). Mixing a
# decorrelated image through it gives the muted, natural palette of real photos.
_COLOR_CORR = torch.tensor(
    [[0.26, 0.09, 0.02], [0.27, 0.00, -0.05], [0.27, -0.09, 0.03]]
)
_COLOR_CORR = _COLOR_CORR / torch.linalg.norm(_COLOR_CORR, dim=1).max()


# --- Fourier image parameterization ------------------------------------------
def rfft_freqs(h: int, w: int) -> np.ndarray:
    """Spatial-frequency magnitude at each rfft2 coefficient, shape [h, w//2+1]."""
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    return np.sqrt(fy**2 + fx**2)


class FourierImage:
    """A learnable image parameterised by its (scaled) Fourier spectrum."""

    def __init__(self, size: int, sd: float = 0.01, device="cpu"):
        freqs = rfft_freqs(size, size)
        scale = 1.0 / np.maximum(freqs, 1.0 / size)  # 1/f: suppress high frequencies
        scale *= np.sqrt(size * size)
        self.scale = torch.tensor(scale, dtype=torch.float32, device=device)[None, :, :]
        init = torch.randn(3, *freqs.shape, 2, device=device) * sd
        self.spectrum = init.requires_grad_(True)
        self.size = size

    def parameters(self):
        return [self.spectrum]

    def image(self) -> torch.Tensor:
        """Return a [1, 3, size, size] image in [0, 1]."""
        s = self.spectrum * self.scale[..., None]
        c = torch.complex(s[..., 0], s[..., 1])
        img = torch.fft.irfft2(c, s=(self.size, self.size)) / 4.0  # lucid magnitude constant
        # colour-decorrelate, then squash to [0,1]
        t = torch.einsum("chw,dc->dhw", img, _COLOR_CORR.to(img.device))
        return torch.sigmoid(t)[None]


# --- transform robustness ----------------------------------------------------
def jitter_scale_rotate(img: torch.Tensor, pad: int = 12) -> torch.Tensor:
    """Random pad+shift+scale+rotate so the optimiser can't exploit single pixels."""
    img = F.pad(img, [pad] * 4, mode="reflect")
    angle = float(torch.empty(1).uniform_(-10, 10))
    scale = float(torch.empty(1).uniform_(0.9, 1.1))
    tx = int(torch.randint(-8, 9, (1,)))
    ty = int(torch.randint(-8, 9, (1,)))
    img = TF.affine(img, angle=angle, translate=[tx, ty], scale=scale, shear=[0.0, 0.0],
                    interpolation=TF.InterpolationMode.BILINEAR)
    return img[:, :, pad:-pad, pad:-pad]


def normalize(img: torch.Tensor) -> torch.Tensor:
    return (img - IMAGENET_MEAN.to(img.device)) / IMAGENET_STD.to(img.device)


# --- optimization ------------------------------------------------------------
def visualize_channel(model, layer_module, channel, size, steps, lr, device):
    """Optimise an image to maximise one channel's mean activation. Returns HWC [0,1]."""
    captured = {}
    handle = layer_module.register_forward_hook(lambda m, i, o: captured.__setitem__("a", o))
    param = FourierImage(size, device=device)
    opt = torch.optim.Adam(param.parameters(), lr=lr)
    try:
        for _ in range(steps):
            opt.zero_grad()
            img = param.image()
            x = normalize(jitter_scale_rotate(img))
            model(x)
            loss = -captured["a"][0, channel].mean()
            loss.backward()
            opt.step()
        with torch.no_grad():
            final = param.image()[0].permute(1, 2, 0).cpu().numpy()
    finally:
        handle.remove()
    return np.clip(final, 0, 1), float(-loss.detach())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--layers", default="inception3b,inception4c,inception4e,inception5b",
                    help="comma-separated InceptionV1 blocks, shallow→deep")
    ap.add_argument("--per-layer", type=int, default=4, help="channels visualised per layer")
    ap.add_argument("--size", type=int, default=160, help="image size in px")
    ap.add_argument("--steps", type=int, default=256, help="optimization steps per channel")
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--channels", default="",
                    help="optional explicit channels for a single --layers entry, e.g. 0,50,100")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cpu")  # small net; CPU is fine and avoids MPS fft gaps

    model = models.googlenet(weights=models.GoogLeNet_Weights.IMAGENET1K_V1).to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    modules = dict(model.named_modules())

    layers = [s.strip() for s in args.layers.split(",") if s.strip()]
    explicit = [int(c) for c in args.channels.split(",") if c.strip()] if args.channels else None
    rng = np.random.default_rng(args.seed)

    rows = []
    for layer in layers:
        mod = modules[layer]
        n_ch = mod.branch1.conv.out_channels if hasattr(mod, "branch1") else None
        # InceptionV1 block output = concat of 4 branches; infer total channels from a probe
        with torch.no_grad():
            probe = {}
            h = mod.register_forward_hook(lambda m, i, o: probe.__setitem__("a", o))
            model(torch.zeros(1, 3, args.size, args.size))
            h.remove()
            total = probe["a"].shape[1]
        chans = explicit if (explicit and len(layers) == 1) else \
            sorted(rng.choice(total, size=min(args.per_layer, total), replace=False).tolist())
        print(f"{layer}: {total} channels — visualising {chans}")
        imgs = []
        for ch in chans:
            img, act = visualize_channel(model, mod, ch, args.size, args.steps, args.lr, device)
            imgs.append((ch, img, act))
            print(f"  ch{ch}: final activation {act:.2f}")
        rows.append((layer, imgs))

    # montage: one row per layer, shallow (top) → deep (bottom) --------------------------
    ncol = max(len(imgs) for _, imgs in rows)
    fig, axes = plt.subplots(len(rows), ncol, figsize=(2.1 * ncol, 2.3 * len(rows)))
    axes = np.atleast_2d(axes)
    for r, (layer, imgs) in enumerate(rows):
        for c in range(ncol):
            ax = axes[r, c]
            ax.set_xticks([]); ax.set_yticks([])
            if c < len(imgs):
                ch, img, _ = imgs[c]
                ax.imshow(img)
                ax.set_title(f"ch{ch}", fontsize=8)
            else:
                ax.axis("off")
        axes[r, 0].set_ylabel(layer, fontsize=10, rotation=0, ha="right", va="center", labelpad=34)
    fig.suptitle("InceptionV1 feature visualization — shallow (top) to deep (bottom)", fontsize=12)
    fig.tight_layout(rect=(0.04, 0, 1, 0.97))
    out = FIG_DIR / "featureviz_montage.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
