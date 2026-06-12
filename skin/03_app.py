"""Gradio web app: upload a skin photo, get the model's top diagnoses.

Run on your Mac:   uv run python skin/03_app.py
Then on your iPhone (same WiFi), open the "Running on local URL" address
shown in the terminal (http://<your-mac-ip>:7860).

NOT a medical device. HAM10000 is dermatoscope imagery (polarized light,
10x magnification, no perspective distortion) — phone-camera photos are
out of distribution, so treat predictions as a toy. See a dermatologist
for anything real.
"""

import json
import math
from pathlib import Path

import gradio as gr
import pillow_heif
import torch
import torch.nn as nn
from torchvision import models, transforms

pillow_heif.register_heif_opener()

RESULTS = Path(__file__).resolve().parent / "results"
CLASSES = json.loads((RESULTS / "classes.json").read_text())

# model_config.json is written by the Colab training notebook; older runs
# (plain EfficientNet-B0) predate it, hence the fallback
CFG_PATH = RESULTS / "model_config.json"
if CFG_PATH.exists():
    CFG = json.loads(CFG_PATH.read_text())
else:
    CFG = {"model": "efficientnet_b0", "img_size": 224, "checkpoint": "effnet_b0_best.pt"}
CKPT = RESULTS / CFG["checkpoint"]

# Per-class display name, plain-language description, and clinical risk tier.
# Tiers drive the colour coding: a melanoma reading should never look the same
# as a harmless mole.
BENIGN, PRECANCER, MALIGNANT = "benign", "precancerous", "malignant"
LESIONS = {
    "melanoma": ("Melanoma", "The most serious skin cancer.", MALIGNANT),
    "basal_cell_carcinoma": ("Basal cell carcinoma", "A common, treatable skin cancer.", MALIGNANT),
    "actinic_keratoses": ("Actinic keratosis", "A sun-damage lesion that can turn cancerous.", PRECANCER),
    "melanocytic_nevi": ("Melanocytic nevus", "An ordinary mole. Harmless.", BENIGN),
    "benign_keratosis-like_lesions": ("Benign keratosis", "A harmless, non-cancerous growth.", BENIGN),
    "dermatofibroma": ("Dermatofibroma", "A harmless firm skin nodule.", BENIGN),
    "vascular_lesions": ("Vascular lesion", "A blood-vessel mark, mostly harmless.", BENIGN),
}
TIER_STYLE = {
    MALIGNANT: ("#dc2626", "Malignant"),
    PRECANCER: ("#d97706", "Pre-cancerous"),
    BENIGN: ("#16a34a", "Benign"),
}

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model = models.get_model(CFG["model"], weights=None)
model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASSES))
model.load_state_dict(torch.load(CKPT, map_location=device, weights_only=True))
model.to(device).eval()

IMG_SIZE = CFG["img_size"]
preprocess = transforms.Compose(
    [
        transforms.Resize(round(IMG_SIZE * 256 / 224)),  # same ratio as training val transform
        transforms.CenterCrop(IMG_SIZE),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
)


def _bar(label: str, prob: float, colour: str, strong: bool) -> str:
    pct = f"{prob * 100:.1f}%"
    weight = "600" if strong else "500"
    track = "#e5e7eb"
    return f"""
    <div class="bar-row">
      <div class="bar-head">
        <span style="font-weight:{weight}">{label}</span>
        <span class="bar-pct">{pct}</span>
      </div>
      <div class="bar-track" style="background:{track}">
        <div class="bar-fill" style="width:{prob * 100:.1f}%;background:{colour}"></div>
      </div>
    </div>"""


def _placeholder() -> str:
    return """
    <div class="verdict-empty">
      <div class="empty-icon">&#129516;</div>
      <p>Upload or capture a photo of a skin lesion to see the model's read.</p>
    </div>"""


def predict(image):
    if image is None:
        return _placeholder()

    x = preprocess(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probs = torch.softmax(model(x)[0], dim=0).cpu()

    ranked = sorted(zip(CLASSES, probs.tolist()), key=lambda kv: -kv[1])
    top_key, top_p = ranked[0]
    name, blurb, tier = LESIONS.get(top_key, (top_key, "", BENIGN))
    colour, tier_label = TIER_STYLE[tier]

    # Normalised entropy in [0, 1]: a flat distribution → ~1 (model unsure).
    entropy = -sum(p * math.log(p + 1e-12) for p in probs.tolist())
    uncertainty = entropy / math.log(len(CLASSES))

    if top_p < 0.45 or uncertainty > 0.7:
        warn = (
            '<div class="warn">&#9888;&#65039; <b>Low confidence.</b> The model is '
            "unsure — likely an unclear or out-of-distribution photo. Don't read into this result.</div>"
        )
    else:
        warn = ""

    bars = "".join(
        _bar(LESIONS.get(k, (k, "", BENIGN))[0], p, TIER_STYLE[LESIONS.get(k, (k, "", BENIGN))[2]][0], i == 0)
        for i, (k, p) in enumerate(ranked[:3])
    )

    return f"""
    <div class="verdict">
      <div class="verdict-top">
        <span class="tier-badge" style="background:{colour}">{tier_label}</span>
        <span class="conf">{top_p * 100:.0f}% confidence</span>
      </div>
      <h2 class="verdict-name">{name}</h2>
      <p class="verdict-blurb">{blurb}</p>
      {warn}
      <div class="breakdown-label">Top predictions</div>
      {bars}
    </div>"""


CSS = """
.gradio-container {max-width: 980px !important; margin: auto !important;}
#hero {text-align:center; padding: 8px 0 4px;}
#hero h1 {font-size: 2rem; font-weight: 700; letter-spacing:-0.02em; margin:0;}
#hero p {color:#6b7280; margin:6px 0 0; font-size:0.98rem;}
#disclaimer {background:#fffbeb; border:1px solid #fde68a; color:#92400e;
  border-radius:12px; padding:12px 16px; font-size:0.88rem; margin:14px 0 6px;}
#disclaimer b {color:#78350f;}
.verdict, .verdict-empty {border:1px solid #e5e7eb; border-radius:16px;
  padding:22px; background:#fff; min-height:300px;
  box-shadow:0 1px 2px rgba(0,0,0,0.04), 0 8px 24px rgba(0,0,0,0.04);}
.verdict-empty {display:flex; flex-direction:column; align-items:center;
  justify-content:center; text-align:center; color:#9ca3af; gap:10px;}
.empty-icon {font-size:2.6rem;}
.verdict-top {display:flex; align-items:center; justify-content:space-between;}
.tier-badge {color:#fff; font-weight:600; font-size:0.8rem; padding:5px 12px;
  border-radius:999px; letter-spacing:0.01em;}
.conf {color:#6b7280; font-size:0.85rem; font-weight:500;}
.verdict-name {font-size:1.6rem; font-weight:700; margin:14px 0 2px; letter-spacing:-0.02em;}
.verdict-blurb {color:#6b7280; margin:0 0 14px; font-size:0.95rem;}
.warn {background:#fef2f2; border:1px solid #fecaca; color:#991b1b;
  border-radius:10px; padding:10px 12px; font-size:0.86rem; margin-bottom:14px;}
.breakdown-label {text-transform:uppercase; letter-spacing:0.06em;
  font-size:0.72rem; color:#9ca3af; font-weight:600; margin:6px 0 10px;}
.bar-row {margin-bottom:12px;}
.bar-head {display:flex; justify-content:space-between; font-size:0.9rem; margin-bottom:5px;}
.bar-pct {color:#6b7280; font-variant-numeric:tabular-nums;}
.bar-track {height:9px; border-radius:999px; overflow:hidden;}
.bar-fill {height:100%; border-radius:999px; transition:width 0.4s ease;}
#footer {text-align:center; color:#9ca3af; font-size:0.8rem; margin-top:18px; line-height:1.6;}
"""

THEME = gr.themes.Soft(
    primary_hue="indigo",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "system-ui", "sans-serif"],
).set(body_background_fill="#f8fafc")

# one representative image per class, if the data subset is present locally
EXAMPLE_DIR = Path(__file__).resolve().parent / "data" / "val"
examples = []
if EXAMPLE_DIR.exists():
    for cls_dir in sorted(EXAMPLE_DIR.iterdir()):
        imgs = sorted(cls_dir.glob("*.jpg")) if cls_dir.is_dir() else []
        if imgs:
            examples.append([str(imgs[0])])

with gr.Blocks(title="Skin Lesion Classifier") as demo:
    gr.HTML(
        """
        <div id="hero">
          <h1>Skin Lesion Classifier</h1>
          <p>EfficientNet fine-tuned on dermatoscope images — """
        f"{CFG['model'].replace('_', '-').upper()}, 7 lesion types</p>"
        "</div>"
    )
    gr.HTML(
        """
        <div id="disclaimer">
          <b>Not medical advice.</b> This is a research demo trained on
          dermatoscope imagery. Phone-camera photos look very different to the
          model, so results on them are unreliable by design. For any real
          concern, see a dermatologist.
        </div>"""
    )

    with gr.Row(equal_height=False):
        with gr.Column(scale=1):
            image_in = gr.Image(
                type="pil",
                sources=["upload", "webcam"],
                label="Lesion photo",
                height=340,
            )
            with gr.Row():
                clear_btn = gr.ClearButton(image_in, value="Clear")
                run_btn = gr.Button("Analyze", variant="primary")
        with gr.Column(scale=1):
            verdict = gr.HTML(_placeholder())

    if examples:
        gr.Examples(
            examples=examples,
            inputs=image_in,
            label="Example dermatoscope images (one per class)",
        )

    gr.HTML(
        """
        <div id="footer">
          Model: """
        f"{CFG['model']} @ {IMG_SIZE}px &middot; trained on a balanced HAM10000 subset.<br>"
        "Predictions are probabilities, not diagnoses."
        "</div>"
    )

    run_btn.click(predict, inputs=image_in, outputs=verdict)
    image_in.change(predict, inputs=image_in, outputs=verdict)
    clear_btn.click(lambda: _placeholder(), outputs=verdict)


if __name__ == "__main__":
    demo.launch(theme=THEME, css=CSS, server_name="0.0.0.0", server_port=7860)
