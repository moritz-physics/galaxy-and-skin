# Skin Lesion Classifier

Transfer learning on dermatoscopic images of skin lesions, with a focus — like
the [galaxy project](../README.md) next door — on **whether the model's
confidence can be trusted**, not just how often it is right. A fine-tuned
EfficientNet classifies a lesion into one of seven HAM10000 categories; an
offline analysis then measures how well-calibrated those probabilities are and
how they behave when the image is degraded.

> ⚠️ **Not medical advice.** This is a research/education demo trained on
> dermatoscope imagery (polarised light, ~10× magnification). Ordinary
> phone-camera photos are out of distribution, so predictions on them are
> unreliable by construction. See a dermatologist for anything real.

---

## ⭐ Model of record — EfficientNet-B3 @ 300px

The promoted model (loaded by default from `results/`) is **EfficientNet-B3 @
300px**, the best of several architectures we trialled. It is what the app and
analysis scripts use unless you override `SKIN_MODEL_DIR`.

| | |
|---|---|
| Architecture | EfficientNet-B3 (ImageNet → HAM10000) @ 300px |
| Val balanced accuracy | **0.874** (95% CI [0.852, 0.896]) on the local split |
| Calibration (ECE) | 0.048 (→ 0.040 with temperature scaling) |
| Melanoma AUC | **0.977** |
| ⚠️ Clinical caveat | melanoma sensitivity is only **0.45 at argmax** — see [`CLINICAL_REPORT.md`](CLINICAL_REPORT.md) |

Hyperparameters came from a Weights & Biases sweep
([`results/hpo/`](results/hpo/)). Previous models are kept under
`results/archive_*` for comparison. **New here?** Read
[`../STRUCTURE.md`](../STRUCTURE.md) for the repo map and
[`CHANGELOG.md`](CHANGELOG.md) for how the analysis evolved.

---

## Table of contents

- [Model of record](#-model-of-record--efficientnet-b3--300px)
- [The task](#the-task)
- [Dataset](#dataset)
- [Model architecture](#model-architecture)
- [Training recipe](#training-recipe)
- [Combating class imbalance](#combating-class-imbalance)
- [Results](#results)
- [Calibration & uncertainty](#calibration--uncertainty)
- [Robustness under perturbation](#robustness-under-perturbation)
- [The web app](#the-web-app)
- [Running it yourself](#running-it-yourself)
- [Files](#files)

---

## The task

Given a cropped dermatoscopic image, predict which of seven lesion types it is:

| Class | Plain name | Risk tier |
|-------|------------|-----------|
| `melanoma` | Melanoma | malignant |
| `basal_cell_carcinoma` | Basal cell carcinoma | malignant |
| `actinic_keratoses` | Actinic keratosis | pre-cancerous |
| `melanocytic_nevi` | Melanocytic nevus (mole) | benign |
| `benign_keratosis-like_lesions` | Benign keratosis | benign |
| `dermatofibroma` | Dermatofibroma | benign |
| `vascular_lesions` | Vascular lesion | benign |

The interesting part is not raw accuracy but **calibration**: a model that says
"90% melanoma" should be right about 90% of the time, and it should grow visibly
*unsure* when shown a degraded or out-of-distribution image. We measure both.

## Dataset

[HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
("Human Against Machine with 10000 training images"), streamed from the
[`marmal88/skin_cancer`](https://huggingface.co/datasets/marmal88/skin_cancer)
Hugging Face mirror so the full ~6 GB is never stored locally.

HAM10000 is heavily imbalanced — melanocytic nevi alone are ~67% of it — so the
default subset **caps each class** to rebalance the training signal, and the
validation split is capped for a clean, comparable metric. Two countermeasures
keep the rare-but-dangerous classes from being ignored: a **class-weighted
loss** and **balanced accuracy** (mean per-class recall) as the headline metric.

| Class | Train | Val |
|-------|------:|----:|
| actinic_keratoses | 315 | 82 |
| basal_cell_carcinoma | 487 | 100 |
| benign_keratosis-like_lesions | 500 | 100 |
| dermatofibroma | 110 | 30 |
| melanocytic_nevi | 500 | 100 |
| melanoma | 500 | 100 |
| vascular_lesions | 136 | 35 |
| **Total** | **2,548** | **547** |

Images are stored with their short side resized (LANCZOS) so that later
resize/crop never has to upscale. The training notebook can instead use the
**full** dataset — the class-weighted loss still compensates for the imbalance.

## Model architecture

The classifier is an **ImageNet-pretrained EfficientNet** with its 1000-class
head replaced by a fresh 7-class linear layer. The **model of record is
EfficientNet-B3 @ 300px** (promoted in `results/`), the best of several backbones
we trialled in an architecture bake-off and tuned with a Weights & Biases
hyperparameter sweep. Two earlier checkpoints are kept under `results/archive_*`
for comparison: an EfficientNetV2-S @ 384px run with the imbalance correctors
described [below](#combating-class-imbalance), and a plain-CE V2-S baseline.

| | **B3 — model of record** | V2-S + LA/cRT (archived) | V2-S baseline (archived) |
|---|---|---|---|
| Backbone | **EfficientNet-B3** | EfficientNetV2-S | EfficientNetV2-S |
| Input resolution | 300×300 | 384×384 | 384×384 |
| Imbalance handling | class-weighted CE | logit adjustment + cRT | class-weighted CE |
| Best val bal-acc (training split) | **0.955** | 0.829 (grouped) | 0.791 (grouped) |
| Local capped-val bal-acc | **0.874** | 0.731 | — |
| Melanoma recall (local val, argmax) | **0.45** | 0.39 | 0.29 |

> **Reading the numbers honestly.** The V2-S figures use the **leakage-free,
> `lesion_id`-grouped** Kaggle split — the trustworthy generalisation protocol.
> B3's **0.955** is its best epoch on its own training-time val split; its
> **0.874 / 0.45** come from the offline `04_analysis.py` / `08_clinical.py` runs
> on the local capped `data/val` set, which can overlap training images and so
> reads a little optimistically. Treat grouped-split numbers as the real
> generalisation estimate, and the local-val numbers as the basis for the
> *relative* calibration and clinical story (see
> [`CLINICAL_REPORT.md`](CLINICAL_REPORT.md)).

### Why EfficientNet

EfficientNet uses **compound scaling** — depth, width, and input resolution are
scaled together by a single coefficient rather than independently — so each step
up the family (B0 → B3 → …) buys accuracy at a near-optimal parameter/FLOP cost.
The backbone is built from **MBConv** blocks (inverted residuals with depthwise
separable convolutions and squeeze-and-excitation attention), which are cheap
relative to their representational power — ideal for fine-tuning on a small
medical dataset where a heavier model would simply overfit.

We also trialled the larger **EfficientNetV2-S** (Fused-MBConv blocks, 384px
native), but **B3 won the bake-off** on this small medical dataset — bigger isn't
always better when a heavier model just overfits. B3 is kept as the model of
record; the V2-S runs are archived under `results/archive_*`.

### Generic, swappable head

Rather than hard-code EfficientNet's `classifier[1]`, the head swap is
**architecture-agnostic** (`skin/skin_model.py::replace_head`): it locates the
last `nn.Linear` in whichever head container the backbone uses
(`.classifier` for EfficientNet/ConvNeXt, `.head` for Swin, `.heads` for ViT,
`.fc` for ResNet) and replaces it with a `Linear(in_features → 7)`. This means
you can set `MODEL_NAME = "convnext_small"` / `"swin_v2_t"` / `"vit_b_16"` in the
notebook and everything downstream — training, the app, the analysis — adapts
with no code changes. The app rebuilds the exact trained architecture by reading
`results/model_config.json`.

### Input pipeline

Inputs are normalised with ImageNet statistics (mean `[0.485, 0.456, 0.406]`,
std `[0.229, 0.224, 0.225]`) since the backbone was pretrained that way.
Evaluation resizes the short side to `round(img_size · 256/224)` then
centre-crops to `img_size`, preserving the 256/224 ratio of the standard
ImageNet eval recipe. Training adds light augmentation: random-resized crop,
horizontal **and vertical** flips (lesions have no canonical orientation), and
mild colour jitter.

## Training recipe

**Two-phase transfer learning**, plus an optional third rebalancing phase. Two
interchangeable notebooks run the same recipe and produce the same artefacts —
pick whichever compute you have:

- **`skin/kaggle_training.ipynb`** — Kaggle **GPU** (P100/T4), reads the local
  *Skin Cancer MNIST: HAM10000* dataset and uses mixed precision (AMP). See
  [Training on Kaggle](#training-on-kaggle) below.

1. **Head warm-up** — freeze the pretrained backbone, train only the new
   7-class head for a few epochs. This stops the randomly-initialised head from
   sending large, destructive gradients back through the good pretrained
   features.
2. **Full fine-tune** — unfreeze everything and train at a low learning rate,
   with **early stopping** on validation balanced accuracy (patience 6), so you
   can set the epoch budget high and only pay for the epochs that actually help.
3. **Classifier re-training (cRT)** — *optional, `CRT_EPOCHS > 0`._ Freeze the
   backbone again and re-train only the head on a class-balanced sampler. See
   [Combating class imbalance](#combating-class-imbalance) below for why.

| Hyperparameter | Value |
|---|---|
| Loss | logit-adjusted cross-entropy (`LOGIT_ADJUST_TAU=1`), or class-weighted CE |
| Optimiser | AdamW |
| Head LR / fine-tune LR / cRT LR | 5e-4 / 1.5e-4 / 1e-3 (from the W&B sweep) |
| Head / fine-tune / cRT epochs | 3 / up to 40 (early-stopped) / 5 |
| Batch size | 32 (B3 @ 300 on a Kaggle T4) |
| Selection metric | validation balanced accuracy |

The best checkpoint (by val balanced accuracy) is saved every time it improves,
so a disconnect or early stop always leaves the best weights on disk. A local
Apple-GPU (MPS) fallback (`02_finetune.py`, EfficientNet-B0) exists for when no
TPU is available.

### Combating class imbalance

HAM10000 is heavily long-tailed — melanocytic nevi are ~67% of it, dermatofibroma
~1% — so on the **full** training set a plain cross-entropy model collapses toward
the majority class (the plain-CE V2-S baseline's melanoma recall was only 0.29 for
exactly this reason; the logit-adjusted + cRT V2-S run raised it to 0.39). The capping trick
(`TRAIN_CAP`) fixes this by *discarding* majority images, which throws away data.
The notebooks instead wire in two modern, better-targeted correctors, each
independently toggleable in the config cell:

**1. Logit adjustment** (Menon et al., [*Long-tail learning via logit
adjustment*](https://arxiv.org/abs/2007.07314), ICLR 2021), `LOGIT_ADJUST_TAU`.
The loss is computed on `logits + τ·log P(y)` instead of the raw logits, which is
the Bayes-consistent correction for a known label distribution and is equivalent to
enforcing a larger margin for rarer classes. Because the prior is baked in *during
training*, the **raw logits at inference are already corrected** — the app and the
analysis script need no change. It is an *alternative* to inverse-frequency class
weighting (using both double-corrects), so `τ>0` switches the class weights off;
`τ=1` is the standard setting, `τ=0` disables it.

**2. Decoupled classifier re-training / cRT** (Kang et al., [*Decoupling
representation and classifier for long-tailed recognition*](https://arxiv.org/abs/1910.09217),
ICLR 2020), `CRT_EPOCHS`. The key empirical finding of that paper is that imbalance
mostly harms the *classifier*, not the learned *features*. So the backbone is
trained on the natural distribution (phases 1–2, seeing every nevi image), then
**frozen**, and only the head is re-trained for a few epochs on a class-balanced
sampler (phase 3) with plain cross-entropy. This rebalances the decision boundary
without disturbing the representation. Phase 3 carries the best val score forward,
so it only overwrites the checkpoint if it actually improves — enabling it can
never regress the saved model.

These two plus the existing **balanced-accuracy selection metric** (mean per-class
recall, so the majority class can't dominate model choice) are the levers in play.
Other standard options not wired in — effective-number reweighting, focal/LDAM
losses, MixUp/CutMix, per-class threshold tuning, and (the real fix for the rarest
classes) more external data — are noted as future work.

**Measured effect.** These correctors were validated on the **archived V2-S run**
(`LOGIT_ADJUST_TAU=1`, `CRT_EPOCHS=5`). Against the plain class-weighted-CE
baseline (same backbone, same split, archived under
`results/archive_efficientnet_v2_s_baseline/`), on the leakage-free `lesion_id`-grouped
Kaggle validation split:

| | Baseline (plain CE) | + logit adjustment + cRT |
|---|---|---|
| Grouped-split val balanced acc | 0.791 | **0.829** |
| Melanoma recall (local val) | 0.29 | **0.39** |
| Expected Calibration Error (local val) | 0.088 | **0.071** |

Logit adjustment did most of the work (the best epoch was in phase 2); the cRT
phase here matched but didn't beat it, so the phase-2 checkpoint was kept. The gain
that matters most is **melanoma recall climbing from 0.29 to 0.39** — fewer missed
cancers, exactly what correcting the nevi bias should buy.

### Training on Kaggle

`skin/kaggle_training.ipynb` is the main training notebook (Kaggle **GPU**, CUDA
+ AMP). It also logs live to Weights & Biases and its config carries the
sweep-tuned hyperparameters. HAM10000 ships no train/val split, so the notebook
builds its own —
**seeded and grouped by `lesion_id`** so the same lesion never appears in both
splits (it averages ~2 images per lesion; an ungrouped split would leak and
inflate the score). The abbreviated `dx` codes in the metadata are mapped to the
project's full class names, reproducing the exact class order in `classes.json`,
so a Kaggle-trained model is drop-in compatible with the app and analysis.

To run it:

1. **New Notebook** on Kaggle (Code → New Notebook), then **File → Import
   Notebook** and upload `skin/kaggle_training.ipynb`.
2. **Add Input** (right sidebar) → search *"Skin Cancer MNIST: HAM10000"* (by
   *kmader*) and add it. It mounts at `/kaggle/input/skin-cancer-mnist-ham10000/`.
3. **Settings → Accelerator → GPU T4 x2.** Use the **T4**, not the P100 —
   Kaggle's PyTorch build no longer supports the P100's older sm_60 compute
   capability, so it errors out (the notebook checks for this and fails fast with
   a clear message). The T4 is sm_75 and works; the notebook uses one GPU, so the
   second T4 idles.
4. **Run All.** Outputs are written to `/kaggle/working/results/`
   (`<model>_best.pt`, `model_config.json`, `classes.json`, `training_log.json`,
   `confusion_matrix.png`, `training_curves.png`, `progress.log`).
5. Click **Save Version** to persist `/kaggle/working/`, then download the
   contents of `results/` from the version's *Output* tab into this repo's
   `skin/results/` directory. The app reads `model_config.json` and adapts to
   whatever architecture you trained.

Change `MODEL_NAME` / `IMG_SIZE` in the config cell to try a different backbone;
lower `BATCH_SIZE` to 16 if you hit CUDA OOM.

## Results

The model of record, **EfficientNet-B3 @ 300px**, reaches **0.955** balanced
accuracy on its training-time val split. The figures below are the offline
`04_analysis.py` run on the 547-image local capped val set, where it scores
**0.845 accuracy / 0.874 balanced accuracy** (95% CI [0.852, 0.896]) with a
well-calibrated ECE of 0.048. See [`CLINICAL_REPORT.md`](CLINICAL_REPORT.md) for
the melanoma-sensitivity story the balanced accuracy hides.

![Per-class precision, recall and F1](results/figures/performance/per_class_f1.png)

![Confusion matrix](results/figures/performance/confusion_matrix.png)

Per-class F1 ranges from 1.00 on vascular lesions down to 0.62 on melanoma. The
most clinically important number is melanoma recall:

> **Melanoma recall is only 0.45 at argmax** — the model *misses* over half of
> melanomas, usually confusing them for benign nevi. Its melanoma **precision** is
> very high (0.98), so when it *does* call melanoma it is almost always right; it is
> just far too conservative about raising the alarm. Crucially, its melanoma AUC is
> 0.977 — the *ranking* is excellent, so **threshold tuning** recovers 0.91
> sensitivity at 0.93 specificity with no retraining. This is the central finding
> of [`CLINICAL_REPORT.md`](CLINICAL_REPORT.md), and why the app foregrounds the
> full probability distribution and an uncertainty warning rather than a single label.

## Calibration & uncertainty

Are the predicted probabilities trustworthy? The reliability diagram bins
predictions by confidence and plots confidence against actual accuracy; a
perfectly calibrated model sits on the diagonal.

![Reliability diagram, ECE 0.048](results/figures/calibration/reliability.png)

**Expected Calibration Error is 0.048** — well calibrated, with the curve sitting
slightly below the diagonal (mildly **overconfident**). Post-hoc **temperature
scaling** is implemented in `04_analysis.py`: fitting `T = 1.24` on a calibration
split and evaluating on a held-out half trims ECE from 0.054 to 0.040 — see
`results/figures/calibration/calibration_temperature.png`.

A second view: predictive entropy (normalised to [0, 1]) split by whether the
prediction was correct. A useful uncertainty signal should be low when the model
is right and high when it is wrong.

![Entropy of correct vs incorrect predictions](results/figures/calibration/uncertainty_split.png)

It is: mean entropy is **0.11 on correct** predictions vs **0.36 on incorrect**
ones. The app uses this same entropy to show a "low confidence" warning.

## Robustness under perturbation

The headline experiment. We corrupt the validation images with increasing
severity along three axes — additive Gaussian noise, blur, and darkening — and
track both balanced accuracy and mean uncertainty. A *trustworthy* model should
become **less accurate and more uncertain together**, so that low confidence
flags the degraded inputs.

![Accuracy and uncertainty under perturbation](results/figures/robustness/robustness.png)

As severity rises, balanced accuracy falls toward chance (~0.14 for 7 classes) for
all three corruptions, but the uncertainty signal only tracks it for two of them:

- **Gaussian noise**: accuracy 0.73 → 0.14, entropy 0.37 → 0.73 ✅ (uncertainty rises).
- **Blur**: accuracy 0.73 → 0.14, entropy 0.37 → 0.88 ✅ (uncertainty rises).
- **Darken**: accuracy 0.73 → 0.16, entropy 0.37 → **0.03** ❌ (uncertainty *collapses*).

So this model catches noise and blur — the entropy warning would flag those — but
**under heavy darkening it becomes confidently wrong**: accuracy falls to near
chance while it gets *more* certain, the dangerous failure mode for any
confidence-gated system. Curiously this is the opposite of the corruptions the
earlier checkpoints failed on (B3 broke under Gaussian noise; the plain-CE V2-S
baseline handled all three). The likely cause is that the logit-adjusted loss bakes
in the training-set prior, and on very dark, far-out-of-distribution inputs the
network saturates toward one class rather than spreading probability. The natural
fix is to add brightness/contrast corruption to the training augmentation (it is
currently only mild jitter) so the model learns to distrust badly-lit images.

This is the same "calibrated uncertainty under perturbations" question the galaxy
project studies, reproduced on a medical-imaging model — including a clean example
of a corruption that defeats the confidence signal.

## The web app

`03_app.py` is a [Gradio](https://www.gradio.app/) app. Upload or capture a
photo and it returns a risk-tiered verdict card (colour-coded benign /
pre-cancerous / malignant), the top-3 class probabilities, and a low-confidence
warning driven by the normalised entropy above. It rebuilds whatever
architecture was trained by reading `results/model_config.json`, so it works
unchanged whether the checkpoint is B3, V2-S, or a ConvNeXt/Swin/ViT you train
later.

## Running it yourself

From the `skin/` directory (everything here uses `__file__`-relative paths and
shares the repo's `uv` environment):

```bash
cd skin
uv run python 01_data.py      # stream + save the balanced subset (~10 min, one-time)
uv run python 02_finetune.py  # local two-phase fine-tune on Apple GPU (MPS) — fallback
uv run python 03_app.py       # web app: open on your phone over WiFi, upload a photo
uv run python 04_analysis.py  # offline calibration + robustness analysis
```

To (re)train, run `kaggle_training.ipynb` on a Kaggle **GPU** (see
[Training on Kaggle](#training-on-kaggle)) and download the results into
`skin/results/`. To search hyperparameters, run `kaggle_sweep.ipynb` (a W&B
sweep). Change `MODEL_NAME` / `IMG_SIZE` in the config cell to try another
backbone — the head-swap and freeze logic are generic.

Tests (run from the repo root):

```bash
uv run pytest skin/tests/
```

## Files

```
skin/
├── README.md             # this file
├── CLINICAL_REPORT.md    # clinically-honest evaluation (melanoma sensitivity)
├── CHANGELOG.md          # why the analysis evolved (read to catch up)
├── skin_model.py         # shared helpers: head-swap, model rebuild, preprocessing, entropy
│
├── 01_data.py            # stream HAM10000 subset from Hugging Face -> data/
├── 02_finetune.py        # local EfficientNet-B0 fine-tune on Apple GPU (fallback)
├── 03_app.py             # Gradio web app (reads results/model_config.json)
├── 04_analysis.py        # calibration, bootstrap CI, temperature scaling, error analysis
├── 05_visualize.py       # Grad-CAM, t-SNE, soft-confusion visualisations
├── 06_wandb_backfill.py  # push past runs into Weights & Biases
├── 07_wandb_eval.py      # rich W&B eval: interactive prediction table, ROC/PR
├── 08_clinical.py        # clinical metrics -> CLINICAL_REPORT.md figures
├── 09_wandb_report.py    # assemble a W&B Report from the runs
│
├── kaggle_training.ipynb # MAIN training (Kaggle GPU; sweep-tuned, W&B logging)
├── kaggle_sweep.ipynb    # W&B hyperparameter sweep (Bayesian + Hyperband)
├── kaggle_bakeoff*.ipynb # architecture comparison (how B3 was chosen) — historical
│
├── tests/                # unit tests for the shared helpers
├── data/                 # train/ and val/ image folders (gitignored)
└── results/              # ⭐ B3 model of record + configs/logs (gitignored)
    ├── figures/          # git-tracked, grouped by theme:
    │   ├── performance/      # confusion matrix, per-class F1, training curves
    │   ├── calibration/      # reliability, temperature scaling, entropy split
    │   ├── robustness/       # accuracy/uncertainty under perturbation
    │   ├── clinical/         # sensitivity, melanoma ROC, referral, missed cases
    │   └── interpretability/ # Grad-CAM, t-SNE
    ├── hpo/              # W&B sweep summary + plots (git-tracked)
    ├── bakeoff/          # architecture-bakeoff results (git-tracked)
    └── archive_*/        # previous models, kept for comparison/evolution
```

`data/` and the model checkpoints in `results/` are gitignored (large /
regenerable); the small artefacts — `results/figures/`, `hpo/`, `bakeoff/` — are
tracked. The app reads `results/model_config.json` to rebuild whatever
architecture is promoted, so it adapts to any backbone with no code changes. See
[`../STRUCTURE.md`](../STRUCTURE.md) for the whole-repo map.
