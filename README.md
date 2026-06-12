# galaxy-uq

A study of **calibrated uncertainty under perturbations** for galaxy morphology
classification. Three classifiers are compared on the Galaxy10 DECaLS dataset
(17,736 RGB images, 256×256, 10 morphology classes), with the focus on how well
each model's predicted probabilities remain calibrated when the inputs are
corrupted. Cross-validation, hyperparameter tuning, and evaluation metrics are
all implemented from scratch.

## Repository layout

Two separate projects, split at the top level:

- **`galaxy/`** — the main project described above (`scripts/`, `src/galaxy_uq/`,
  `tests/`, `report/`, `notebooks/`, plus its own `data/` and `results/`).
- **`skin/`** — a self-contained sub-project: transfer learning on HAM10000 skin
  lesions, with its own code, `data/`, and `results/`. See its section below.

Galaxy scripts read and write paths relative to the working directory, so **run
them from the `galaxy/` directory**.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) with Python 3.11. Install
the dependencies into a virtual environment:

```bash
uv sync
```

### Download the data

The dataset is not tracked in git. Download the Galaxy10 DECaLS HDF5 file into
`galaxy/data/raw/` and verify its checksum:

```bash
mkdir -p galaxy/data/raw
curl -L -o galaxy/data/raw/Galaxy10_DECals.h5 \
  https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5

# Expected SHA256:
# 19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571
shasum -a 256 galaxy/data/raw/Galaxy10_DECals.h5
```

The file is ~2.54 GB.

## Reproducing task 1 (EDA)

Run the exploratory data analysis from the `galaxy/` directory:

```bash
cd galaxy
uv run python scripts/01_eda.py
```

This loads the dataset, logs ground-truth integrity checks, writes five
diagnostic figures to `results/figures/`, and a JSON summary to
`results/metrics/01_eda_summary.json`. Pass `--seed` (default 0) to control
random sampling and `--out-dir` to change the figure destination.

## Reproducing results

Run the scripts from the `galaxy/` directory (`cd galaxy`) in the order below.
All scripts write figures to `results/figures/` and metric JSON / NPZ artefacts
to `results/metrics/`. Runtimes are approximate, measured on an Apple M-series
laptop.

| # | Script | Purpose | Approx. runtime |
|---|--------|---------|-----------------|
| 1 | `scripts/01_eda.py` | Dataset integrity checks and EDA figures. | ~1 min |
| 2 | `scripts/02_features.py` | Extract handcrafted features (HOG, colour, shape) from raw 256×256 images. | ~10 min |
| 3 | `scripts/03_logreg.py` | Logistic regression baseline under nested CV. | ~5 min |
| 4 | `scripts/04_rf.py` | Random forest baseline under nested CV. | ~30 min |
| 5 | `scripts/05_cnn.py` | Deep-ensemble CNN under nested CV (the headline model). | ~3–4 h |
| 6 | `scripts/05b_analysis.py` | Cross-model comparison and aggregate plots. | ~1 min |
| 7 | `scripts/05c_temperature_scaling.py` | Post-hoc temperature scaling for calibration. | ~1 min |
| 8 | `scripts/05d_uncertainty_decomposition.py` | Epistemic / aleatoric uncertainty decomposition. | ~1 min |
| 9 | `scripts/05e_failure_analysis.py` | Confidently-wrong and high-entropy failure cases. | ~1 min |
| 10 | `scripts/05f_boundary_cases.py` | Class-boundary / ambiguous sample analysis. | ~1 min |

For the CNN, a smoke-test mode runs in a few minutes instead of hours and is
useful for verifying the pipeline end-to-end before the full run:

```bash
uv run python scripts/05_cnn.py --fast   # img_size=32, epochs=5, M=2
uv run python scripts/05_cnn.py          # full pipeline (default)
```

## Tests

From the repository root (pytest is configured to find `galaxy/tests`):

```bash
uv run pytest
```

## Second use case: skin lesion classification (`skin/`)

A self-contained sub-project for transfer learning on medical images, sharing
this repo's environment. Everything skin-related — code, the data subset, and
trained model artefacts — lives under `skin/`, keeping it fully separate from
the galaxy work above. It fine-tunes an ImageNet-pretrained EfficientNet on a
balanced subset of [HAM10000](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
(7 lesion classes), streamed from Hugging Face so the full ~6 GB is never
stored locally.

The current trained model is **EfficientNet-B3 @ 300px** (≈0.955 validation
balanced accuracy), trained on a free Colab TPU via `skin/colab_training.ipynb`.
That notebook is now configured to train **EfficientNetV2-S @ 384px** on the full
dataset with early stopping — open it in Colab and run all cells. The local
scripts below train a smaller EfficientNet-B0 on Apple GPU (MPS) as a fallback
when no TPU is available.

```bash
cd skin
uv run python 01_data.py      # stream + save balanced subset (~10 min, one-time)
uv run python 02_finetune.py  # local two-phase fine-tune on Apple GPU (MPS)
uv run python 03_app.py       # web app: open on iPhone over WiFi, upload a photo
```

For the better TPU-trained model, open `skin/colab_training.ipynb` in Google
Colab (Runtime → TPU), run all cells, then download the contents of Drive's
`galaxy-uq/results/skin/` into `skin/results/`. The app reads
`skin/results/model_config.json` and loads whichever architecture was trained.

Artefacts go to `skin/results/` and the data subset to `skin/data/` (both
gitignored). **Not medical advice** — HAM10000 contains dermatoscope images, so
phone-camera photos are out of distribution and predictions on them are
unreliable by construction.
