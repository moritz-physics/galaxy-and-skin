# CLAUDE.md

Guidance for working in this repo. Two separate projects, split at the top level:

- **`galaxy/`** (the primary project): calibrated uncertainty under perturbations for
  galaxy morphology classification. Code in `galaxy/scripts/` (entry points) and
  `galaxy/src/galaxy_uq/` (library). Artefacts in `galaxy/results/{figures,metrics}`;
  data in `galaxy/data/`; tests in `galaxy/tests/`. Galaxy scripts use CWD-relative
  paths, so **run them from the `galaxy/` directory** (`cd galaxy && uv run python
  scripts/01_eda.py`). Their `sys.path.insert(... parents[1]/"src")` resolves to
  `galaxy/src`, so imports work without installing the package.
- **`skin/`** (self-contained sub-project): transfer learning on HAM10000 skin lesions.
  Everything skin lives under `skin/` — code, the `skin/data/` subset, and
  `skin/results/` model artefacts. Keep both projects' data/results under their own
  top-level folder; nothing crosses over. `skin/skin_model.py` is the shared helper
  (generic head-swap, model rebuild from `model_config.json`, val preprocessing,
  normalised entropy) imported by the app, the analysis script, and the tests — keep
  it in sync with the training notebook's `replace_head`. `skin/04_analysis.py` runs
  calibration + perturbation-robustness analysis on the trained model **fully offline**
  (no Colab) and writes figures to `skin/results/figures/` (the one skin results path
  that is git-tracked). Skin tests live in `skin/tests/`.

Tests run from the repo root via `uv run pytest` (pyproject sets
`pythonpath = ["galaxy/src", "skin"]` and `testpaths = ["galaxy/tests", "skin/tests"]`).

`skin/data/` and most of `skin/results/` are gitignored (large; the exception is
`skin/results/figures/`). The app reads
`skin/results/model_config.json` to rebuild whatever architecture was trained, so it
adapts to any backbone without code changes.

## Skin model training on Google Colab (`skin/colab_training.ipynb`)

Training runs on a **free Colab TPU** via PyTorch/XLA, because local Mac (MPS) training
is slow and crash-prone for this. The last trained/promoted model is **EfficientNet-B3
@ 300px** (~0.955 val balanced accuracy); the notebook's current default is
**EfficientNetV2-S @ 384px** on the full dataset with early stopping (not yet run). The
notebook is architecture-agnostic: change `MODEL_NAME` / `IMG_SIZE` in the config cell to
try ConvNeXt, Swin, ViT, etc. — the head-swap and freeze logic are generic. Two-phase
fine-tuning (head, then full) with early stopping on val balanced accuracy; `FT_EPOCHS`
can be set high because `EARLY_STOP_PATIENCE` cuts it off when it stops improving.

### Colab compute efficiency — what is already built in

These are implemented in the notebook; preserve them when editing:

- **Drive for persistence, local disk for speed.** Checkpoints/logs save to
  `MyDrive/galaxy-uq/results/skin/` (survive disconnects). Image data lives on the VM's
  local `/content/` disk, never read directly off Drive (Drive is slow for many small
  files).
- **Dataset cached as a single tar on Drive.** Keyed by `VARIANT`
  (`skin_t{TRAIN_CAP}_v{VAL_CAP}_s{SHORT_SIDE}.tar.gz`). First run downloads from Hugging
  Face (~10–15 min) and caches; later sessions restore in seconds. Changing the caps or
  `SHORT_SIDE` makes a new cache automatically — no stale reuse.
- **Best-checkpoint-to-Drive every improving epoch**, plus epoch summaries mirrored to
  `progress.log` on Drive so training is monitorable even if the browser tab dies.
- **`FORCE_RETRAIN` guard.** If a checkpoint for the current `MODEL_NAME` already exists,
  training is skipped and the model is loaded. Set `FORCE_RETRAIN = True` to retrain.
- **Dependencies installed once at the top**; pinned `torch_xla` matched to Colab's torch.
- **Memory-cleanup cell** at the end frees the model/loaders + clears the XLA/CUDA cache,
  so you can change `MODEL_NAME` and re-run **without restarting the runtime** (which
  would otherwise risk OOM when stacking architectures).

### Manual habits that save Colab quota (cannot be coded)

- **Use a CPU runtime while editing code or doing analysis**; only switch to TPU
  (Runtime → Change runtime type) when actually training. TPU/GPU time is the metered
  resource on the free tier.
- **Watch the Resources tab** (top-right) for RAM/disk pressure before it crashes a run.
- **Raising `IMG_SIZE` above ~300** (e.g. EfficientNetV2-S at 384) requires raising
  `SHORT_SIDE` to match (≥ the val resize), otherwise stored images get upscaled. This
  triggers a one-time re-download because the cache key changes.

## Conventions

- Python 3.11, managed with `uv`. Run scripts with `uv run python ...`.
- Galaxy scripts run from the repo root; skin scripts use `__file__`-relative paths and
  run from anywhere (README shows `cd skin`).
- Tests: `uv run pytest tests/`.
