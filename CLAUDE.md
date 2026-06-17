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
  (no cloud needed) and writes figures to `skin/results/figures/` (the one skin results path
  that is git-tracked). Skin tests live in `skin/tests/`.

Tests run from the repo root via `uv run pytest` (pyproject sets
`pythonpath = ["galaxy/src", "skin"]` and `testpaths = ["galaxy/tests", "skin/tests"]`).

`skin/data/` and most of `skin/results/` are gitignored (large; the exception is
`skin/results/figures/`). The app reads
`skin/results/model_config.json` to rebuild whatever architecture was trained, so it
adapts to any backbone without code changes.

## Model selection: bake-off → confirmation → HPO

Three Kaggle GPU notebooks, run in order, decide the backbone and its hyperparameters
*before* the full training run. All three run at **proxy fidelity** (256px, a stratified
subset that preserves the class prior so logit adjustment stays valid, short epochs): we
only need the **ranking** of candidates/configs, not final accuracy. They reuse the
training notebook's split / head-swap / imbalance recipe verbatim, and mirror
`04_analysis.py`'s ECE + perturbations, so the numbers are comparable to the full analysis.
Decision records and conventions live in **`skin/results/bakeoff/README.md`** (read it
first). The git-tracked artefacts are small JSON/PNG/MD only (see `.gitignore`);
checkpoints `*.pt` and Kaggle `*.log` are not.

1. **`skin/kaggle_bakeoff.ipynb`** — ranks torchvision-native backbones
   (`efficientnet_v2_s`, `convnext_small`, `swin_v2_s`; torchvision-native keeps the winner
   drop-in loadable by the app — ConvNeXt-V2 / DINOv2 are timm-only and would need a
   `skin_model` loader change) on balanced accuracy, ECE, and darken/noise/blur. Single seed.
2. **`skin/kaggle_bakeoff_confirm.ipynb`** — re-runs the top finalists across **3 seeds** on
   a fixed split, reporting mean ± std. This exists because **seed noise on the ~700-image
   val set is large enough to flip rankings**: the single-seed bake-off favoured Swin-V2-S
   largely on an ECE of 0.040 that turned out to be a lucky seed (never repeated). Across
   seeds the finalists tie on accuracy/calibration and **ConvNeXt-Small wins robustness**
   (darken + noise), so ConvNeXt @ 256px is the chosen backbone.
3. **`skin/kaggle_sweep.ipynb`** — a **Weights & Biases** hyperparameter sweep (method
   `bayes` + Hyperband early-termination) over a cheap proxy (EfficientNet-B3 @ 224px, capped
   data) covering `head_lr, ft_lr, weight_decay, logit_adjust_tau, batch_size, crt_epochs`.
   It *ranks* hyperparameters without burning Kaggle quota; results live in the W&B project
   `skin-lesion → Sweeps`, with a downloaded summary in `skin/results/hpo/`. (Replaced an
   earlier Optuna notebook, which was never used and has been removed.)

The architecture bake-off explored ConvNeXt/Swin/V2-S, but the model ultimately promoted as
the **model of record is EfficientNet-B3 @ 300px** (see below). To re-promote a different
backbone: set `MODEL_NAME` + the sweep-winning HPs in `kaggle_training.ipynb` and run the
full-fidelity recipe.

## Skin model training in the cloud (`skin/kaggle_training.ipynb`)

`kaggle_training.ipynb` runs the two-phase recipe on a Kaggle **GPU** (CUDA + AMP) — local
Mac (MPS) training is too slow/crash-prone for this. It reads the local *Skin Cancer MNIST:
HAM10000* dataset (`kmader`, mounts at `/kaggle/input/skin-cancer-mnist-ham10000/`). HAM10000
has no split, so it builds a **seeded, `lesion_id`-grouped** train/val split (no leakage) and
maps the metadata's abbreviated `dx` codes to the project's full class names —
`sorted(DX_TO_CLASS.values())` reproduces the exact `classes.json` order, so a Kaggle model
stays drop-in compatible. It logs live to W&B (key from the Kaggle `WANDB_API_KEY` secret)
and emits `<model>_best.pt`, `model_config.json`, `classes.json`, `training_log.json`, plots
to `/kaggle/working/results/`; download them into `skin/results/` after **Save Version**.

Keep the training notebook's recipe logic (head-swap, freeze, two-phase loop, early stopping)
in sync with `skin/skin_model.replace_head`. The **promoted model of
record is EfficientNet-B3 @ 300px** (in `skin/results/`; local capped-val balanced accuracy
**0.874**, melanoma AUC 0.977 — see `skin/CLINICAL_REPORT.md`), selected via the architecture
bake-off and a W&B hyperparameter sweep. Scripts load it by default; override with
`SKIN_MODEL_DIR=<dir>`. Earlier checkpoints are archived for comparison under
`skin/results/archive_*` (the EfficientNetV2-S @ 384px run, grouped-split 0.829 / local 0.731;
and a plain-CE V2-S baseline, 0.791). Architecture-agnostic: change `MODEL_NAME` / `IMG_SIZE`
in the config cell to try ConvNeXt, Swin, ViT, etc. Set `FT_EPOCHS` high — `EARLY_STOP_PATIENCE`
cuts it off when val stops improving. See `STRUCTURE.md` for the repo map and `skin/CHANGELOG.md`
for how the evaluation evolved.

**Class-imbalance handling (config-gated, in the training notebook).** HAM10000 is ~67% melanocytic
nevi, which biases a plain-CE model toward the majority class (the plain-CE V2-S baseline had
melanoma recall ~0.29). The archived logit-adjusted + cRT V2-S run corrected this —
grouped-split balanced accuracy 0.791→0.829, melanoma recall 0.29→0.39, ECE 0.088→0.071. One caveat surfaced in
`04_analysis.py`: the corrected model goes *confidently wrong* under heavy darkening (entropy
collapses while accuracy falls to chance), unlike the baseline — likely the logit-adjusted
prior saturating on far-OOD dark inputs; the fix is brightness/contrast train augmentation.
Two methods are wired into the three-phase training loop:
- **Logit adjustment** (`LOGIT_ADJUST_TAU`, default 1.0; Menon et al. 2021) — phases 1–2
  train on `logits + τ·log P(y)` via the `LogitAdjustedLoss` module. It *replaces*
  inverse-frequency class weighting (`τ>0` switches the weighted CE off; the two would
  double-correct). The prior is baked in during training, so **inference uses raw logits and
  the app / `04_analysis.py` need no change**.
- **Decoupled classifier re-training / cRT** (`CRT_EPOCHS`, default 5; Kang et al. 2020) —
  phase 3 reloads phase-2's best weights, freezes the backbone, and re-trains only the head
  on a `WeightedRandomSampler`-balanced loader with plain CE. It carries `best` forward, so
  it only overwrites the checkpoint if val balanced accuracy improves.

Set either knob to 0 to ablate. Both are documented in the README's *Combating class
imbalance* section. In the promoted run, logit adjustment did the work (best epoch was in
phase 2); cRT matched but didn't beat it, so the phase-2 checkpoint was kept.

## Conventions

- Python 3.11, managed with `uv`. Run scripts with `uv run python ...`.
- Galaxy scripts run from the repo root; skin scripts use `__file__`-relative paths and
  run from anywhere (README shows `cd skin`).
- Tests: `uv run pytest tests/`.
