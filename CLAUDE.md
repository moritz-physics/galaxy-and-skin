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

## Model selection: bake-off → confirmation → HPO

Three Kaggle GPU notebooks, run in order, decide the backbone and its hyperparameters
*before* the full training run. All three run at **proxy fidelity** (256px, a stratified
subset that preserves the class prior so logit adjustment stays valid, short epochs): we
only need the **ranking** of candidates/configs, not final accuracy. They reuse the
training notebook's split / head-swap / imbalance recipe verbatim, and mirror
`04_analysis.py`'s ECE + perturbations, so the numbers are comparable to the full analysis.
Decision records and conventions live in **`skin/results/bakeoff/README.md`** (read it
first). The git-tracked artefacts are small JSON/PNG/MD only (see `.gitignore`);
checkpoints `*.pt`, Optuna DBs `*.db`, and Kaggle `*.log` are not.

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
3. **`skin/kaggle_hpo.ipynb`** — Optuna (TPE + Hyperband pruning) search over ConvNeXt-Small:
   `head_lr, ft_lr, weight_decay, label_smoothing, tau` (imbalance) and `aug_strength` (the
   darken-failure fix). Carries the seed-noise lesson: **searches at one seed for speed, then
   re-validates the top-K configs across seeds** and promotes on mean bal-acc with ties
   broken on ECE then darken. The study is a resumable SQLite file in
   `/kaggle/working/results/` (`load_if_exists=True`) so it survives Kaggle's session cap.
   Writes `best_params.json` + `hpo_results.json` + plots → download into
   `skin/results/hpo/`.

To promote: set `MODEL_NAME=convnext_small` + the winning HPs in `kaggle_training.ipynb`
and run the full-fidelity recipe.

## Skin model training in the cloud (`skin/kaggle_training.ipynb`, `skin/colab_training.ipynb`)

Two interchangeable notebooks run the **same** two-phase recipe and emit the **same**
artefacts (`<model>_best.pt`, `model_config.json`, `classes.json`, `training_log.json`,
plots) — local Mac (MPS) training is too slow/crash-prone for this:

- **`kaggle_training.ipynb`** — Kaggle **GPU** (CUDA + AMP). Reads the local *Skin Cancer
  MNIST: HAM10000* dataset (`kmader`, mounts at `/kaggle/input/skin-cancer-mnist-ham10000/`)
  rather than streaming from HF. HAM10000 has no split, so it builds a **seeded,
  `lesion_id`-grouped** train/val split (no leakage) and maps the metadata's abbreviated
  `dx` codes to the project's full class names — `sorted(DX_TO_CLASS.values())` reproduces
  the exact `classes.json` order, so a Kaggle model stays drop-in compatible. Outputs go to
  `/kaggle/working/results/`; download them into `skin/results/` after **Save Version**.
- **`colab_training.ipynb`** — free Colab **TPU** via PyTorch/XLA; streams the dataset from
  Hugging Face and persists to Drive. The Colab-specific efficiency notes below apply only
  to this notebook.

Keep the two notebooks' shared logic (head-swap, freeze, two-phase loop, early stopping) in
sync, and keep both in sync with `skin/skin_model.replace_head`. The **promoted model of
record is EfficientNet-B3 @ 300px** (in `skin/results/`; local capped-val balanced accuracy
**0.874**, melanoma AUC 0.977 — see `skin/CLINICAL_REPORT.md`), selected via the architecture
bake-off and a W&B hyperparameter sweep. Scripts load it by default; override with
`SKIN_MODEL_DIR=<dir>`. Earlier checkpoints are archived for comparison under
`skin/results/archive_*` (the EfficientNetV2-S @ 384px run, grouped-split 0.829 / local 0.731;
and a plain-CE V2-S baseline, 0.791). Architecture-agnostic: change `MODEL_NAME` / `IMG_SIZE`
in the config cell to try ConvNeXt, Swin, ViT, etc. Set `FT_EPOCHS` high — `EARLY_STOP_PATIENCE`
cuts it off when val stops improving. See `STRUCTURE.md` for the repo map and `skin/CHANGELOG.md`
for how the evaluation evolved.

**Class-imbalance handling (both notebooks, config-gated).** HAM10000 is ~67% melanocytic
nevi, which biases a plain-CE model toward the majority class (the plain-CE V2-S baseline had
melanoma recall ~0.29). The promoted model corrects this — grouped-split balanced accuracy
0.791→0.829, melanoma recall 0.29→0.39, ECE 0.088→0.071. One caveat surfaced in
`04_analysis.py`: the corrected model goes *confidently wrong* under heavy darkening (entropy
collapses while accuracy falls to chance), unlike the baseline — likely the logit-adjusted
prior saturating on far-OOD dark inputs; the fix is brightness/contrast train augmentation.
Two methods are wired into the three-phase training loop and must be kept in sync across both
notebooks:
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
