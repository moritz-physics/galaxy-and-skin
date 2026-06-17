# Repository structure

A quick map of this repo. It holds **two independent projects** that share a
Python environment but nothing else — keep their data/results under their own
top-level folder.

```
galaxy-uq/
├── README.md            ← repo overview
├── STRUCTURE.md         ← you are here
├── CLAUDE.md            ← notes for AI coding agents
├── pyproject.toml       ← one uv environment for both projects
│
├── galaxy/              ── PROJECT 1: galaxy morphology + uncertainty
│   ├── scripts/         ← numbered pipeline entry points (run from galaxy/)
│   ├── src/galaxy_uq/   ← library code
│   ├── data/ results/   ← inputs & artefacts (gitignored except figures/)
│   └── tests/
│
└── skin/               ── PROJECT 2: skin-lesion classifier (the active one)
    ├── README.md            ← full write-up; ⭐ model of record at the top
    ├── CLINICAL_REPORT.md   ← clinically-honest evaluation (melanoma sensitivity)
    ├── CHANGELOG.md         ← why the analysis evolved (read this to catch up)
    ├── skin_model.py        ← shared helpers (head-swap, model rebuild, preprocessing)
    │
    ├── 01_data.py           ← build the local train/val image split
    ├── 02_finetune.py       ← original local fine-tune (EfficientNet-B0) — historical
    ├── 03_app.py            ← Gradio demo app (serves results/ model)
    ├── 04_analysis.py       ← calibration, bootstrap CI, temperature scaling, errors
    ├── 05_visualize.py      ← Grad-CAM, t-SNE, soft-confusion visualisations
    ├── 06_wandb_backfill.py ← push past runs (training_log.json) into W&B
    ├── 07_wandb_eval.py     ← rich W&B eval: interactive prediction table, ROC/PR
    ├── 08_clinical.py       ← clinical metrics → CLINICAL_REPORT.md figures
    ├── 09_wandb_report.py   ← assemble a W&B Report from the runs
    │
    ├── kaggle_training.ipynb     ← MAIN training notebook (Kaggle GPU); sweep-tuned HPs
    ├── kaggle_sweep.ipynb        ← W&B hyperparameter sweep (Bayesian + Hyperband)
    ├── kaggle_bakeoff*.ipynb     ← architecture comparison (how B3 was chosen) — historical
    │
    ├── data/                 ← HAM10000 subset (gitignored)
    ├── tests/                ← pytest
    └── results/
        ├── efficientnet_b3_best.pt + model_config.json  ← ⭐ MODEL OF RECORD
        ├── figures/          ← tracked plots (analysis + clinical + sweep)
        ├── hpo/              ← sweep summary + importance plots (tracked)
        ├── bakeoff/          ← architecture-bakeoff results (tracked)
        └── archive_*/        ← previous models, kept for comparison/evolution
```

## How to run things

- **Galaxy** scripts use CWD-relative paths — run them from `galaxy/`:
  `cd galaxy && uv run python scripts/01_eda.py`
- **Skin** scripts use `__file__`-relative paths — run from anywhere, e.g.
  `cd skin && uv run python 04_analysis.py`. Point them at a different checkpoint
  with `SKIN_MODEL_DIR=results/archive_... uv run python 04_analysis.py`.
- **Tests** (both projects) from the repo root: `uv run pytest`.
- **Training** happens on Kaggle GPUs, not locally — open the notebooks
  above. Trained artefacts are downloaded back into `skin/results/`.

## The numbered skin scripts are a pipeline

Read `01 → 09` roughly in order: prepare data, train (notebooks), serve
(`03_app`), then the evaluation/uncertainty/clinical analyses (`04`, `05`, `08`)
and the experiment-tracking helpers (`06`, `07`, `09`). Files marked *historical*
(the architecture bake-off) are kept deliberately — they show how the project
evolved: architecture bake-off → best model → hyperparameter sweep → rigorous
evaluation.

## What's gitignored

Large or regenerable things: `data/`, model checkpoints (`*.pt`), local `wandb/`
run caches, `__pycache__/`, `.venv/`. Tracked under `results/` are only the small,
useful artefacts: `figures/`, `hpo/`, `bakeoff/`.
