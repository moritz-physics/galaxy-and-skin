# Architecture selection — record & conventions

This folder is the **decision record** for which backbone the skin model uses. The small
JSON/PNG/MD files here are git-tracked (the experiment is part of the project's history);
verbose Kaggle logs under `logs/` are not. The model selection pipeline has three stages,
each a Kaggle GPU notebook run at *proxy fidelity* (256 px, a stratified subset preserving
the class prior, short epochs) — proxy fidelity is enough because we only need the
**ranking** of candidates, not their final accuracy.

## Files

| File | Produced by | What it is |
|---|---|---|
| `threeway_results.json` | `skin/kaggle_bakeoff.ipynb` | 3-way bake-off: EfficientNetV2-S vs ConvNeXt-Small vs Swin-V2-S, single seed |
| `threeway.png` | `skin/kaggle_bakeoff.ipynb` | bar charts: clean bal-acc / ECE / darken bal-acc |
| `confirm_results.json` | `skin/kaggle_bakeoff_confirm.ipynb` | 3-seed re-run of the two finalists, mean ± std |
| `logs/` | (gitignored) | raw Kaggle kernel logs |

## The decision (2026-06)

**Chosen backbone: ConvNeXt-Small @ 256 px.**

The single-seed three-way bake-off *suggested* Swin-V2-S, largely on a standout ECE of
0.040. The 3-seed confirmation showed that **was a lucky seed** — Swin never repeated it
(0.089 / 0.084 / 0.108). Across seeds the two finalists are a statistical tie on accuracy
(0.788 vs 0.786, gap ≪ std) and calibration, and robustness splits in ConvNeXt's favour:

| metric (mean ± std, 3 seeds) | swin_v2_s | convnext_small |
|---|---|---|
| balanced accuracy | 0.788 ± 0.010 | 0.786 ± 0.011 |
| ECE ↓ | 0.094 ± 0.010 | 0.089 ± 0.015 |
| darken bal-acc | 0.469 ± 0.041 | **0.582 ± 0.042** |
| noise bal-acc | 0.178 ± 0.054 | **0.247 ± 0.033** |
| blur bal-acc | **0.475 ± 0.016** | 0.297 ± 0.040 |

ConvNeXt wins **darken** (the original motivating failure) and noise; Swin wins only blur.
ConvNeXt also has practical edges: resolution-flexible (Swin-V2-S is locked to 256-native
inputs), faster, conv inductive bias suited to small data, torchvision-native (drop-in for
the app). EfficientNetV2-S was clearly worst on every axis and was dropped.

**Lesson carried forward:** seed noise on the ~700-image val set is large enough to flip
rankings. Never trust a single run — `kaggle_hpo.ipynb` searches at one seed for speed but
**re-validates the top configs across seeds** before promoting anything.

## Conventions for future iterations

- **Where things live.** `skin/results/bakeoff/` = architecture selection;
  `skin/results/hpo/` = hyperparameter search (`hpo_results.json`, `best_params.json`,
  plots). Both are git-tracked for JSON/PNG/MD only (see `.gitignore`); checkpoints
  (`*.pt`), Optuna DBs (`*.db`), and logs are not.
- **Downloading from Kaggle.** The notebooks write to `/kaggle/working/results/`, so
  `kaggle kernels output <slug> -p <dir>/` recreates a nested `results/` subfolder and may
  nest further if your shell CWD is already `skin/`. After downloading, **flatten** into the
  flat names above (drop the inner `results/`, move logs into `logs/`). Mind your CWD — run
  the download from the **repo root**.
- **Re-running a stage.** Bump seeds or add candidates in the notebook's config cell; keep
  proxy fidelity identical across a comparison so numbers stay comparable. Append new result
  files with a clear suffix (e.g. `threeway_v2_results.json`) rather than overwriting, and
  update the table above.
