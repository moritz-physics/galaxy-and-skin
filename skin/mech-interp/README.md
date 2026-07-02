# mech-interp — mechanistic interpretability of the skin model

A clean workspace for probing *what the trained skin-lesion classifier has actually
learned*, separate from the calibration/robustness work in `skin/04_analysis.py`.

> **Narrative writeup with figures:** see [`REPORT.md`](REPORT.md) — a living,
> non-expert-friendly report of the findings so far. This README is the technical index.

- **What is interpreted:** the deployed network rebuilt by
  `skin_model.load_trained_model(results_dir)` — currently EfficientNet-B3 @ 300px
  (a fine-tuned ImageNet backbone + a 7-class head). No separate toy network: the
  point is to interpret *our* model.
- **Outputs:** everything visual/mathematical lands in `figures/`. Heavy/derived
  arrays (activation dumps, embeddings) go in `cache/` (gitignored).
- **Prerequisite:** the checkpoint (`results/<ckpt>.pt`, `results/model_config.json`,
  `results/classes.json`) and `data/val/` must be present locally. Both are
  gitignored — pull them from Drive/Colab before running anything here.

## Two tracks

- **Learn the technique on InceptionV1 first** (the canonical mech-interp model).
  No private data, runs on an M1 CPU. This is the warm-up before touching the skin model.
- **Apply it to our skin model** once comfortable. Needs the trained checkpoint +
  `data/val/`, which live on Drive/Colab — so the skin scripts run on a GPU box
  (Colab/Kaggle), not the local Mac.

## Files

- `00_inceptionv1_featureviz.py` — **runs locally on M1.** Activation maximization on
  InceptionV1: synthesise what each channel wants to see. Writes a shallow→deep montage
  to `figures/inceptionv1/`.
- `inceptionv1_kaggle.ipynb` — the GPU version of the same, for exploring many
  channels/layers fast. Self-contained (weights download; no upload). Set accelerator to
  GPU T4.
- `01_feature_extraction.py` — **skin model, GPU only.** Per-channel class selectivity +
  max-activating dataset patches. Needs the checkpoint + val data present.

## Roadmap (increasing difficulty, decreasing certainty of payoff)

0. **Feature visualization warm-up on InceptionV1** (done): activation maximization with
   the lucid recipe (Fourier param + colour decorrelation + transform robustness).
1. **Feature extraction on the skin model**: per-channel class selectivity +
   max-activating dataset examples — a labelled inventory of what the network detects.
2. **Feature-space geometry**: penultimate embeddings → PCA / UMAP coloured by class;
   channel-ablation importance ranking.
3. **Concept & circuits** (research-grade): TCAV for dermatological concepts,
   linear probes per layer, sparse autoencoders for polysemantic channels.

Run scripts from `skin/`: `uv run python mech-interp/<script>.py`.
