# Skin sub-project — change log

A running, human-readable record of *why* the skin analysis evolved, written so a
new contributor (or a fresh Claude with no chat history) can understand the intent
behind each addition without reverse-engineering the diffs. Newest first. Dates
are absolute.

---

## 2026-06-17 — Analyse the SOTA model; `SKIN_MODEL_DIR` override

The Tier-1 and clinical analyses initially ran on whatever sat in `results/`,
which was a partial **EfficientNet-V2-S** experiment (local val balanced accuracy
0.73), *not* the repo's best model. Added an optional **`SKIN_MODEL_DIR`** env var
to `04_analysis.py` and `08_clinical.py`: it points model loading at any checkpoint
folder while figures/metrics still write to the canonical `results/` location.

The canonical figures + `analysis_metrics.json` + `clinical_metrics.json` +
`CLINICAL_REPORT.md` now reflect the **state-of-the-art EfficientNet-B3 @ 300px**
(`results/archive_efficientnet_b3/`, local val balanced accuracy **0.874**), run via:

```
SKIN_MODEL_DIR=results/archive_efficientnet_b3 uv run python 04_analysis.py
SKIN_MODEL_DIR=results/archive_efficientnet_b3 uv run python 08_clinical.py
```

**Key clinical finding (and it survives the upgrade):** even B3, which ranks
melanomas almost perfectly (AUC 0.977), has an argmax melanoma sensitivity of only
**0.45** (44/100 melanomas called benign). Tuning the threshold to ≥90% sensitivity
recovers **0.91 sensitivity at 0.93 specificity** with no retraining. The lesson is
about the *decision rule*, not model capacity.

---

## 2026-06-17 — Tier-1 rigor pass on `04_analysis.py`

**Motivation.** The model was already strong (archived EfficientNet-B3 ≈ 0.955 val
balanced accuracy), so further hyperparameter chasing has diminishing returns. The
higher-value move is to make the evaluation *trustworthy* — which is the whole
premise of this repo ("can the model's confidence be trusted"). Three additions,
all in `04_analysis.py`, all offline on the existing checkpoint + `data/val`:

1. **Bootstrap 95% CI on balanced accuracy.**
   - *What:* resample the val set with replacement 2000× and recompute balanced
     accuracy to get a percentile confidence interval.
   - *Why:* a single val split yields one point estimate. Without a CI you can't
     tell whether a 0.5% gap between two models (or two sweep trials) is real or
     sampling noise. Overlapping CIs ⇒ not meaningfully different.
   - *Where:* `bootstrap_balanced_accuracy_ci()`; reported in stdout and under
     `balanced_accuracy_ci95` in `results/analysis_metrics.json`.

2. **Temperature scaling (post-hoc calibration, Guo et al. 2017).**
   - *What:* fit a single scalar `T` that minimises NLL of `softmax(logits / T)`,
     then report Expected Calibration Error (ECE) before vs after.
   - *Why:* the script already measured ECE but never *fixed* miscalibration.
     Temperature scaling improves calibrated confidence without changing accuracy
     (argmax is scale-invariant) — exactly what a project about trustworthy
     probabilities should do.
   - *Honesty detail:* `T` is fit on a random **50% calibration split** of the val
     set and ECE is reported on the held-out 50%. Fitting and evaluating `T` on the
     same data understates ECE; the split avoids that.
   - *Where:* `fit_temperature()`; figure `results/figures/calibration_temperature.png`;
     numbers under `temperature_scaling` in the metrics JSON.
   - *Empirical result.* On the **B3 SOTA checkpoint** `T = 1.24` modestly helped
     (held-out ECE **0.054 → 0.040**). On the weaker V2-S checkpoint `T = 1.65`
     made it *worse* (0.081 → 0.104) — the V2-S model is already near-calibrated
     and its held-out half is noisy, so there was no real miscalibration to fix.
     Either way the calibration/test split is doing its job: reporting the true
     held-out effect rather than an optimistic same-data number.
   - *Note:* this is an analysis-time correction. It is **not** yet wired into the
     Gradio app (`03_app.py`); doing so would mean saving `T` alongside the
     checkpoint and dividing logits by it at inference. Left as a follow-up.

3. **Confident-and-wrong error analysis.**
   - *What:* rank wrong predictions by confidence, render the worst as a labelled
     image grid, and tally which `true → predicted` class pairs dominate the
     high-confidence (≥0.5) errors.
   - *Why:* the dangerous failures are the *confident* mistakes, not the unsure
     ones. This surfaces them for inspection (label noise? a specific class pair?)
     and is the qualitative complement to the metrics.
   - *Where:* figure `results/figures/confident_errors.png`; details under
     `confident_errors` in the metrics JSON.

**Mechanical change.** The clean inference pass now goes through
`run_inference_logits()` (returns logits, needed for temperature scaling) and
derives probabilities from those; the robustness sweep still uses `run_inference`.

---

## 2026-06-17 — Hyperparameter sweep (W&B) and clinical evaluation

- **`kaggle_sweep.ipynb`** — a Weights & Biases Bayesian sweep (method `bayes` +
  Hyperband early-termination) over a cheap proxy (EfficientNet-B3 @ 224px, capped
  data) to *rank* hyperparameters without burning Kaggle GPU quota. Results live in
  the W&B project `skin-lesion → Sweeps`; a downloaded summary is in
  `results/hpo/sweep_ukf4v67c.json` with plots alongside it.
  - **Finding:** fine-tune LR dominates (best ≈ 1–3e-4), lower head LR is better,
    logit-adjustment `tau` 1.0–1.5 beats 0, `batch_size` 32 ≥ 16, while
    `weight_decay` and `crt_epochs` barely matter at proxy scale.
  - **Applied to `kaggle_training.ipynb`:** `FT_LR=1.5e-4`, `HEAD_LR=5e-4`,
    `LOGIT_ADJUST_TAU=1.0`, `BATCH_SIZE=32`, `CRT_EPOCHS=3`, retraining
    EfficientNet-B3 @ 300px (the architecture the sweep tuned).

- **W&B live logging** added to `kaggle_training.ipynb` (guarded by `USE_WANDB`;
  key read from the Kaggle `WANDB_API_KEY` secret, never hard-coded). Backfill of
  pre-W&B runs from `training_log.json` is in `06_wandb_backfill.py`; rich offline
  evaluation (interactive prediction table, confusion matrix, PR/ROC) is in
  `07_wandb_eval.py`.

- **`08_clinical.py`** — clinically-honest evaluation: per-class sensitivity,
  melanoma-vs-rest ROC with a high-sensitivity operating point, a "refer vs
  reassure" screening trade-off, and the dangerous missed-melanoma cases. Writes
  `results/clinical_metrics.json` and `clinical_*.png`; narrative in
  `CLINICAL_REPORT.md`.
