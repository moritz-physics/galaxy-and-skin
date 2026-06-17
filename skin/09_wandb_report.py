"""Create a Weights & Biases Report that stitches the whole story together:
the backfilled run comparison, the hyperparameter sweep, and the held-out
evaluation — one shareable narrative document on wandb.ai.

A W&B *Report* is a living web page that pulls live panels from your runs (so the
charts stay interactive and update if you add runs), unlike a static export. This
is item #7 of the "what to learn next" list.

Run:  cd skin && uv run wandb login   # once
      uv run python 09_wandb_report.py
It prints the report URL.
"""

from __future__ import annotations

import wandb_workspaces.reports.v2 as wr

ENTITY = "m-heidtmann-ludwig-maximilian-university-of-munich"
PROJECT = "skin-lesion"
SWEEP = "ukf4v67c"


def main() -> None:
    all_runs = wr.Runset(entity=ENTITY, project=PROJECT, name="All runs")
    sweep_runs = wr.Runset(
        entity=ENTITY, project=PROJECT, name="Sweep ukf4v67c",
        filters=f'sweep == "{SWEEP}"',
    )

    blocks = [
        wr.H1("Skin lesion classifier — sweep, training & calibrated evaluation"),
        wr.MarkdownBlock(
            "A transfer-learning model (EfficientNet) classifies dermatoscopic "
            "images into seven HAM10000 lesion types. The interesting question is "
            "**not raw accuracy** but whether the model's confidence can be trusted "
            "and how it behaves clinically. This report walks from hyperparameter "
            "search to a calibrated, clinically-honest evaluation.\n\n"
            "> Not medical advice — research/education demo."
        ),

        wr.H2("1. Run comparison"),
        wr.MarkdownBlock(
            "Every training run logs validation balanced accuracy and train loss. "
            "The archived **EfficientNet-B3** remains the strongest model "
            "(≈0.955 val balanced accuracy)."
        ),
        wr.PanelGrid(
            runsets=[all_runs],
            panels=[
                wr.LinePlot(title="Validation balanced accuracy", x="Step", y=["val_bal_acc"]),
                wr.LinePlot(title="Training loss", x="Step", y=["train_loss"]),
            ],
        ),

        wr.H2("2. Hyperparameter sweep (Bayesian + Hyperband)"),
        wr.MarkdownBlock(
            "A W&B Bayesian sweep over a cheap proxy (EfficientNet-B3 @ 224px, "
            "capped data) ranked the hyperparameters. **Fine-tune learning rate "
            "dominates**; a lower head LR and logit-adjustment `tau` 1.0–1.5 help; "
            "`weight_decay` and `crt_epochs` barely matter. The parameter-importance "
            "panel below is the payoff a plain training loop cannot give you."
        ),
        wr.PanelGrid(
            runsets=[sweep_runs],
            panels=[
                wr.ParameterImportancePlot(with_respect_to="best_val_bal_acc"),
                wr.ParallelCoordinatesPlot(
                    columns=[
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("ft_lr"), log=True),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("head_lr"), log=True),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("logit_adjust_tau")),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.Config("batch_size")),
                        wr.ParallelCoordinatesPlotColumn(metric=wr.SummaryMetric("best_val_bal_acc")),
                    ]
                ),
                wr.ScalarChart(title="Best val balanced accuracy", metric="best_val_bal_acc"),
            ],
        ),

        wr.H2("3. Held-out evaluation & calibration"),
        wr.MarkdownBlock(
            "Beyond accuracy: a bootstrap 95% CI on balanced accuracy, temperature "
            "scaling for calibration, an interactive prediction table, and a "
            "clinically-honest read (melanoma sensitivity, refer-vs-reassure). "
            "Headline (EfficientNet-B3, the best model): **balanced accuracy hides "
            "a low melanoma sensitivity at the default decision rule** — the scores "
            "carry the signal (melanoma AUC ≈0.98) but argmax catches only 45% of "
            "melanomas; a tuned threshold recovers ≈91% sensitivity at ≈93% "
            "specificity. See `CLINICAL_REPORT.md` in the repo for the figures."
        ),
        wr.PanelGrid(
            runsets=[all_runs],
            panels=[wr.RunComparer(diff_only=True)],
        ),

        wr.H2("4. Takeaways"),
        wr.MarkdownBlock(
            "- The model is strong but **not clinically tuned**: at argmax it misses "
            "55% of melanomas; threshold tuning to high sensitivity is essential.\n"
            "- It is **well-calibrated** (ECE ≈ 0.05); temperature scaling trimmed "
            "held-out ECE slightly (0.054 → 0.040).\n"
            "- The sweep says **fine-tune LR is the lever**; everything else is "
            "second-order."
        ),
    ]

    report = wr.Report(
        entity=ENTITY, project=PROJECT, width="fluid",
        title="Skin Lesion UQ — sweep, training & calibrated evaluation",
        description="From hyperparameter search to clinically-honest evaluation.",
        blocks=blocks,
    )
    report.save()
    print("Report saved:", report.url)


if __name__ == "__main__":
    main()
