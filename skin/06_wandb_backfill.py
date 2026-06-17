"""Backfill already-trained runs into Weights & Biases.

You trained these models *before* adding W&B, so there is no live data in W&B for
them. But each run saved a `training_log.json` with its per-epoch history, so we
can "replay" that history into W&B after the fact and get the same dashboards we
would have gotten live -- curves, run comparison, config table -- without
retraining anything.

This is a great first hands-on W&B exercise because it is pure logging: no GPU,
no Kaggle, no model loading. Just read JSON and call wandb.log().

Run it (from the repo root or the skin/ dir):

    # one-time: log in with your W&B API key from https://wandb.ai/authorize
    uv run wandb login

    # then replay all three runs into a project called "skin-lesion"
    uv run python skin/06_wandb_backfill.py

Open the URL it prints. You will see three runs you can overlay and compare.
"""

from __future__ import annotations

import json
from pathlib import Path

import wandb

# Each tuple is (folder, friendly run name). The folder must contain a
# training_log.json. Add or remove lines here to control what gets uploaded.
RESULTS = Path(__file__).parent / "results"
RUNS = [
    (RESULTS / "archive_efficientnet_b3", "efficientnet_b3"),
    (RESULTS / "archive_efficientnet_v2_s_baseline", "efficientnet_v2_s_baseline"),
    (RESULTS, "efficientnet_v2_s_current"),
]

PROJECT = "skin-lesion"


def backfill_one(folder: Path, run_name: str) -> None:
    log = json.loads((folder / "training_log.json").read_text())
    history = log["history"]

    # Everything that is constant for the run goes in `config`. W&B shows this as
    # a sortable/filterable column in the runs table -- this is how you later say
    # "show me every run with img_size=384" or sort runs by model.
    run = wandb.init(
        project=PROJECT,
        name=run_name,
        config={
            "model": log["model"],
            "img_size": log["img_size"],
            "num_classes": len(log["classes"]),
            "best_val_bal_acc": log["best_val_bal_acc"],
            "source": "backfilled from training_log.json",
        },
        reinit=True,  # allow several wandb.init() calls in one script
    )

    # Per-epoch metrics go in wandb.log(). The `step` is the x-axis. The three
    # training phases (head -> ft -> crt) each restart their own epoch counter at
    # 0, so we use a single monotonically increasing `global_step` to lay them out
    # left-to-right on one curve, and also keep `phase` so you can colour by it.
    for global_step, row in enumerate(history):
        run.log(
            {
                "train_loss": row["loss"],
                "val_bal_acc": row["val_bal_acc"],
                "phase": row["phase"],
                "phase_epoch": row["epoch"],
            },
            step=global_step,
        )

    # A "summary" value is the single headline number for the run (shown in the
    # runs table, used to rank runs). Set it explicitly so it is the *best* val
    # accuracy, not just whatever the last epoch happened to be.
    run.summary["best_val_bal_acc"] = log["best_val_bal_acc"]
    run.finish()
    print(f"  uploaded {run_name}: {len(history)} epochs, best {log['best_val_bal_acc']:.3f}")


def main() -> None:
    for folder, run_name in RUNS:
        if not (folder / "training_log.json").exists():
            print(f"  skip {run_name}: no training_log.json in {folder}")
            continue
        backfill_one(folder, run_name)
    print("\nDone. Open the project URL above to compare the runs.")


if __name__ == "__main__":
    main()
