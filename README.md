# galaxy-uq

A study of **calibrated uncertainty under perturbations** for galaxy morphology
classification. Three classifiers are compared on the Galaxy10 DECaLS dataset
(17,736 RGB images, 256×256, 10 morphology classes), with the focus on how well
each model's predicted probabilities remain calibrated when the inputs are
corrupted. Cross-validation, hyperparameter tuning, and evaluation metrics are
all implemented from scratch.

## Setup

This project uses [uv](https://docs.astral.sh/uv/) with Python 3.11. Install
the dependencies into a virtual environment:

```bash
uv sync
```

### Download the data

The dataset is not tracked in git. Download the Galaxy10 DECaLS HDF5 file into
`data/raw/` and verify its checksum:

```bash
mkdir -p data/raw
curl -L -o data/raw/Galaxy10_DECals.h5 \
  https://astro.utoronto.ca/~hleung/shared/Galaxy10/Galaxy10_DECals.h5

# Expected SHA256:
# 19aefc477c41bb7f77ff07599a6b82a038dc042f889a111b0d4d98bb755c1571
shasum -a 256 data/raw/Galaxy10_DECals.h5
```

The file is ~2.54 GB.

## Reproducing task 1 (EDA)

Run the exploratory data analysis from the project root:

```bash
uv run python scripts/01_eda.py
```

This loads the dataset, logs ground-truth integrity checks, writes five
diagnostic figures to `results/figures/`, and a JSON summary to
`results/metrics/01_eda_summary.json`. Pass `--seed` (default 0) to control
random sampling and `--out-dir` to change the figure destination.
