# Experimental protocol and reproduction guide

This document describes how the paper's results were produced and how they
can be reproduced. Commands marked **verified** were executed against the
included data; the full-scale experiments require the restricted research
datasets (see [Data availability](data_availability.md)).

## Experimental design

The study compares **spatial-information aggregation mechanisms** (the
*spatial support*) for velocity-update prediction on a shared U-Net-style
backbone:

1. **Convolutional kernel-size scaling** — `--conv_filter_size_x/z` in
   {3, 5, 7, 9, 13} (grid-search directories `gs_*_3x3` … `gs_*_13x13`);
2. **Dilated convolutions** — `--conv_dilation_size_x/z` > 1
   (r ∈ {2, 4} in the original runs);
3. **Transformer self-attention** — `--model_arch transunet`
   (`--use_transformers true`), with attention depth `--att_num_layers`.

All three share the same U-Net skeleton, losses and optimizers, so the
comparison isolates the spatial-support mechanism.

## Data

Two synthetic seismic datasets (deep-tomography pairs: input gathers →
velocity models):

| Name | Origin | Included subset |
|---|---|---|
| *shallower* | subset of the *S2* dataset from "Deep-tomography: Iterative Velocity Model Building with Deep Learning" | `data/shallower/` |
| *deeper* | subset of *D_deep* from the present paper | `data/deeper/` |

Preprocessing resamples inputs to the target resolution (high/medium/low)
and pairs each input with its target velocity model. The **full** datasets
are restricted; access is described in
[Data availability](data_availability.md).

## Hyper-parameter optimization

- Framework: scikit-learn `GridSearchCV` driven by
  `src/apps/grid_search_parallel.py` (one process per GPU via
  `--device cuda:all`).
- Search space per architecture: kernel size {3, 5, 7, 9, 13} × optimizer
  {Adam, AdamW} × learning rate / weight-decay ranges; batch size 16;
  `max_epochs` 110; validation score = mean validation MSE across CV folds.
- CV split protocol: holdout (`--test_perc`) followed by k-fold CV on the
  training part (`--cv`).
- Tracking: every configuration logged as an MLflow run (experiment
  `deep_tomography_2.0` in the original runs; `--experiment_tracking_uri`).
- Selection: best run per (architecture, kernel, dataset) by validation
  score, extracted by `analyse_evaluation_results.py` into `best_model` /
  `best_kernel` tables.

Command (verified shape; full data required for the paper's numbers):

```bash
DEVICE=cuda:all MAX_EPOCHS=110 FILTER_SIZE=13 MODEL_ARCH=transunet \
    bash scripts/run_grid_search.sh
```

## Training and evaluation

- Final training (**refit**) of each best configuration with multiple seeds
  (`SEED ∈ {0, 1, 2, ...}`), logged to
  `dt.<resolution>res.refit.seed.<seed>`:

  ```bash
  GS_BEST_RUN_ID=<best-run-id> SEED=0 DEVICE=cuda:all \
      bash scripts/run_refit.sh
  ```

- Metrics: MSE (primary), MAE and SSIM, computed per example on the held-out
  test set; aggregated as mean ± std across seeds.
- Determinism: seeds fixed via `src/utl/classes.py::utl.set_seed`
  (Python/NumPy/PyTorch/CUDA, `CUBLAS_WORKSPACE_CONFIG=:4096:8`,
  deterministic cuDNN); residual non-determinism is documented in the code
  (`grid_search_parallel.py` header: thread race conditions after the split).

## Statistical analysis and figures

- `src/apps/utilities/analyse_evaluation_results.py` builds nested
  pandas DataFrames from the MLflow runs (shallow/deep × adam/adamw), applies
  cleaning filters and injects `best_model` / `best_kernel` tables.
- `src/apps/utilities/analyse_evaluation_examples*.py` drive the `src/pdf/`
  builders (PyLaTeX) that generate the paper's figures (violin/box/scatter
  plots, per-example heatmaps) and LaTeX report sections per metric
  (`mae_metric`, `mse_metric`, `ssim_metric`).
- Confidence intervals: Student-t intervals (scipy) over seed repetitions;
  pairwise comparisons between architectures on the shared test set.

```bash
# TODO: verified commands for the analysis/views stage
```

## Environment

- Reconstructed container: `configs/Dockerfile` (provenance and validation
  in the file header: **pip freeze matches the original research image,
  292/292 packages**). Build and convert:

  ```bash
  podman build -f configs/Dockerfile -t dt2_v4_reconstructed .
  podman save -o dt2_v4_reconstructed.tar dt2_v4_reconstructed
  singularity build dt2_v4_reconstructed.sif docker-archive:///tmp/../dt2_v4_reconstructed.tar
  ```

- Original runs: NVIDIA A100-SXM4-80GB (4×), CUDA 12.1.1, cuDNN 8.9.0,
  CPython 3.11.9, PyTorch 2.3.0+cu121.

## Reproduction limitations

- The full research datasets are **restricted**; the included subsets
  validate the software but not the paper's numbers.
- The three-iteration inference pipeline (RTM + network) is work in progress
  (`src/inference/`) and therefore out of scope for reproduction.
- Exact bitwise reproduction across GPU models/drivers is not guaranteed;
  the container pins the software stack, not the hardware behavior.
