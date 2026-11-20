# Experimental protocol and reproduction guide

This document describes how the paper's results were produced and how they
can be reproduced. Commands marked **verified** were executed against the
included data; the full-scale experiments require the restricted research
datasets (see [Data availability](data_availability.md)).

## Experimental design

The study evaluates **spatial-information aggregation mechanisms** (the
*spatial support*) for velocity-update prediction at the **third iteration**
of the Deep-Tomography workflow, on a shared U-Net-style encoder–decoder
backbone (five encoder–decoder levels in every configuration):

1. **Convolutional kernel-size scaling** — `--conv_filter_size_x/z` in
   {3, 5, 7, 9, 13} (grid-search directories `gs_*_3x3` … `gs_*_13x13`);
2. **Dilated convolutions** — `--conv_dilation_size_x/z` with dilation rates
   r ∈ {1, 2, 3}, where r = 1 is the standard-convolution baseline;
3. **Transformer self-attention** — `--model_arch transunet`
   (`--use_transformers true`), which augments the U-Net backbone with a
   transformer pathway while retaining the kernel-size and dilation settings
   of the corresponding U-Net configuration (attention depth tuned via
   `--att_num_layers`).

All three share the same U-Net skeleton, losses and optimizers, so the
comparison isolates the spatial-support mechanism.

## Data

Two synthetic datasets of supervised pairs (migrated subsurface-offset
gathers X → velocity update Y = Δv), each with N = 1000 examples:

| Name | Origin | Included subset |
|---|---|---|
| *shallower* (𝒟s2 in the paper) | introduced in Muller et al. (2023), "Deep-tomography: Iterative Velocity Model Building with Deep Learning"; Marmousi-like geology, 9 km × 3.5 km domain | `data/shallower/` |
| *deeper* (𝒟deep) | same number of examples, similar horizontal extent, nearly twice the depth | `data/deeper/` |

Acquisition (both datasets): 20 Hz Ricker wavelet; 225 shots; 701 surface
receivers at 10 m spacing; maximum offset 3.5 km; 3.5 s recordings at 4 ms
sampling. For each velocity model, RTM generates **15 common
subsurface-offset gather panels** that serve as the input channels (X shape
`(351, 901, 15)` in the included *shallower* subset). Inputs are clipped at
the 96th percentile and normalized prior to training; targets are converted
from m/s to km/s. The **full** datasets are restricted; access is described
in [Data availability](data_availability.md).

## Hyper-parameter optimization

The evaluation protocol is two-stage (Figure 5 in the paper): independent
hyper-parameter optimization (HPO) on a dedicated subset, followed by
refitting and evaluation on the full dataset with the selected
hyperparameters (next section).

- HPO subset: 500 examples per dataset (50%; `--dataset_max_samples=500`).
- Framework: scikit-learn `GridSearchCV` driven by
  `src/apps/grid_search_parallel.py` (one process per GPU via
  `--device cuda:all`).
- Searches: one **5-fold cross-validation** grid search per (architecture,
  kernel size, dilation rate) — 30 searches per dataset for the full design
  space (2 architectures × k ∈ {3, 5, 7, 9, 13} × r ∈ {1, 2, 3}).
- Search space: learning rate, batch size, and weight decay; for TransUNet,
  the number of transformer layers (`--att_num_layers`) is optimized as an
  additional hyper-parameter. Dropout is fixed at 0.2, the optimizer is
  fixed to Adam, and the training budget is 300 epochs.
- Tracking: every configuration logged as an MLflow run (experiment
  `deep_tomography_2.0` in the original runs; `--experiment_tracking_uri`).
- Selection: best configuration per (architecture, kernel, dilation) by the
  cross-validation score, extracted by `analyse_evaluation_results.py` into
  `best_model` / `best_kernel` tables.

Command (verified shape; full data required for the paper's numbers):

```bash
DEVICE=cuda:all MAX_EPOCHS=300 FILTER_SIZE=13 MODEL_ARCH=transunet \
    bash scripts/run_grid_search.sh
```

## Training and evaluation (MCCV refit)

- Generalization performance is estimated with **Monte Carlo
  Cross-Validation (MCCV)**: J = 10 resampled runs per selected
  configuration. At each run, the full dataset is randomly partitioned into
  90% training / 10% held-out test; within the training part, a further
  80/20 split defines the validation subset used for early stopping.
- MCCV seeds: **7, 79, 181, 307, 433, 571, 701, 853, 997, 1129** — each seed
  controls all sources of stochasticity (data partitioning, network
  initialization, mini-batch ordering, and dropout masks). Runs are logged
  as `dt.<resolution>res.refit.seed.<seed>`.
- Refitting budget: maximum 500 epochs with an early-stopping patience of
  300 epochs.

  ```bash
  GS_BEST_RUN_ID=<best-run-id> SEED=7 DEVICE=cuda:all \
      bash scripts/run_refit.sh
  ```

- Metrics: MSE (primary), MAE and SSIM, computed per example on the
  held-out test set of each split and summarized across the 10 MCCV runs.
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
- Confidence intervals: **Nadeau–Bengio corrected variance** with Student-t
  intervals (J − 1 = 9 degrees of freedom) over the MCCV splits.
- Multiple comparisons: **Bonferroni correction** over the family of
  pairwise comparisons against the best configuration (m = 29 on 𝒟s2,
  m = 16 on 𝒟deep); a configuration is a statistical tie when the corrected
  interval for the mean paired difference includes zero.
- Generalization to 𝒟deep: compute-budget constraints restricted the MCCV
  evaluation to a **17-configuration subset** of the design space (k = 9
  excluded; r ≥ 2 evaluated only up to k = 7; r = 3 evaluated for U-Net
  only).

```bash
# TODO: verified commands for the analysis/views stage
```

## Environment

- Reconstructed container: `configs/Dockerfile` (provenance and validation
  in the file header: **pip freeze matches the original research image,
  292/292 packages**). Build and convert:

  ```bash
  # restore the offline wheel cache (~4 GB, stored outside the repository):
  rsync -a <wheel-cache-dir>/ .wheels/
  ( cd .wheels && sha256sum -c SHA256SUMS --quiet )

  podman build -f configs/Dockerfile -t dt2_v4_reconstructed .
  podman save -o dt2_v4_reconstructed.tar dt2_v4_reconstructed
  singularity build dt2_v4_reconstructed.sif docker-archive:///tmp/../dt2_v4_reconstructed.tar
  ```

- Original runs: NVIDIA A100-SXM4-80GB (4×), CUDA 12.1.1, cuDNN 8.9.0,
  CPython 3.11.9, PyTorch 2.3.0+cu121.

## Reproduction limitations

- The full research datasets are **restricted**; the included subsets
  validate the software but not the paper's numbers.
- Following the paper's protocol, HPO uses a dedicated 50% subset while
  refitting/evaluation covers the full dataset; some examples used during
  HPO may therefore appear in MCCV test sets — a pragmatic trade-off
  between computational feasibility and statistical isolation, documented
  in the paper.
- The three-iteration inference pipeline (RTM + network) is work in progress
  (`src/inference/`) and therefore out of scope for reproduction.
- Exact bitwise reproduction across GPU models/drivers is not guaranteed;
  the container pins the software stack, not the hardware behavior.
- The repository is the research codebase and also contains exploratory
  variants beyond the paper's protocol (e.g., AdamW grid-search classes in
  `gs_grid.py`). The protocol described here — fixed Adam optimizer,
  searched learning rate / batch size / weight decay, and the epoch budgets
  above — is authoritative for reproducing the paper's results.
