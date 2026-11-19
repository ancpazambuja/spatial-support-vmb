# User guide

## Scope

This guide describes how to run the software associated with the paper
*"On the Calibration of Spatial Support in Deep Learning-Assisted Velocity
Model Building"*: hyper-parameter grid search, final training (refit), and
evaluation of deep-learning models that predict seismic velocity updates.

## Installation

See [Installation](installation.md). In short: build the container
(`configs/Dockerfile`) or install the package locally (`pip install -e .`).
The container reproduces the paper's environment exactly (validated:
`pip freeze` matches the original research image, 292/292 packages).

## Inputs

### Datasets

The pipeline resolves datasets **relative to the working directory**:
commands must run from a directory containing `./data/`.

Two dataset layouts are supported (selected with `--dataset_type`):

| `--dataset_type` | Layout (original dataset) | Included subset |
|---|---|---|
| `shallow` | `<data>/DadosRede/inputs{1,2,3}_*.npy` + `outputs{1,2,3}_*.npy` | `data/shallower/` (subset of the *S2* dataset) |
| `deep` | `<data>/input_suave{,2,3}_*.npy`, `model_suave{,2,3}_*.npy`, `mask_agua_*.npy` | `data/deeper/` (subset of *D_deep*) |

- `inputs1/2/3` are the same scenes at **high/medium/low** input resolution
  (selected with `--dataset_resolution`); `outputs1/2/3` are the
  corresponding target velocity models.
- Array shapes in the included `shallow` subset: input `(351, 901, 15)`
  float32 per sample (z, x, channels); models share the spatial shape.
- The **full research datasets are restricted** — see
  [Data availability](data_availability.md). The included subsets are enough
  for the quickstart and for software validation only.

### experiment tracking

All runs are logged to [MLflow](https://mlflow.org). The command line accepts
`--experiment_tracking_uri`:

- `file://<path>` — local file store, no server (used by every script in this
  repository by default);
- `http://<server>` — the original research tracking server (set `TRACKING_URI`
  in the scripts).

## Outputs

- **MLflow runs**: parameters, metrics (`mean_valid_score`, `std_valid_score`,
  training losses), and artifacts, under the tracking URI.
- **Console logs**: `<label_id>.out` and `<label_id>.err` next to the run
  directory (convention inherited from the original experiments).
- **Checkpoints/models**: logged as MLflow artifacts during refit.
- Grid-search results are analysed afterwards by
  `src/apps/utilities/analyse_evaluation_results.py` (DataFrames from MLflow
  runs) and rendered to figures/LaTeX by the `src/pdf/` builders.

## Options (main entry points)

### `src/apps/grid_search_parallel.py`

| Argument | Default | Meaning |
|---|---|---|
| `--label_id` | — | Run label; also names the `.out/.err` logs |
| `--dataset_type` | — | `shallow` or `deep` |
| `--dataset_resolution` | — | `high`, `medium` or `low` |
| `--dataset_max_samples` | `-1` | Limit the number of samples (`-1` = all) |
| `--test_perc` | `0.1` | Holdout fraction (scripts in this repo use `0.5`) |
| `--cv` | `5` | Cross-validation folds for the grid search |
| `--model_arch` | — | `unet` or `transunet` |
| `--conv_filter_size_x/z` | — | Convolution kernel size (the *spatial support* under study) |
| `--conv_dilation_size_x/z` | `1,1` | Dilation (the alternative spatial-support mechanism) |
| `--use_transformers` | — | Transformer self-attention in the latent space (transunet) |
| `--unembed_type` | — | `none`, `paper` or `linear` |
| `--max_epochs` | — | Training epochs per grid element |
| `--seed` | `-1` | Random seed (`-1` = non-deterministic); scripts use `0` |
| `--device` | — | `cpu`, `cuda:N` or `cuda:all` (one task per GPU) |
| `--experiment_tracking_uri/name` | internal server | MLflow destination |

### `src/apps/refit.py`

Same data/model arguments, plus:

| Argument | Meaning |
|---|---|
| `--refit` | Required flag (double check that this is a final training) |
| `--gs_best_run_id` | MLflow run id of the best grid-search configuration |
| `--att_num_layers` | Transformer depth (transunet only) |
| `--one_step_training` | Single-phase training (required for unet) |

## Typical workflows

All verified commands live in the repository scripts; the shell variables
provide the knobs.

1. **Smoke test (no data, no server, CPU):**
   ```bash
   bash examples/quickstart.sh
   ```
2. **Grid search** (mirrors the original `gs_*.sh` experiments):
   ```bash
   # research configuration (GPU, restricted data, MLflow server):
   DEVICE=cuda:all MAX_EPOCHS=110 FILTER_SIZE=13 MODEL_ARCH=transunet \
   TRACKING_URI=http://<server>/ bash scripts/run_grid_search.sh
   ```
3. **Refit of the best configuration:**
   ```bash
   GS_BEST_RUN_ID=<run-id> DEVICE=cuda:all bash scripts/run_refit.sh
   ```
4. **Analysis and report generation** — from MLflow runs to DataFrames,
   figures and LaTeX sections:
   ```bash
   # TODO: verified commands for the analysis/views stage
   ```

## Known limitations

- The complete three-iteration deep-tomography **inference** pipeline
  (RTM + network inference) is work in progress: `src/inference/` is marked
  `WORK IN PROGRESS / TODO` and is not part of the reproducible pipeline yet.
- Results may vary across GPU models/drivers; the paper's experiments ran on
  NVIDIA A100-SXM4-80GB GPUs. Bitwise reproduction across hardware is not
  guaranteed.
