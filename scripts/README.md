# Scripts

Entry points for the reproducibility pipeline. All scripts:

- resolve datasets relative to the working directory (`./data/`), pointing it at a subset in `data/` by default (override with `DATA_SUBSET`);
- pass the MLflow tracking URI **explicitly** — default is a local file store (`./mlruns` in a per-run workdir), so no server is needed; set `TRACKING_URI` to use the original MLflow server;
- write console logs as `<label_id>.out` / `<label_id>.err` in the run directory, matching the original experiment conventions.

| Script | Purpose | Corresponds to the original |
|---|---|---|
| `run_grid_search.sh` | Hyper-parameter grid search for one architecture/kernel/dataset | `results/dataset_new/grid_search/gs_*/shallower\|deeper/gs_*.sh` |
| `run_refit.sh` | Final training of the best grid-search configuration (by MLflow run id) with a chosen seed | `results/dataset_new/training/refit_*.sh` |

## Grid search examples

```bash
# Original research configuration (GPU, full data, restricted dataset required):
DEVICE=cuda:all MAX_EPOCHS=110 FILTER_SIZE=13 MODEL_ARCH=transunet \
TRACKING_URI=http://<your-mlflow-server>/ \
    bash scripts/run_grid_search.sh

# CPU smoke run on the included subset:
MODEL_ARCH=unet DEVICE=cpu MAX_EPOCHS=1 MAX_SAMPLES=5 \
EXTRA_ARGS="--cv 2" \
    bash scripts/run_grid_search.sh
```

## Refit example

```bash
GS_BEST_RUN_ID=<best-run-mlflow-id> \
DEVICE=cuda:all SEED=0 \
    bash scripts/run_refit.sh
```

<!-- TODO: verified commands for the statistical analysis and figure/report
     generation stages (analyse_evaluation_results / analyse_evaluation_examples
     -> src/pdf builders) will be added after those stages are exercised. -->
