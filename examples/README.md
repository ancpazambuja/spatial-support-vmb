# Examples

## Quickstart (smoke test)

Run a miniature grid search on the small data subset shipped in `data/shallower/`:

```bash
bash examples/quickstart.sh
```

What it does:

1. Creates a temporary working directory and points `./data/` at the included subset;
2. Runs the real package entry point (`src/apps/grid_search_parallel.py`) with
   `--model_arch=unet`, 1 epoch, 5 samples, cross-validation `--cv=2`, CPU only;
3. Logs everything to a **local file-based MLflow** store (no server needed);
4. Asserts that MLflow runs were recorded.

Requirements: the package installed (`pip install -e .`) or run from the
repository root with `PYTHONPATH` (the script sets it), CPU only, a few
minutes. No restricted research data and no MLflow server are required.

**This is a software smoke test.** It validates that the package installs and
runs end-to-end (data loading → preprocessing → training → MLflow logging).
It does **not** reproduce the paper's scientific results.

Expected output: `SUCCESS` at the end, plus the workdir path where
`mlruns/` (the local MLflow store) is kept for inspection.

**Verified configuration** (2026-11, inside the reference container
`dt2_v4.sif` on a CPU-only context): 13 MLflow runs recorded with training
parameters (`batch_size`, `device`, `folds`, `lr`, `max_epochs`) and metrics
(`mean_valid_score`, `std_valid_score`). Wall time: a few minutes with
`OMP_NUM_THREADS=4`.
