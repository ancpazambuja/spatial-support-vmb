# Tutorials

## Quickstart: miniature grid search (verified)

**What it does:** runs the real package entry point on the small data subset
included in the repository — data loading, preprocessing, U-Net training
(1 epoch, 5 samples, cross-validation), and MLflow logging. CPU-only, no
server, a few minutes.

```bash
bash examples/quickstart.sh
```

**Prerequisites:** the package installed (`pip install -e .`, see
[Installation](installation.md)) *or* nothing at all (the script sets
`PYTHONPATH` itself); no restricted data; no MLflow server.

**What you should see:**

- `FINDING OUT DATA FORMAT` → `File shape: (351, 901, 15)` — the included
  shallow subset being loaded;
- training progress of the grid-search fits;
- at the end:

  ```
  [quickstart] SUCCESS
  [quickstart] MLflow runs recorded under: /tmp/.../mlruns
  ```

**Expected result:** 13 MLflow runs recorded locally (experiment
`quickstart`) with parameters (`batch_size`, `device`, `folds`, `lr`,
`max_epochs`) and metrics (`mean_valid_score`, `std_valid_score`).

**Inside the reference container** (as used for validation):

```bash
singularity exec --bind /path/to/repo:/workspace dt2_v4_reconstructed.sif \
    bash /workspace/examples/quickstart.sh
```

**Important:** this is a **software smoke test**. It verifies that the
package installs and runs end-to-end; it does not reproduce the paper's
scientific findings (those need the restricted datasets — see
[Data availability](data_availability.md) and the
[experimental protocol](experimental_protocol.md)).

## Next steps

- Real grid searches and refits: [scripts/](../scripts/README.md)
- Full workflow: [User guide](user_guide.md) and
  [Experimental protocol](experimental_protocol.md)
