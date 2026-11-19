# Computational environment

The exact environment of the paper. Reconstructed and **validated**: the
image built from `configs/Dockerfile` has a `pip freeze` identical to the
original research image (`dt2_v4.sif`), 292/292 packages — see the
provenance header in `configs/Dockerfile`.

| Component | Value |
|---|---|
| Base image | `nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04` |
| OS | Ubuntu 22.04 |
| Python | CPython 3.11.9 |
| Deep-learning framework | PyTorch 2.3.0+cu121 (torchvision 0.18.0) |
| CUDA / cuDNN | 12.1.1 / 8.9.0 |
| GPU (paper's runs) | NVIDIA A100-SXM4-80GB (4× per node, `--device cuda:all`) |
| Key libraries | scikit-learn 1.7.0, skorch 1.1.0, mlflow 2.21.3, numpy 1.26.4, scipy 1.12.0 |
| LaTeX toolchain | included (required by the report-generation stage, `src/pdf/`) |
| CPU-only smoke test | supported (`examples/quickstart.sh`) |

## Package lists

- `requirements.txt` — the direct dependencies, pinned to the versions above.
- `requirements-frozen.txt` — the complete frozen environment (292 packages,
  from `pip freeze` of the reference container).

## Storage

- Container image: ~9.1 GB (Apptainer SIF).
- Offline wheel cache: ~4 GB at `/scr01/ntbs/spatial-support-vmb-wheels`
  (harpia-dev2), with a `SHA256SUMS` manifest — see
  [Installation](../docs/installation.md) for the restore-and-verify step.

## Reproducibility caveats

- The container pins the **software** stack; it cannot pin hardware behavior.
  Exact bitwise reproduction across GPU models/drivers is not guaranteed.
- Residual training non-determinism is documented in the code
  (`src/apps/grid_search_parallel.py` header: thread race conditions after
  the data split).
