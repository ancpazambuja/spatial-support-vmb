# Installation and Computational Requirements

## Supported environment

- Linux x86_64 with an NVIDIA GPU (CUDA 12.1, cuDNN 8) for training and full evaluation.
- A CPU-only environment is sufficient for the quickstart smoke test.
- Python 3.11 (the version shipped by the container image below is CPython
  3.11.9; local installations support 3.10–3.12).

## Option A — container (recommended)

Build the image from the **repository root** (the build context must be the repo root, not the `configs/` directory). The offline wheel cache lives **outside the repository** (shared scratch on `harpia-dev2`); copy it into the build context and verify it before building:

```bash
cd spatial-support-vmb

# Restore the offline wheel cache (4 GB) and verify its integrity:
rsync -a /scr01/ntbs/spatial-support-vmb-wheels/ .wheels/
( cd .wheels && sha256sum -c SHA256SUMS --quiet && echo "wheels OK" )

podman build -f configs/Dockerfile -t spatial-support-vmb:0.1.0 .
# or, with Docker:
docker build -f configs/Dockerfile -t spatial-support-vmb:0.1.0 .
```

Run an interactive shell in the container:

```bash
# Docker (NVIDIA Container Toolkit required):
docker run --rm -it --gpus all spatial-support-vmb:0.1.0

# Podman (requires nvidia-container-toolkit / CDI configured):
podman run --rm -it --security-opt label=disable --device nvidia.com/gpu=all spatial-support-vmb:0.1.0
```

The image contains the complete pinned Python environment (including the LaTeX toolchain). The source code and data are **not baked into the image**; they are provided through bind mounts, e.g.:

```bash
singularity run --nv -e --bind /path/to/repo:/workspace dt2_v4_reconstructed.sif bash -i
```

## Option B — local installation (without a container)

Requires Python 3.10–3.12 and a working CUDA 12.1 stack for GPU execution.

```bash
cd spatial-support-vmb
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e .

# For development (tests):
pip install -e ".[dev]"
```

## Dependency pinning

Two files describe the Python environment:

- `requirements.txt` — the **direct dependencies** imported by the source code, pinned to the versions used for the paper. Intended for lightweight local installations.
- `requirements-frozen.txt` — the **complete frozen environment** (292 packages, from `pip freeze`) of the research container used for the paper. This is what `configs/Dockerfile` installs.

The container image is the **exact environment of the paper** (reconstructed from the original research image `dt2_v4.sif`; see the provenance header in `configs/Dockerfile`): Ubuntu 22.04, CUDA 12.1.1, cuDNN 8.9.0, CPython 3.11.9, and the frozen package list, including the LaTeX toolchain required by the report-generation stage (`src/pdf/`).

A successful image build alone does not guarantee bitwise-identical results across hardware; see the [reproducibility checklist](reproducibility_checklist.md).

<!-- Build verified: the image was rebuilt from this Containerfile and validated
against the original dt2_v4.sif (pip freeze identical, 292/292 pins, 2026-11-19).
The docker/podman run invocations above follow the standard NVIDIA Container
Toolkit patterns; the GPU runtime used for the paper was Apptainer/Singularity
(see docs/experimental_protocol.md). -->
