# Installation and computational requirements

## Supported environment

**TODO:** State the exact tested container runtime, base image, Python version, ML framework, CUDA/cuDNN versions, and hardware requirements.

## Build the container

After adapting the Dockerfile and dependency specification to the existing package, the intended build command is:

```bash
docker build -t spatial-support-vmb:0.1.0 .
```

For Podman, the corresponding command is typically:

```bash
podman build -t spatial-support-vmb:0.1.0 .
```

These commands are templates; verify them after the Dockerfile has been completed.

## Run a command

Document the tested command for the synthetic smoke test and for the full research workflow. For GPU execution, document the runtime-specific GPU access options and the required host driver. Do not assume Docker and Podman GPU options are identical across installations.

## Resource requirements

Document separately:
- minimum resources for the smoke test;
- recommended resources for training;
- GPU memory and system RAM for the full experiments;
- expected runtime, if measured;
- required disk space for data, checkpoints, and outputs.

## Troubleshooting

Add verified solutions for dependency installation, data paths, GPU visibility, and common runtime errors.
