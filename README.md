# On the Calibration of Spatial Support in Deep Learning-Assisted Velocity Model Building

This repository contains the software and reproducibility materials for the paper:

**On the Calibration of Spatial Support in Deep Learning-Assisted Velocity Model Building**

> **Status:** Repository scaffold. Replace every `TODO` and verify every command against the actual package before making the repository public.

## Scientific overview

The study compares spatial-information aggregation mechanisms for velocity-update prediction in deep learning-assisted seismic velocity model building:

- convolutional kernel-size scaling;
- dilated convolutions;
- Transformer-based self-attention.

The experiments use two synthetic seismic datasets. Access to the original research data is restricted; see [Data availability](docs/data_availability.md).

## Repository layout

- `src/`: available original source code. 
- `configs/`: environment configuration (docker file and requirements).
- `scripts/`: entry points for training, evaluation, statistical analysis, and figure/table generation.
- `examples/`: lightweight synthetic smoke test and usage examples.
- `tests/`: automated tests.
- `docs/`: installation, user guide, experimental protocol, and data availability.
- `data/`: small dataset for testing.
- `results/`: documentation and optionally small, shareable result summaries.

Adjust this layout to fit the existing package rather than duplicating code.

## Quick start

The quickstart is intended to run without access to the restricted research datasets and without downloading trained checkpoints.

1. Install the supported container runtime (Docker, Podman, or another compatible runtime).
2. Build the image using the instructions in [Installation](docs/installation.md).
3. Run the synthetic smoke test described in [Tutorials](docs/tutorials.md).

**TODO:** Add the verified build/run commands and the actual smoke-test command after adapting this scaffold to the package API.

## Full research reproduction

The full workflow requires access to the research datasets and may require GPU resources. Read:

- [Experimental protocol](docs/experimental_protocol.md)
- [Data availability](docs/data_availability.md)
- [User guide](docs/user_guide.md)

The repository should provide the exact configurations, seeds, training/evaluation commands, and analysis scripts used to produce the paper's reported results. If any results cannot be regenerated because of data-access restrictions or computational constraints, document the limitation precisely.

## Software and computational requirements

See [Installation](docs/installation.md). The supported Python, framework, CUDA/cuDNN, GPU, memory, and container versions are **TODO: verify and document**. Do not assume that a successful image build alone guarantees bitwise-identical results across hardware.

## Tests

**TODO:** Add and run tests for model construction, expected input/output shapes, preprocessing, metric calculations, and the synthetic example.

## Citation

Citation metadata is provided in [`CITATION.cff`](CITATION.cff). Update it with the final publication details when available.

## License

The proposed license is MIT; review [LICENSE](LICENSE) and confirm that you have the right to license all included code. Third-party code, data, pretrained models, and other assets may have separate terms.
