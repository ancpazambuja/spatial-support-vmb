# Experimental protocol and reproduction guide

This document should be the authoritative description of how the paper's results can be reproduced.

## Experimental design

Document the comparison among:
- convolutional kernel-size scaling;
- dilated convolutions;
- Transformer-based self-attention.

Record the common U-Net design choices and all architecture-specific differences.

## Data

Describe both research datasets, their provenance, acquisition/simulation settings, preprocessing, normalization, train/validation/test splits, and data-access procedure. Do not expose restricted data or credentials.

## Hyperparameter optimization

Document the search space, search procedure, selection metric, search budget, split protocol, and the selected configuration for each experiment.

## Training and evaluation

Record optimizer and settings, dropout, batch size, epoch limits, early stopping, seeds, checkpoint-selection rule, and all evaluation metrics.

## Statistical analysis

Document the repeated evaluation protocol, confidence-interval method, pairwise comparisons, correction for multiple comparisons, and how tables and figures are generated.

## Commands and expected artifacts

**TODO:** Provide tested commands for each stage and list the expected output files. Identify which stages require restricted data, GPU resources, or long execution times.

## Reproduction limitations

State which results can be reproduced directly, which require authorized data access, and which may vary with hardware/software versions. Do not claim exact reproduction unless it has been verified.
