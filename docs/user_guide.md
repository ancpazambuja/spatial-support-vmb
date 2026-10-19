# User guide

## Scope

This guide will describe how to run the software associated with the paper. Replace the TODOs with commands and parameter names from the actual package.

## Inputs

Document:
- input data format, tensor dimensions, channel order, units, and coordinate conventions;
- preprocessing and normalization;
- required metadata;
- dataset configuration and split definition;
- any expected file naming or directory conventions.

## Outputs

Document:
- prediction format and units;
- model checkpoints and their selection rule;
- evaluation metrics and aggregation;
- logs, figures, tables, and output directory layout.

## Options

Document every supported CLI argument and configuration field, including defaults and valid ranges. Explain behavior for invalid or missing inputs.

## Typical workflows

1. Run the synthetic smoke test.
2. Run inference with a compatible trained checkpoint, if one is available.
3. Train a model using an authorized dataset.
4. Evaluate predictions and generate the reported analysis.

Add the exact verified commands for each workflow.
