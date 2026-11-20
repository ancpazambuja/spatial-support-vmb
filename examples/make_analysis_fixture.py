#!/usr/bin/env python
"""Generate a synthetic MLflow file store for smoke-testing the analysis stage.

The analysis scripts (`src/apps/utilities/analyse_evaluation_results.py` and
the `src/pdf/` report builders) expect the paper's experiment layout:

  * one MLflow experiment per (kernel size x optimizer), all sharing a name
    prefix; the analysis asserts exactly 10 experiments matching the prefix,
    and the runs of ONE dataset per invocation (shallow OR deep),
  * final-training runs named
      ``REFIT__<label>_<data>-data_<res>-res_<ns>-samples_<arch>-arch_<ts>``
    where ``<label>`` encodes the architecture and kernel size
    (e.g. ``UNET_3x3`` / ``TRANSUNET_5x5``),
  * runs grouped per architecture in "MCCV sessions" of 8-13 executions
    (the 10 MCCV seeds) executed close in time — the analysis discards
    smaller groups,
  * per-run parameters (seed, optimizer class repr, kernel sizes, attention
    configuration, train/test indexes, ...) and summary metrics
    (``<m>_mean/std/min/max_test_set``),
  * per-example metric histories ``<m>_values`` (one step per test example).

This script fabricates a small store with exactly that structure so the
analysis stage can be exercised without the restricted research datasets.
It does NOT reproduce the paper's numbers.

Usage (one invocation per dataset, mirroring the paper's workflow):
    python examples/make_analysis_fixture.py --out ./mlruns-analysis-fixture
"""

import argparse
import time

import numpy as np
from ml_collections import ConfigDict
from mlflow import MlflowClient

KERNELS = [3, 5, 7, 9, 13]
OPTIMIZERS = {
    "Adam": "<class 'torch.optim.adam.Adam'>",
    "AdamW": "<class 'torch.optim.adamw.AdamW'>",
}
ARCHS = {"UNET": "False", "TRANSUNET": "True"}
DATASETS = ["shallow", "deep"]
ATTENTION_CONFIG = {
    "hidden_size": 768,
    "transformer": {
        "num_heads": 12,
        "num_layers": 2,
        "attention_dropout_rate": 0.0,
        "dropout_rate": 0.0,
        "mlp_dim": 3072,
    },
}
# Plausible per-dataset metric baselines (NOT the paper's numbers).
BASELINES = {
    "shallow": {"mse": 0.032, "mae": 0.092, "ssim": 0.69},
    "deep": {"mse": 0.0075, "mae": 0.034, "ssim": 0.80},
}
METRICS = ["mse", "mae", "ssim"]


def make_run_name(arch, k, data, seed, ts):
    # label encodes arch + kernel; matches the analysis filters
    # (e.g. ILIKE '%\_UNET%3x3%high-res%').
    label = f"{arch}_{k}x{k}"
    return (f"REFIT__{label}_{data}-data_high-res_full-samples_"
            f"{arch.lower()}-arch_{ts}_{seed}")


def make_params(arch, k, opt_key, seed, n_test, n_train):
    return {
        "_x_seed": str(seed),
        "batch_size": "16",
        "max_epochs": "500",
        "lr": "0.0005",
        "optimizer__weight_decay": "0",
        "optimizer": OPTIMIZERS[opt_key],
        "module__conv_kernel_size": f"({k}, {k})",
        "module__conv_dilation_size": "(1, 1)",
        "module__transposed_conv_kernel_size": "(2, 2)",
        "module__in_channels": "15",
        "module__n_filters_init": "16",
        "module__attention_in_channels": "512",
        "module__use_transformers": ARCHS[arch],
        "module__attention_img_size": "(11, 28)",
        # str(ConfigDict) is the exact format the analysis stage parses
        # (MLFlowUtils.string_to_dict).
        "module__attention_config": str(ConfigDict(ATTENTION_CONFIG)),
        # str(numpy_array) format (space-separated) is what parse_string
        # expects in the analysis stage.
        "_x_test_indexes": str(np.array(range(n_test))),
        "_x_train_indexes": str(np.array(range(n_train))),
    }


def make_metric_set(rng, data, n_examples):
    """Per-example vectors + summary statistics for one run."""
    values = {}
    for m in METRICS:
        base = BASELINES[data][m]
        # heterogeneous per-example difficulty (log-normal spread)
        noise = rng.lognormal(mean=0.0, sigma=0.35, size=n_examples)
        v = np.clip(base * noise, 0.0, None) if m != "ssim" else \
            np.clip(base * noise, 0.0, 1.0)
        values[m] = v
    out = {}
    for m in METRICS:
        v = values[m]
        out[f"{m}_mean_test_set"] = float(v.mean())
        out[f"{m}_std_test_set"] = float(v.std())
        out[f"{m}_min_test_set"] = float(v.min())
        out[f"{m}_max_test_set"] = float(v.max())
    return out, values


def generate_dataset_store(client, prefix, data, seeds, n_test, n_train):
    """One dataset per prefix: 10 experiments (5 kernels x 2 optimizers)."""
    run_count = 0
    for k in KERNELS:
        for opt_name, opt_repr in OPTIMIZERS.items():
            exp_name = f"{prefix}_{k}x{k}_{opt_name}"
            exp_id = client.create_experiment(exp_name)
            for arch in ARCHS:
                for seed in seeds:
                    ts = time.strftime("%Y-%m-%dT%H.%M.%S")
                    name = make_run_name(arch, k, data, seed, ts)
                    run = client.create_run(
                        exp_id,
                        tags={"mlflow.runName": name},
                    )
                    for p_key, p_val in make_params(
                            arch, k, opt_name, seed, n_test, n_train).items():
                        client.log_param(run.info.run_id, p_key, p_val)

                    rng = np.random.default_rng(seed * 1000 + k)
                    summary, per_example = make_metric_set(rng, data, n_test)
                    for m_key, m_val in summary.items():
                        client.log_metric(run.info.run_id, m_key, m_val)
                    for m in METRICS:
                        for step, val in enumerate(per_example[m]):
                            client.log_metric(run.info.run_id,
                                              f"{m}_values", float(val),
                                              step=step)
                    client.set_terminated(run.info.run_id)
                    run_count += 1
    return run_count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="./mlruns-analysis-fixture",
                        help="output directory for the file-based MLflow store")
    parser.add_argument("--prefix", default="dtfix",
                        help="experiment name prefix; one experiment family is "
                             "created per dataset as <prefix>-<dataset>")
    parser.add_argument("--seeds", default="7,79,181,307,433,571,701,853,997,1129",
                        help="MCCV seeds to simulate (10, matching the paper)")
    parser.add_argument("--n-examples", type=int, default=20,
                        help="number of test examples per run")
    args = parser.parse_args()

    seeds = [int(s) for s in args.seeds.split(",")]
    n_train, n_test = 30, args.n_examples
    client = MlflowClient(tracking_uri=f"file://{args.out}")

    for data in DATASETS:
        prefix = f"{args.prefix}-{data}"
        count = generate_dataset_store(client, prefix, data, seeds,
                                       n_test, n_train)
        print(f"[{data}] experiments: {len(KERNELS) * len(OPTIMIZERS)} "
              f"(prefix '{prefix}'), runs: {count}")

    print(f"fixture store created: {args.out}")
    print("next: run the analysis once per dataset, e.g.")
    print(f"  --experiment_tracking_uri=file://<absolute-path-to-{args.out}>")
    print(f"  --experiment_prefix={args.prefix}-shallow  (then -deep)")


if __name__ == "__main__":
    main()
