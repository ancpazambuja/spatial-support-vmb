"""Tests for the command line interface contracts
(src/apps/command_line_interface.py).

The arguments below are the exact ones passed by examples/quickstart.sh and
scripts/run_grid_search.sh / run_refit.sh, so any change here breaks the
documented entry points.
"""

import pytest

from src.apps.command_line_interface import RefitCLI, TrainCLI

QUICKSTART_ARGS = [
    "--label_id=QUICKSTART",
    "--experiment_tracking_uri=file:///tmp/mlruns",
    "--experiment_tracking_name=quickstart",
    "--dataset_type=shallow",
    "--dataset_resolution=high",
    "--model_arch=unet",
    "--unembed_type=paper",
    "--conv_filter_size_x=3", "--conv_filter_size_z=3",
    "--max_epochs=1",
    "--dataset_max_samples=5",
    "--test_perc=0.5",
    "--cv=2",
    "--seed=0",
    "--device=cpu",
    "--verbose_level=1",
]


def test_traincli_parses_the_quickstart_arguments():
    args = TrainCLI().cl.parse_args(QUICKSTART_ARGS)

    assert args.label_id == "QUICKSTART"
    assert args.experiment_tracking_uri == "file:///tmp/mlruns"
    assert args.experiment_tracking_name == "quickstart"
    assert args.dataset_type == "shallow"
    assert args.dataset_resolution == "high"
    assert args.model_arch == "unet"
    assert args.unembed_type == "paper"
    assert (args.conv_filter_size_x, args.conv_filter_size_z) == (3, 3)
    assert args.max_epochs == 1
    assert args.dataset_max_samples == 5
    assert args.test_perc == 0.5
    assert args.cv == 2
    assert args.seed == 0
    assert args.device == "cpu"
    assert args.verbose_level == 1


def test_traincli_defaults():
    args = TrainCLI().cl.parse_args([])

    assert args.device == "cuda:0"
    assert args.max_epochs == 130
    assert args.cv == 5
    assert args.test_perc == 0.1
    assert args.dataset_max_samples == -1
    assert args.dataset_resolution == "high"
    assert args.experiment_tracking_uri.startswith("http")


def test_traincli_rejects_unknown_arguments():
    with pytest.raises(SystemExit):
        TrainCLI().cl.parse_args(["--not_a_real_flag=1"])


def test_refitcli_boolean_flags_accept_explicit_on_and_off():
    on = RefitCLI().cl.parse_args(["--refit", "--gs_best_run_id=abc123"])
    assert on.refit is True
    assert on.gs_best_run_id == "abc123"

    off = RefitCLI().cl.parse_args(["--no-refit"])
    assert off.refit is False


def test_refitcli_refit_flag_is_unset_by_default():
    args = RefitCLI().cl.parse_args([])
    assert args.refit is None
