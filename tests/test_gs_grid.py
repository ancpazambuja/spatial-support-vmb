"""Tests for the grid-search parameter grids in src/processing/gs_grid.py.

The device entry of the skorch ``param_grid`` is resolved from
``global_vars.get_device()`` (the task-level 'cuda:all' special case maps
to plain 'cuda'); these tests pin that contract.
"""

import numpy as np
import pytest
import torch

from src.processing.gs_grid import Grid, GSGridAdamW


class _FakeRawData:
    """Minimal stand-in with the Data interface used to derive model size."""

    def get_current_data(self):
        return self

    def get_X(self):
        return np.zeros((5, 15, 352, 896), dtype=np.float32)


class _FakeGlobalVars:
    """Just the getters consumed by Grid.get_grid()."""

    def __init__(self, data_type="shallow", device="cuda:all",
                 model_arch="unet", max_epochs=1):
        self._data_type = data_type
        self._device = device
        self._model_arch = model_arch
        self._max_epochs = max_epochs

    def get_raw_train_data(self):
        return _FakeRawData()

    def get_conv_filter_size(self):
        return (3, 3)

    def get_transposed_conv_filter_size(self):
        return (2, 2)

    def get_conv_dilation_size(self):
        return (1, 1)

    def get_data_type(self):
        return self._data_type

    def get_device(self):
        return self._device

    def is_transunet(self):
        return self._model_arch == "transunet"

    def get_max_epochs(self):
        return self._max_epochs

    def get_unembed_type(self):
        return "paper"


def test_grid_maps_cuda_all_to_plain_cuda_for_skorch():
    grid = Grid(global_vars=_FakeGlobalVars(device="cuda:all"))

    assert grid.grid, "param grid must not be empty"
    for param_grid in grid.grid:
        assert param_grid["device"] == ["cuda"]


def test_grid_passes_cpu_and_single_gpu_devices_through():
    for device in ("cpu", "cuda:2"):
        grid = Grid(global_vars=_FakeGlobalVars(device=device))
        for param_grid in grid.grid:
            assert param_grid["device"] == [device]


def test_grid_attention_img_size_shallow_vs_deep():
    shallow = Grid(global_vars=_FakeGlobalVars(data_type="shallow"))
    for param_grid in shallow.grid:
        assert param_grid["module__attention_img_size"] == [(11, 28)]

    deep = Grid(global_vars=_FakeGlobalVars(data_type="deep"))
    for param_grid in deep.grid:
        assert param_grid["module__attention_img_size"] == [(21, 25)]


def test_grid_unknown_data_type_raises():
    with pytest.raises(ValueError):
        Grid(global_vars=_FakeGlobalVars(data_type="bogus"))


def test_grid_channels_come_from_model_data_size():
    grid = Grid(global_vars=_FakeGlobalVars())
    for param_grid in grid.grid:
        assert param_grid["module__in_channels"] == [15]


def test_grid_uses_adam_and_adamw_grid_uses_adamw():
    base = Grid(global_vars=_FakeGlobalVars())
    for param_grid in base.grid:
        assert param_grid["optimizer"] == [torch.optim.Adam]

    adamw = GSGridAdamW(global_vars=_FakeGlobalVars())
    for param_grid in adamw.grid:
        assert param_grid["optimizer"] == [torch.optim.AdamW]


def test_grid_expand_produces_gridsearchcv_suitable_entries():
    grid = Grid(global_vars=_FakeGlobalVars())

    expanded = grid.expand()

    # The (lr, weight_decay) groups expand to 3 + 3 + 3 + 2 = 11 grid
    # points -- the same 11 "hpset_*" runs produced by every quickstart
    # execution of the pipeline.
    assert len(expanded) == 11
    for entry in expanded:
        assert isinstance(entry, dict)
        assert all(isinstance(v, list) and len(v) == 1 for v in entry.values())
        assert "lr" in entry and "optimizer__weight_decay" in entry
