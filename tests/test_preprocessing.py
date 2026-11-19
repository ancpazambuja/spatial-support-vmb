"""Tests for the sklearn-style preprocessing transformers in
src/preprocessing/pipeline.py, using small synthetic arrays.

Shape conventions of the pipeline (as used by the apps):
    X: (n_samples, ..., h, w, n_channels)  (channels-last)
    Y: (n_samples, h, w)                    (velocity model, m/s)
"""

import numpy as np
import torch

from src.preprocessing.pipeline import (
    BatchAbsMaxScaler,
    ToChannelsFirstOrder,
    ToTensor,
    XClipAndNormalization,
    YUnitTransform,
)

rng = np.random.default_rng(0)


def test_batch_abs_max_scaler_maps_to_unit_range_and_inverts():
    X = rng.normal(size=(2, 3, 4, 5)).astype(np.float32) * 100.0

    scaler = BatchAbsMaxScaler()
    fitted = scaler.fit(X)
    assert fitted is scaler
    assert scaler.absmax.shape == (4, 5)
    assert (scaler.absmax > 0).all()

    X_norm = scaler.transform(X.copy())
    assert np.isclose(np.max(np.abs(X_norm)), 1.0, rtol=1e-5)
    assert np.max(np.abs(X_norm)) <= 1.0 + 1e-6

    X_back = scaler.inverse_transform(X_norm.copy())
    assert np.allclose(X_back, X, rtol=1e-4, atol=1e-4)


def test_batch_abs_max_scaler_is_fitted_per_channel_block():
    # A large value in one (h, w) position must define the scale of the
    # whole (0, 1) slice pair it belongs to (normalization over axis (0, 1)).
    X = np.ones((2, 3, 4, 5), dtype=np.float32)
    X[0, 0, 2, 1] = 50.0

    scaler = BatchAbsMaxScaler().fit(X)
    assert scaler.absmax[2, 1] == 50.0
    assert scaler.absmax.sum() > 50.0  # other positions are 1.0


def test_to_tensor_converts_numpy_to_float32_tensor():
    X = rng.normal(size=(2, 4, 5, 3)).astype(np.float64)

    out = ToTensor().fit_transform(X)

    assert isinstance(out, torch.Tensor)
    assert out.dtype == torch.float32
    assert out.shape == X.shape


def test_to_channels_first_order_bhwc_to_bchw():
    X = torch.as_tensor(rng.normal(size=(2, 4, 5, 3)).astype(np.float32))

    out = ToChannelsFirstOrder().fit_transform(X)

    assert out.shape == (2, 3, 4, 5)
    # Values must be preserved under the permutation.
    assert torch.equal(out, X.permute(0, 3, 1, 2))


def test_to_channels_first_order_hwc_to_chw():
    X = torch.as_tensor(rng.normal(size=(4, 5, 3)).astype(np.float32))

    out = ToChannelsFirstOrder().fit_transform(X)

    assert out.shape == (3, 4, 5)


def test_x_clip_and_normalization_maps_to_0_1_range():
    X = (rng.normal(size=(2, 10, 8, 3)) * 1000.0).astype(np.float32)

    norm = XClipAndNormalization()
    norm.fit(X)
    assert norm.clip.shape == (3,)
    assert (norm.clip > 0).all()

    out = norm.transform(X.copy())

    # Float32 rounding can leave ~1e-8 residue at the clipped bounds.
    assert out.min() > -1e-6
    assert out.max() <= 1.0 + 1e-6


def test_y_unit_transform_converts_mps_to_kmps():
    Y = (rng.uniform(1500.0, 5000.0, size=(3, 4, 5))).astype(np.float32)

    out = YUnitTransform().fit_transform(Y.copy())

    assert np.allclose(out, Y / 1000.0)
