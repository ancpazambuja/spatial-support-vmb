"""Tests for metrics and model-size helpers in src/utl/classes.py."""

import numpy as np
import torch

from src.utl.classes import Utils


def _random_image(seed=0, size=64):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(size, size, generator=g)


def test_ssim_of_identical_images_is_one():
    img = _random_image()
    mean, ssim_map = Utils.ssim(img, img.clone())
    assert np.isclose(mean, 1.0)
    assert ssim_map.shape == img.shape


def test_ssim_of_noisy_copy_is_below_one_but_positive():
    img = _random_image()
    noisy = (img + 0.1 * torch.randn(img.shape)).clamp(0.0, 1.0)
    mean, _ = Utils.ssim(img, noisy)
    assert 0.0 < mean < 1.0


def test_ssim_is_symmetric():
    img1, img2 = _random_image(1), _random_image(2)
    mean_12, _ = Utils.ssim(img1, img2)
    mean_21, _ = Utils.ssim(img2, img1)
    assert np.isclose(mean_12, mean_21)


class _StubData:
    """Minimal stand-in with the Data interface used by get_model_data_size."""

    def get_current_data(self):
        return self

    def get_X(self):
        return np.zeros((5, 15, 352, 896), dtype=np.float32)


def test_get_model_data_size_matches_the_apps_logged_values():
    # Same values logged by every run of the pipeline for the high
    # resolution "shallower" data (see MLflow params / run logs).
    size = Utils.get_model_data_size(_StubData(), depth=5)

    assert size == {"nchannels": 15, "nz_cut": 352, "nx_cut": 896}


def test_find_size_keeps_sizes_already_divisible_by_the_pooling_depth():
    # (352, 896) is already divisible by 2**5, so it is returned unchanged.
    assert Utils.find_size(352, 896, depth=5) == (352, 896)
