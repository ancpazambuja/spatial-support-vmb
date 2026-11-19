"""Smoke tests for the neural network modules and their skorch builders.

The models are constructed with a tiny ``n_filters_init`` on purpose: these
tests check interfaces and shapes, not capacity or accuracy.

NOTE: the apps never build ``UNet`` directly -- both grid search and refit
construct ``TransUNet`` through ``SkorchTransUNetModelBuilder``, with
``use_transformers=False`` for the plain U-Net architecture.
"""

import pytest
import torch

from src.processing.ml_models import UNet, TransUNet
from src.processing.ml_models_builder import (
    ScoredSkorchModel,
    SkorchGeneralModelBuilder,
    SkorchTransUNetModelBuilder,
)

# The contracting path pools 5 times: the input side must be a multiple of
# 2**5 for the shapes to close.
INPUT_SHAPE = (2, 1, 32, 32)


def _plain_unet_module(**kwargs):
    """The module actually used by the apps for the 'unet' architecture."""
    defaults = dict(
        in_channels=1, out_channels=1, n_filters_init=2,
        use_transformers=False, unembed_type="paper",
        attention_config=None, attention_img_size=None,
    )
    defaults.update(kwargs)
    return TransUNet(**defaults)


def test_unet_forward_preserves_spatial_shape():
    model = UNet(in_channels=1, out_channels=1, n_filters_init=2)
    model.eval()

    x = torch.rand(INPUT_SHAPE)
    with torch.no_grad():
        y = model(x)

    assert y.shape == INPUT_SHAPE


def test_transunet_without_transformers_forward_preserves_spatial_shape():
    model = _plain_unet_module()
    model.eval()

    x = torch.rand(INPUT_SHAPE)
    with torch.no_grad():
        y = model(x)

    assert y.shape == INPUT_SHAPE


def test_transunet_rejects_invalid_unembed_type():
    with pytest.raises(AssertionError, match="unembed_type"):
        TransUNet(unembed_type="bogus")


def test_transunet_with_transformers_forward_preserves_spatial_shape():
    from src.transunet.vit_seg_configs import get_attention_regression_config

    model = _plain_unet_module(
        use_transformers=True,
        attention_config=get_attention_regression_config(
            num_layers=1, hidden_size=8, num_heads=2, mlp_dim=16),
        attention_img_size=(1, 1),  # bottleneck of a 32x32 input after 5 pools
        attention_in_channels=64,   # n_filters_init * 32
        unembed_type="paper",       # as used by the paper's configurations
    )
    model.eval()

    x = torch.rand(INPUT_SHAPE)
    with torch.no_grad():
        y = model(x)

    assert y.shape == INPUT_SHAPE


def test_general_builder_returns_scored_skorch_model():
    # NOTE: SkorchGeneralModelBuilder is legacy (no call sites in src/):
    # it re-instantiates the module class with a fixed parameter set that
    # lacks ``use_transformers``, so initialize() would default to the
    # transformer path and fail without an attention config. The apps only
    # use SkorchTransUNetModelBuilder. Here we pin the builder contract:
    # it must produce a ScoredSkorchModel carrying the given module.
    module = _plain_unet_module()

    net = SkorchGeneralModelBuilder().build(
        model=module,
        max_epochs=1,
        batch_size=2,
        device="cpu",
    )

    assert isinstance(net, ScoredSkorchModel)
    assert net.module is module
    assert net.max_epochs == 1
    assert net.device == "cpu"


def test_transunet_builder_returns_scored_skorch_model():
    net = SkorchTransUNetModelBuilder().build(
        seed=0,
        max_epochs=1,
        batch_size=2,
        device="cpu",
        train_split=None,
        in_channels=1,
        out_channels=1,
        n_filters_init=2,
        use_transformers=False,
        unembed_type="paper",
        attention_img_size=None,
        attention_in_channels=64,
        attention_hidden_size=8,
        attention_num_heads=2,
        attention_num_layers=1,
        attention_mlp_dim=16,
    )

    assert isinstance(net, ScoredSkorchModel)

    net.initialize()
    assert isinstance(net.module_, TransUNet)
    assert net.module_.use_transformers is False

    x = torch.rand(INPUT_SHAPE)
    y_hat = net.predict(x)

    assert y_hat.shape == INPUT_SHAPE
