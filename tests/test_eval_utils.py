"""Tests for the architecture-name mapping used in the paper tables
(src/pdf/eval_utils.py::EvalUtils.arch_name2arch_name_paper).

Expected values are the function's actual output format (``"U-Net k=3 r=1"``
-- no commas), which is what the paper's tables and MLflow run names use.
The standalone self-test ``test_arch_name2arch_name_paper`` in eval_utils.py
uses the same expectations.
"""

import pytest

from src.pdf.eval_utils import EvalUtils

VALID_CASES = [
    ("unet.cks_3x3.cds_1x1.tcks_2x2.opt_Adam", "U-Net k=3 r=1"),
    ("unet.cks_5x5.cds_2x2.tcks_2x2.opt_AdamW", "U-Net k=5 r=2"),
    ("transunet.cks_7x7.cds_4x4.tcks_4x4.opt_Adam", "TransUNet k=7 r=4"),
]

INVALID_CASES = [
    "resnet.cks_3x3.cds_1x1.tcks_2x2.opt_Adam",    # unknown architecture
    "unet.kernel_3x3.cds_1x1.tcks_2x2.opt_Adam",   # invalid cks token
    "unet.cks_3x3.dilation_1x1.tcks_2x2.opt_Adam"  # invalid cds token
]


@pytest.mark.parametrize(("arch_name", "expected"), VALID_CASES)
def test_valid_arch_names_map_to_paper_names(arch_name, expected):
    assert EvalUtils.arch_name2arch_name_paper(arch_name) == expected


@pytest.mark.parametrize("arch_name", INVALID_CASES)
def test_malformed_arch_names_raise_value_error(arch_name):
    with pytest.raises(ValueError):
        EvalUtils.arch_name2arch_name_paper(arch_name)
