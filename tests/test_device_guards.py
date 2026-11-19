"""Tests for the CPU/GPU device guards in src/processing/task_processing.py.

These guards exist because ``torch.cuda.is_available()`` can return ``True``
on machines where a real CUDA call still fails (e.g., a container without
GPU passthrough). They must keep GPU behavior unchanged and degrade
gracefully to CPU.
"""

import torch

from src.processing.task_processing import _cuda_usable, cuda_device_str


def test_cuda_usable_returns_bool():
    assert isinstance(_cuda_usable(), bool)


def test_cuda_usable_is_false_when_torch_reports_no_cuda():
    if torch.cuda.is_available():
        # A CUDA driver is visible; the probe may legitimately succeed or
        # fail depending on whether devices are actually reachable.
        usable = _cuda_usable()
        if usable:
            # If usable, a real CUDA call must succeed.
            assert torch.cuda.current_device() is not None
    else:
        assert _cuda_usable() is False


def test_cuda_device_str_is_printable_device_or_cpu_fallback():
    value = cuda_device_str()
    if _cuda_usable():
        assert value == torch.cuda.current_device()
    else:
        assert value == "cpu (no CUDA)"
