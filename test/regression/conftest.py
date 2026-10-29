"""Shared set-up for the regression tests: CPU determinism, restored afterwards."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """One thread and deterministic algorithms for each test, then restore."""
    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    yield
    torch.use_deterministic_algorithms(det)
    torch.set_num_threads(threads)
