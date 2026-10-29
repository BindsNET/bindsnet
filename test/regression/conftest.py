"""Shared set-up for the regression tests: CPU determinism, restored afterwards."""

import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))


@pytest.fixture(autouse=True)
def deterministic_cpu():
    """``networks.deterministic()`` for each test, then restore."""
    import networks

    threads = torch.get_num_threads()
    det = torch.are_deterministic_algorithms_enabled()
    mkldnn = torch.backends.mkldnn.enabled
    networks.deterministic()
    yield
    torch.backends.mkldnn.enabled = mkldnn
    torch.use_deterministic_algorithms(det)
    torch.set_num_threads(threads)
