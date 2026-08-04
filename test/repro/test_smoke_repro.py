"""
Seeded smoke-reproduction test (NeuroEval WO-07).

Builds a tiny, fully deterministic BindsNET network on CPU, drives it with a
seeded Bernoulli spike train, and asserts an exact, pre-measured output. This is a
fast end-to-end reproduction check: with a fixed seed the simulation must produce
the same result on every run and in CI.

The expected value (179 output spikes) was measured on CPU with the pinned
``torch`` (2.11.x). If a future ``torch`` upgrade changes CPU RNG and this value
shifts, re-measure with the same builder and update ``EXPECTED_SPIKES``.
"""

import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

SEED = 0
TIME = 100
EXPECTED_SPIKES = 179


def _build_and_run(seed: int = SEED, time: int = TIME) -> int:
    """Build the fixed network, run it on CPU, return total output spikes."""
    torch.manual_seed(seed)
    network = Network(dt=1.0, learning=False)

    inpt = Input(n=100)
    out = LIFNodes(n=50)
    network.add_layer(inpt, name="X")
    network.add_layer(out, name="Y")

    # Static weights (no learning rule) generated from the same seed.
    w = 0.3 * torch.rand(100, 50)
    network.add_connection(Connection(inpt, out, w=w), source="X", target="Y")

    monitor = Monitor(out, state_vars=["s"], time=time)
    network.add_monitor(monitor, name="Y")

    # Seeded input spike train, shape [time, n].
    torch.manual_seed(seed)
    inputs = {"X": torch.bernoulli(0.1 * torch.rand(time, inpt.n))}

    network.run(inputs=inputs, time=time)
    return int(monitor.get("s").sum().item())


def test_smoke_repro_matches_expected():
    """The seeded run reproduces the pre-measured output spike count."""
    assert _build_and_run() == EXPECTED_SPIKES


def test_smoke_repro_is_deterministic():
    """Repeated seeded runs produce identical results."""
    results = {_build_and_run() for _ in range(3)}
    assert results == {EXPECTED_SPIKES}
