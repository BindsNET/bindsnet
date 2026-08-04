"""
Shared machinery for the ExampleNetwork MCC benchmarks (imported by
``foldable_pipelines.py``, ``sparse_compute.py`` and ``both.py``).

Each benchmark reports the %-time speedup of one MCC configuration over another by
timing the *same* ExampleNetwork under different ``compute`` modes:

  * ``expansion``   -- the pre-optimization path, reconstructed here as a
                       monkeypatch: materialize ``[batch, src, tgt]``, apply every
                       feature elementwise, then sum over source. (This path was
                       removed from the code when the fold landed, so we rebuild it
                       to serve as the baseline.)
  * ``fold``        -- the current folded path: ``out = s @ A + B.sum(0)``.
  * ``fold_sparse`` -- the fold plus activity-sparse compute (``sparse_compute``):
                       read only the weight rows of source neurons that spiked.

Speedup is wall-time reduction: ``(t_baseline - t_new) / t_baseline * 100``
(positive = faster). Learning is disabled so we time the pure forward path (the
part these optimizations affect).
"""

import os
import statistics
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))  # .../examples/benchmark/<name>
_EXAMPLES = os.path.dirname(os.path.dirname(_HERE))  # .../examples
_ROOT = os.path.dirname(_EXAMPLES)  # repo root (for ``bindsnet``)
_STRESS = os.path.join(_EXAMPLES, "stress_test")  # for ``example_network``
for _p in (_ROOT, _STRESS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch

from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Degradation, Probability
from example_network import ExampleNetwork

# ExampleNetwork sizes per device: 20k excitatory neurons on GPU (where the fold
# shines), a smaller net on CPU so the baseline finishes in reasonable time.
GPU_CONFIG = dict(in_size=100, exc_size=20_000, inh_size=2_000)
CPU_CONFIG = dict(in_size=100, exc_size=2_000, inh_size=200)
GPU_TIME, CPU_TIME = 50, 20
REPS, WARMUP = 5, 2


def _expansion_compute(self, s):
    """The pre-fold ``[batch, src, tgt]`` expansion path (baseline)."""
    s = s.view(s.size(0), self.source.n)
    cs = s.view(s.size(0), self.source.n, 1).repeat(1, 1, self.target.n)
    for f in self.pipeline:
        op = getattr(f, "op", "mul")
        if isinstance(f, Probability):
            v = torch.bernoulli(f.value)
        elif isinstance(f, Degradation):
            v = (
                f.degrade_function(f.value)
                if f.degrade_function is not None
                else f.value
            )
        else:
            v = f.value
        if op == "mul":
            cs = cs * v
        elif op == "add":
            cs = cs + v
        else:
            cs = cs - v
    out = cs.sum(1)
    if getattr(self, "traces", False):
        self.activity = out
    if out.size() != torch.Size([s.size(0)] + self.target.shape):
        return out.view(s.size(0), *self.target.shape)
    return out


def _set_mode(net, mode):
    for c in net.connections.values():
        if isinstance(c, MulticompartmentConnection):
            if mode == "expansion":
                c.compute = _expansion_compute.__get__(c)  # instance override
                c.sparse_compute = False
            else:
                c.__dict__.pop("compute", None)  # restore the class (fold) method
                c.sparse_compute = mode == "fold_sparse"


def _bench(net, inputs, T, device, modes):
    cuda = device.startswith("cuda")
    net.train(False)  # forward-only: time the compute path the optimizations touch
    for m in modes:  # warmup each mode
        _set_mode(net, m)
        for _ in range(WARMUP):
            net.reset_state_variables()
            net.run(inputs=inputs, time=T)
    if cuda:
        torch.cuda.synchronize()
    samples = {m: [] for m in modes}
    for _ in range(REPS):  # interleave modes each rep to counter thermal drift
        for m in modes:
            _set_mode(net, m)
            net.reset_state_variables()
            if cuda:
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            net.run(inputs=inputs, time=T)
            if cuda:
                torch.cuda.synchronize()
            samples[m].append(time.perf_counter() - t0)
    return {m: statistics.median(v) * 1e3 for m, v in samples.items()}


def run_speedup(baseline_mode, test_mode, technique):
    """Build the ExampleNetwork on CPU and GPU, time both modes, print the speedup."""
    print("=" * 72)
    print(f"{technique}")
    print(f"  (%-time speedup of '{test_mode}' vs baseline '{baseline_mode}')")
    print("=" * 72)

    devices = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    if not torch.cuda.is_available():
        print("[note] CUDA not available -> reporting CPU only.\n")

    for dev in devices:
        cfg = GPU_CONFIG if dev == "cuda" else CPU_CONFIG
        T = GPU_TIME if dev == "cuda" else CPU_TIME
        net = ExampleNetwork(device=dev, **cfg)
        inputs = net.make_input(T)
        res = _bench(net, inputs, T, dev, [baseline_mode, test_mode])
        base, new = res[baseline_mode], res[test_mode]
        speedup = (base - new) / base * 100.0
        print(
            f"  [{dev.upper():4s}] exc={cfg['exc_size']:>6d}  time={T:>3d}  |  "
            f"{baseline_mode}={base:8.2f} ms   {test_mode}={new:8.2f} ms   "
            f"|  speedup = {speedup:+6.1f}% time"
        )
        del net
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    print()
