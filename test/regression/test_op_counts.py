"""
Operation-count tests: a slower hot path shows up as more tensor operations.

Each workload is one call on a small, seeded, fixed-shape object: one
``update()`` of every learning rule (``bindsnet.learning`` on a dense
``Connection`` and ``bindsnet.learning.MCC_learning`` on a
``MulticompartmentConnection``) at batch 1 and 4, one ``Network.run`` step of
``DiehlAndCook2015`` (784 inputs, 100 neurons) at batch 1 and 16, and one
``LIFNodes.forward`` at batch 1 and 4. A ``TorchDispatchMode`` counts every ATen
kernel the call dispatches, per operation name. The counts do not depend on the
CPU, the thread count or the clock, so they are exact on any machine.

The stored counts are in ``op_counts.json``. A test fails when a workload's
total count grows. A drop passes and is reported as a warning with the numbers
(update the file so the lower count becomes the new ceiling). Regenerate with

    python test/regression/regen_op_counts.py --reason "why the counts changed"

Counts can change between torch versions (operations are decomposed
differently), so the tests skip when torch's major.minor differs from the file.
"""

import importlib
import inspect
import json
import os
import warnings
from collections import Counter
from pathlib import Path

import pytest
import torch
from torch.utils._python_dispatch import TorchDispatchMode

import bindsnet.learning.learning as learning
import bindsnet.learning.MCC_learning as mcc_learning
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, SRM0Nodes
from bindsnet.network.topology import Connection, MulticompartmentConnection
from bindsnet.network.topology_features import Weight

COUNTS_FILE = Path(__file__).resolve().parent / "op_counts.json"

N_SRC, N_TGT = 20, 10  # learning-rule connection size
WARMUP = 5  # steps run before counting, so traces and eligibilities are non-zero
RULE_BATCHES = (1, 4)
DC_BATCHES = (1, 16)
LIF_BATCHES = (1, 4)
LIF_N = 100

# Workloads that are not counted, with the reason (copied into op_counts.json).
EXCLUDED = {}


class OpCounter(TorchDispatchMode):
    """Counts every ATen operation dispatched while active, by operation name."""

    def __init__(self):
        super().__init__()
        self.counts = Counter()

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        self.counts[func.name()] += 1
        return func(*args, **(kwargs or {}))


def _rule_classes(module, base):
    return {
        name: cls
        for name, cls in inspect.getmembers(module, inspect.isclass)
        if issubclass(cls, base)
        and cls is not base
        and cls.__module__ == module.__name__
    }


RULES = _rule_classes(learning, learning.LearningRule)
MCC_RULES = _rule_classes(mcc_learning, mcc_learning.MCC_LearningRule)


def _pattern(batch, n, offset):
    """Fixed spike pattern with spikes in every sample (every third neuron)."""
    idx = torch.arange(batch * n).view(batch, n)
    return (idx + offset) % 3 == 0


def _rule_workload(rule, batch, mcc):
    """A fresh in->out network after a seeded warm-up; returns the update call."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    additive = rule is learning.Rmax
    src = Input(n=N_SRC, traces=True, traces_additive=additive)
    if additive:  # Rmax needs stochastic SRM0 targets (Vasilaki et al. 2009).
        tgt = SRM0Nodes(n=N_TGT, traces=True, traces_additive=True)
    else:
        tgt = LIFNodes(n=N_TGT, traces=True, thresh=-60.0)
    w = torch.rand(N_SRC, N_TGT, generator=gen)
    if mcc:
        weight = Weight(
            name="w",
            value=w,
            range=(0.0, 1.0),
            learning_rule=rule,
            nu=(1e-2, 1e-2),
        )
        conn = MulticompartmentConnection(
            source=src, target=tgt, device="cpu", pipeline=[weight]
        )
    else:
        conn = Connection(
            src, tgt, w=w, update_rule=rule, nu=(1e-2, 1e-2), wmin=0.0, wmax=1.0
        )
    net = Network(dt=1.0, batch_size=batch)
    net.add_layer(src, name="in")
    net.add_layer(tgt, name="out")
    net.add_connection(conn, source="in", target="out")
    x = torch.rand(WARMUP, batch, N_SRC, generator=gen) < 0.5
    net.run(inputs={"in": x}, time=WARMUP, reward=1.0)
    # Spikes on both sides, so every pre- and post-synaptic branch runs.
    src.s = _pattern(batch, N_SRC, 0)
    tgt.s = _pattern(batch, N_TGT, 1)
    return lambda: conn.update(mask=None, learning=True, reward=1.0)


def _dc2015_workload(batch):
    from bindsnet.models import DiehlAndCook2015

    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    net = DiehlAndCook2015(
        n_inpt=784, n_neurons=100, batch_size=batch, inpt_shape=(1, 28, 28)
    )
    x = torch.rand(WARMUP + 1, batch, 1, 28, 28, generator=gen) < 0.2
    net.run(inputs={"X": x[:WARMUP]}, time=WARMUP)
    step = {"X": x[WARMUP:]}
    return lambda: net.run(inputs=step, time=1)


def _lif_workload(batch):
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    net = Network(dt=1.0, batch_size=batch)
    layer = LIFNodes(n=LIF_N, traces=True)
    net.add_layer(layer, name="lif")
    x = torch.rand(WARMUP + 1, batch, LIF_N, generator=gen) * 20.0
    net.run(inputs={"lif": x[:WARMUP]}, time=WARMUP)
    step = x[WARMUP]
    return lambda: layer.forward(step)


def _node_classes():
    """Every concrete Nodes class in bindsnet.network.nodes except LIFNodes (which
    has its own workload, warmed up through Network.run)."""
    import bindsnet.network.nodes as nodes_module

    return {
        name: cls
        for name, cls in inspect.getmembers(nodes_module, inspect.isclass)
        if issubclass(cls, nodes_module.Nodes)
        and not inspect.isabstract(cls)
        and cls.__module__ == nodes_module.__name__
        and cls is not LIFNodes
    }


def _node_workload(cls, batch):
    """One forward() of ``cls`` after a seeded warm-up of forward() calls."""
    torch.manual_seed(0)
    gen = torch.Generator().manual_seed(1)
    net = Network(dt=1.0, batch_size=batch)
    layer = cls(n=LIF_N, traces=True)
    net.add_layer(layer, name="layer")
    # CSRMNodes takes a window of inputs, [batch, window, n].
    window = (int(layer.res_window_size),) if hasattr(layer, "res_window_size") else ()
    x = torch.rand(WARMUP + 1, batch, *window, LIF_N, generator=gen) * 20.0
    for t in range(WARMUP):
        layer.forward(x[t].clone())
    step = x[WARMUP]
    return lambda: layer.forward(step.clone())


def workloads():
    """Workload name -> function that builds it and returns the call to count."""
    out = {}
    for name, cls in sorted(RULES.items()):
        for b in RULE_BATCHES:
            out[f"rule.{name}.b{b}"] = lambda c=cls, b=b: _rule_workload(c, b, False)
    for name, cls in sorted(MCC_RULES.items()):
        for b in RULE_BATCHES:
            out[f"mcc.{name}.b{b}"] = lambda c=cls, b=b: _rule_workload(c, b, True)
    for b in DC_BATCHES:
        out[f"run.DiehlAndCook2015_784x100.b{b}"] = lambda b=b: _dc2015_workload(b)
    for b in LIF_BATCHES:
        out[f"nodes.LIFNodes_forward.b{b}"] = lambda b=b: _lif_workload(b)
        for name, cls in sorted(_node_classes().items()):
            out[f"nodes.{name}_forward.b{b}"] = lambda c=cls, b=b: _node_workload(c, b)
    return out


WORKLOADS = workloads()
COUNTED = sorted(set(WORKLOADS) - set(EXCLUDED))


def count_ops(name):
    """Build workload ``name`` fresh and count the operations of one call."""
    call = WORKLOADS[name]()
    with OpCounter() as counter:
        call()
    return dict(sorted(counter.counts.items()))


def load_counts():
    return json.loads(COUNTS_FILE.read_text())


def _major_minor(version):
    return tuple(version.split("+")[0].split(".")[:2])


@pytest.fixture(scope="module")
def stored():
    data = load_counts()
    if _major_minor(data["meta"]["torch"]) != _major_minor(torch.__version__):
        # On CI a torch upgrade must regenerate the counts in the same PR.
        (pytest.fail if os.environ.get("CI") else pytest.skip)(
            f"op counts were made with torch {data['meta']['torch']}; regenerate "
            "with test/regression/regen_op_counts.py --reason"
        )
    return data


def _diff(old, new):
    keys = sorted(set(old) | set(new))
    rows = [
        f"  {k}: {old.get(k, 0)} -> {new.get(k, 0)}"
        for k in keys
        if old.get(k, 0) != new.get(k, 0)
    ]
    return "\n".join(rows) or "  (no per-operation change)"


@pytest.mark.parametrize("name", COUNTED)
def test_op_count_does_not_grow(name, stored):
    """The operation count of one call must not grow above the stored count.

    Caught: a second ``self.connection.w.clamp_(wmin, wmax)`` in
    ``LearningRule.update`` (13 rule.* workloads failed, +1 each); and
    ``self.v = self.v.clone()`` added to ``LIFNodes.forward`` (4 failed: both
    nodes.LIFNodes_forward and both run.DiehlAndCook2015, +1 each). Removing an
    operation instead gives a warning and passes.
    """
    assert name in stored["counts"], (
        f"no stored count for new workload {name}; run "
        "test/regression/regen_op_counts.py --reason '...'"
    )
    old = stored["counts"][name]
    new = count_ops(name)
    old_total, new_total = sum(old.values()), sum(new.values())
    assert new_total <= old_total, (
        f"{name}: {new_total} operations per call, stored {old_total} "
        f"(+{new_total - old_total}). A hot path does more work. If intended, "
        "regenerate with test/regression/regen_op_counts.py --reason '...'.\n"
        + _diff(old, new)
    )
    if new != old:
        warnings.warn(
            f"{name}: operation count {old_total} -> {new_total} (not a failure; "
            "regenerate op_counts.json so the new count becomes the ceiling)\n"
            + _diff(old, new)
        )


def test_workload_list_matches_file(stored):
    """Every workload has a stored count or a stated exclusion, and nothing else.

    Caught: a new class ``class Extra(PostPre): pass`` appended to learning.py
    (no stored count), and deleting the rule.Hebbian.b4 entry from op_counts.json.
    """
    have = set(stored["counts"])
    missing = sorted(set(COUNTED) - have)
    extra = sorted(have - set(COUNTED))
    assert not missing and not extra, (
        f"op_counts.json is out of date: missing {missing}, no longer measured "
        f"{extra}; regenerate with test/regression/regen_op_counts.py --reason"
    )
    assert stored.get("excluded", {}) == EXCLUDED


def test_regen_requires_reason():
    """The regeneration script refuses to run without a non-empty reason.

    Caught: the ``ap.error`` check for a missing or blank --reason disabled.
    """
    regen = importlib.import_module("regen_op_counts")
    with pytest.raises(SystemExit):
        regen.main([])
    with pytest.raises(SystemExit):
        regen.main(["--reason", "  "])
