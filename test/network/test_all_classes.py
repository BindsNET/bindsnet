"""Construct-and-run smoke tests for every public class in the core modules.

The classes are found by walking the modules (``inspect.getmembers``), not by a
hand-written list, so a class added later is exercised (or the test tells its author
to add a shape spec) without anyone remembering to write a test for it.

Each case builds the object, runs 10 steps on seeded Bernoulli input and checks that
outputs are finite and have the right shapes. Learning-rule cases also check that the
weights moved, and the table of which rule supports which connection type is pinned.

Classes or combinations that crash on master are listed in ``KNOWN_BUGS`` at the end
with their exact error. When someone fixes one, its reproducer stops raising, the test
fails with "DID NOT RAISE", and the entry must be removed (and the normal test
widened) in the same change.
"""

import inspect

import pytest
import torch

import bindsnet.learning as learning
import bindsnet.learning.MCC_learning as mcc_learning
import bindsnet.learning.reward as reward
import bindsnet.models as models
import bindsnet.network.monitors as monitors
import bindsnet.network.nodes as nodes
import bindsnet.network.topology as topology
import bindsnet.network.topology_features as tf
from bindsnet.network import Network

STEPS = 10
BATCH = 2
RATE = 0.5


def _public_classes(module, base):
    """Concrete classes defined in bindsnet that subclass ``base`` (not ``base``)."""
    found = {}
    for name, cls in inspect.getmembers(module, inspect.isclass):
        if name.startswith("_") or not cls.__module__.startswith("bindsnet"):
            continue
        if not issubclass(cls, base) or cls is base or inspect.isabstract(cls):
            continue
        found[cls.__name__] = cls
    return found


NODE_CLASSES = _public_classes(nodes, nodes.Nodes)
CONNECTION_CLASSES = {
    **_public_classes(topology, topology.AbstractConnection),
    **_public_classes(topology, topology.AbstractMulticompartmentConnection),
}
RULE_CLASSES = _public_classes(learning, learning.LearningRule)
MCC_RULE_CLASSES = _public_classes(mcc_learning, mcc_learning.MCC_LearningRule)
MODEL_CLASSES = _public_classes(models, Network)


def _spikes(shape, batch=BATCH, seed=0):
    """Seeded Bernoulli spikes of shape ``[STEPS, batch, *shape]``."""
    gen = torch.Generator().manual_seed(seed)
    p = torch.full((STEPS, batch, *shape), RATE)
    return torch.bernoulli(p, generator=gen).bool()


def _check_layers(net, batch=BATCH):
    """Every layer: spikes shaped [batch, *layer.shape]; state tensors finite."""
    for name, layer in net.layers.items():
        want = (batch, *layer.shape)
        assert tuple(layer.s.shape) == want, f"{name}.s {tuple(layer.s.shape)}"
        assert layer.s.dtype == torch.bool, f"{name}.s dtype {layer.s.dtype}"
        for attr in ("v", "x", "theta"):
            t = getattr(layer, attr, None)
            if isinstance(t, torch.Tensor) and t.dim() > 0:
                assert torch.isfinite(t).all(), f"{name}.{attr} not finite"


def _weights(conn):
    """The connection's weight tensor (dense, or the MCC Weight feature's value)."""
    if isinstance(conn, topology.AbstractMulticompartmentConnection):
        for f in conn.pipeline:
            if isinstance(f, tf.Weight):
                v = f.value
                return v.to_dense() if v.is_sparse else v
        return None
    w = getattr(conn, "w", None)
    if isinstance(w, torch.Tensor) and w.is_sparse:
        return w.to_dense()
    return w


# --------------------------------------------------------------------------- #
# Nodes                                                                       #
# --------------------------------------------------------------------------- #

NODE_SHAPE = (3, 4)


def _run_nodes(name, shape, batch=BATCH):
    torch.manual_seed(0)
    cls = NODE_CLASSES[name]
    n = int(torch.tensor(shape).prod())
    net = Network(dt=1.0, batch_size=batch)
    net.add_layer(nodes.Input(shape=shape, traces=True), name="in")
    net.add_layer(cls(n=n, shape=shape, traces=True), name="out")
    inputs = {"in": _spikes(shape, batch)}
    if isinstance(net.layers["out"], nodes.AbstractInput):
        inputs["out"] = _spikes(shape, batch, seed=1)
    else:
        w = torch.rand(n, n, generator=torch.Generator().manual_seed(2)) * 4
        conn = topology.Connection(net.layers["in"], net.layers["out"], w=w)
        net.add_connection(conn, source="in", target="out")
    net.run(inputs=inputs, time=STEPS)
    return net


@pytest.mark.parametrize("name", sorted(NODE_CLASSES))
def test_nodes_construct_and_run(name):
    """Every Nodes subclass runs 10 steps with finite state of the right shape.

    Caught: ``self.v = self.v / 0`` inserted at the end of ``BoostedLIFNodes.forward``.
    """
    net = _run_nodes(name, NODE_SHAPE)
    _check_layers(net)


# --------------------------------------------------------------------------- #
# Connections                                                                 #
# --------------------------------------------------------------------------- #

# name -> (source shape, target shape, constructor keyword arguments).
CONNECTION_SPECS = {
    "Connection": ((20,), (10,), {}),
    "SparseConnection": ((20,), (10,), {}),
    "MeanFieldConnection": ((20,), (10,), {}),
    "Conv1dConnection": ((2, 12), (3, 10), {"kernel_size": 3}),
    "Conv2dConnection": ((2, 8, 8), (3, 6, 6), {"kernel_size": 3}),
    "Conv3dConnection": ((1, 6, 6, 6), (2, 4, 4, 4), {"kernel_size": 3}),
    "MaxPool1dConnection": ((2, 8), (2, 4), {"kernel_size": 2, "stride": 2}),
    "MaxPool2dConnection": ((2, 8, 8), (2, 4, 4), {"kernel_size": 2, "stride": 2}),
    "MaxPoo3dConnection": (
        (2, 4, 4, 4),
        (2, 2, 2, 2),
        {"kernel_size": 2, "stride": 2},
    ),
    "LocalConnection": (
        (1, 8, 8),
        (2, 3, 3),
        {"kernel_size": 4, "stride": 2, "n_filters": 2},
    ),
    "LocalConnection1D": (
        (2, 12),
        (2, 4),
        {"kernel_size": 3, "stride": 3, "n_filters": 2},
    ),
    "LocalConnection2D": (
        (2, 8, 8),
        (3, 3, 3),
        {"kernel_size": 4, "stride": 2, "n_filters": 3},
    ),
    "LocalConnection3D": (
        (1, 6, 6, 6),
        (2, 3, 3, 3),
        {"kernel_size": 2, "stride": 2, "n_filters": 2},
    ),
    "MulticompartmentConnection": ((20,), (10,), {}),
}
# A connection class not listed above is tried with this dense spec, and
# test_every_connection_class_has_a_shape_spec fails until it is listed.
DEFAULT_SPEC = ((20,), (10,), {})
# Extra arguments that are not optional in practice (see KNOWN_BUGS).
MAXPOOL = {"MaxPool1dConnection", "MaxPool2dConnection", "MaxPoo3dConnection"}


def _build(conn_name, rule=None, batch=BATCH, spec=None, **extra):
    """Network ``in -> out`` joined by one ``conn_name`` connection."""
    torch.manual_seed(0)
    if spec is None:
        spec = CONNECTION_SPECS.get(conn_name, DEFAULT_SPEC)
    src_shape, tgt_shape, kwargs = spec
    kwargs = {**kwargs, **extra}
    cls = CONNECTION_CLASSES[conn_name]
    additive = rule is learning.Rmax
    src = nodes.Input(shape=src_shape, traces=True, traces_additive=additive)
    if additive:  # Rmax needs stochastic SRM0 targets (Vasilaki et al. 2009).
        tgt = nodes.SRM0Nodes(shape=tgt_shape, traces=True, traces_additive=True)
    else:
        tgt = nodes.LIFNodes(shape=tgt_shape, traces=True, thresh=-60.0)
    net = Network(dt=1.0, batch_size=batch)
    net.add_layer(src, name="in")
    net.add_layer(tgt, name="out")
    if cls is topology.MulticompartmentConnection:
        w = torch.rand(src.n, tgt.n)
        weight = tf.Weight(name="w", value=w, range=(0.0, 1.0), **kwargs)
        conn = cls(source=src, target=tgt, device="cpu", pipeline=[weight])
    else:
        if rule is not None:
            kwargs.update(update_rule=rule, nu=(1e-2, 1e-2), wmin=0.0, wmax=1.0)
        conn = cls(src, tgt, **kwargs)
    net.add_connection(conn, source="in", target="out")
    return net, conn, src_shape


@pytest.mark.parametrize("name", sorted(CONNECTION_CLASSES))
def test_connection_construct_and_run(name):
    """Every connection class runs 10 steps; target state finite and shaped.

    Caught: ``.view_as(indices)`` changed to ``.view(-1)`` in ``MaxPool1dConnection.compute``.
    """
    batch = BATCH
    extra = {"decay": 0.5} if name in MAXPOOL else {}
    net, conn, src_shape = _build(name, batch=batch, **extra)
    net.run(inputs={"in": _spikes(src_shape, batch)}, time=STEPS)
    _check_layers(net, batch)
    out = conn.compute(net.layers["in"].s)
    assert torch.isfinite(out).all()
    # The pre-activation must broadcast onto the target voltage.
    target_v = torch.zeros(batch, *net.layers["out"].shape)
    assert (target_v + out).shape == target_v.shape, tuple(out.shape)
    w = _weights(conn)
    if w is not None:
        assert torch.isfinite(w).all()


def test_every_connection_class_has_a_shape_spec():
    """A new connection class must get an entry in CONNECTION_SPECS.

    Caught: deleting the ``"MaxPoo3dConnection"`` entry from CONNECTION_SPECS.
    """
    missing = sorted(set(CONNECTION_CLASSES) - set(CONNECTION_SPECS))
    assert not missing, f"add source/target shapes for {missing} to CONNECTION_SPECS"


# --------------------------------------------------------------------------- #
# Learning rules on every learnable connection type                           #
# --------------------------------------------------------------------------- #

LEARNABLE = [
    "Connection",
    "SparseConnection",
    "LocalConnection",
    "LocalConnection1D",
    "LocalConnection2D",
    "LocalConnection3D",
    "Conv1dConnection",
    "Conv2dConnection",
    "Conv3dConnection",
]
DENSE_LIKE = {"Connection", "SparseConnection", "LocalConnection"}

# Which (rule, connection) pairs are supported. Every other pair must raise
# NotImplementedError when the connection is built. Changing support must change
# this table, so the change is visible in review.
SUPPORTED = {
    "NoOp": set(LEARNABLE),
    "PostPre": set(LEARNABLE),
    "WeightDependentPostPre": set(LEARNABLE),
    "Hebbian": set(LEARNABLE),
    "MSTDP": set(LEARNABLE),
    "MSTDPET": set(LEARNABLE),
    "DiehlAndCook": DENSE_LIKE,
    "Rmax": DENSE_LIKE,
}


@pytest.mark.parametrize("conn_name", LEARNABLE)
@pytest.mark.parametrize("rule_name", sorted(RULE_CLASSES))
def test_rule_on_connection(rule_name, conn_name):
    """Each rule runs on each supported connection and moves the weights.

    Caught: both ``self.connection.w +=`` lines removed from ``Hebbian._conv2d_connection_update``.
    """
    assert rule_name in SUPPORTED, f"add {rule_name} to SUPPORTED"
    rule = RULE_CLASSES[rule_name]
    if conn_name not in SUPPORTED[rule_name]:
        with pytest.raises(NotImplementedError):
            _build(conn_name, rule=rule)
        return
    batch = BATCH
    net, conn, src_shape = _build(conn_name, rule=rule, batch=batch)
    w0 = _weights(conn).clone()
    net.run(inputs={"in": _spikes(src_shape, batch)}, time=STEPS, reward=1.0)
    _check_layers(net, batch)
    w1 = _weights(conn)
    assert w1.shape == w0.shape
    assert torch.isfinite(w1).all()
    if rule_name == "NoOp":
        assert torch.equal(w1, w0)
    else:
        assert not torch.equal(w1, w0), "weights did not change"


@pytest.mark.parametrize("rule_name", sorted(MCC_RULE_CLASSES))
def test_mcc_rule_on_multicompartment_connection(rule_name):
    """Each MCC rule runs on a MulticompartmentConnection Weight and moves it.

    Caught: both ``addmm_`` updates removed from ``MCC_learning.Hebbian._connection_update``.
    """
    rule = MCC_RULE_CLASSES[rule_name]
    batch = BATCH
    net, conn, src_shape = _build(
        "MulticompartmentConnection", batch=batch, learning_rule=rule, nu=(1e-2, 1e-2)
    )
    w0 = _weights(conn).clone()
    net.run(inputs={"in": _spikes(src_shape, batch)}, time=STEPS, reward=1.0)
    _check_layers(net, batch)
    w1 = _weights(conn)
    assert w1.shape == w0.shape
    assert torch.isfinite(w1).all()
    if rule_name == "NoOp":
        assert torch.equal(w1, w0)
    else:
        assert not torch.equal(w1, w0), "weights did not change"


# --------------------------------------------------------------------------- #
# Models                                                                      #
# --------------------------------------------------------------------------- #

# name -> list of (constructor keyword arguments, batch size), one case each.
MODEL_SPECS = {
    "TwoLayerNetwork": [({"n_inpt": 16, "n_neurons": 8}, BATCH)],
    "DiehlAndCook2015": [
        ({"n_inpt": 16, "n_neurons": 8}, BATCH),
        # Batch 1 and no learning check: see KNOWN_BUGS (sparse-batch2, no-learning).
        ({"n_inpt": 16, "n_neurons": 8, "sparse": True, "batch_size": 1}, 1),
        (
            {
                "n_inpt": 16,
                "n_neurons": 8,
                "learning_rule": mcc_learning.DiehlAndCook,
                "learning_rule_kwargs": {"x_tar": 0.4},
            },
            BATCH,
        ),
    ],
    "DiehlAndCook2015v2": [({"n_inpt": 16, "n_neurons": 8}, BATCH)],
    "IncreasingInhibitionNetwork": [({"n_input": 16, "n_neurons": 9}, BATCH)],
    # Batch 1: it is built on LocalConnection (see KNOWN_BUGS).
    "LocallyConnectedNetwork": [
        (
            {
                "n_inpt": 64,
                "input_shape": [8, 8],
                "kernel_size": 4,
                "stride": 2,
                "n_filters": 2,
            },
            1,
        )
    ],
}
MODEL_NO_LEARNING_CHECK = {("DiehlAndCook2015", 1)}
MODEL_CASES = [
    (name, i)
    for name in sorted(MODEL_CLASSES)
    for i in range(len(MODEL_SPECS.get(name, [None])))
]


def _run_model(name, kwargs, batch, learn=True):
    torch.manual_seed(0)
    net = MODEL_CLASSES[name](**kwargs)
    net.train(learn)
    inputs = {
        k: _spikes(layer.shape, batch)
        for k, layer in net.layers.items()
        if isinstance(layer, nodes.AbstractInput)
    }
    assert inputs, "model has no Input layer"
    net.run(inputs=inputs, time=STEPS)
    return net


@pytest.mark.parametrize("name,case", MODEL_CASES)
def test_model_construct_and_run(name, case):
    """Every model in bindsnet.models builds, runs 10 steps and learns.

    Caught: ``update_rule=PostPre`` changed to ``update_rule=None`` in ``DiehlAndCook2015v2``.
    """
    assert name in MODEL_SPECS, f"add constructor arguments for {name} to MODEL_SPECS"
    kwargs, batch = MODEL_SPECS[name][case]
    net = _run_model(name, kwargs, batch)
    _check_layers(net, batch)
    # Same seed and input with learning off. Weight normalisation runs in both, so
    # any difference comes from the learning rule.
    frozen = _run_model(name, kwargs, batch, learn=False)
    learned = []
    for k, c in net.connections.items():
        w = _weights(c)
        assert torch.isfinite(w).all(), f"{k} weights not finite"
        if not torch.equal(w, _weights(frozen.connections[k])):
            learned.append(k)
    if (name, case) not in MODEL_NO_LEARNING_CHECK:
        assert learned, "no connection learned in 10 steps"


# --------------------------------------------------------------------------- #
# MulticompartmentConnection features                                         #
# --------------------------------------------------------------------------- #

FEATURE_CLASSES = {
    **_public_classes(tf, tf.AbstractFeature),
    **_public_classes(tf, tf.AbstractSubFeature),
}
SRC_N, TGT_N = 6, 4


def _rand(*shape):
    return torch.rand(*shape, generator=torch.Generator().manual_seed(3))


def _ann():
    return [torch.zeros(1, 1), torch.zeros(1, 1)]


# name -> function(weight feature) returning the pipeline to test.
FEATURE_PIPELINES = {
    "Weight": lambda w: [w],
    "Bias": lambda w: [w, tf.Bias("b", _rand(SRC_N, TGT_N), range=(-1.0, 1.0))],
    "Mask": lambda w: [w, tf.Mask("m", _rand(SRC_N, TGT_N) > 0.5)],
    "Probability": lambda w: [tf.Probability("p", _rand(SRC_N, TGT_N)), w],
    "Intensity": lambda w: [w, tf.Intensity("i", _rand(SRC_N, TGT_N), range=(0, 2))],
    "Degradation": lambda w: [
        w,
        tf.Degradation("d", _rand(SRC_N, TGT_N), degrade_function=lambda v: v / 2),
    ],
    "MeanField": lambda w: [tf.MeanField(), w],
    "AdaptationBaseSynapsHistory": lambda w: [
        w,
        tf.AdaptationBaseSynapsHistory("a", _rand(SRC_N, TGT_N), ann_values=_ann()),
    ],
    "AdaptationBaseOtherSynaps": lambda w: [
        w,
        tf.AdaptationBaseOtherSynaps("a", _rand(SRC_N, TGT_N), ann_values=_ann()),
    ],
    "Normalization": lambda w: [tf.Normalization("n", parent_feature=w), w],
    "Updating": lambda w: [tf.Updating("u", parent_feature=w), w],
}


FEATURE_BATCH1_ONLY = {"AdaptationBaseSynapsHistory", "AdaptationBaseOtherSynaps"}
SUB_FEATURES = {"Normalization", "Updating"}


@pytest.mark.parametrize("name", sorted(FEATURE_CLASSES))
def test_feature_in_pipeline(name):
    """Every topology feature runs inside a MulticompartmentConnection for 10 steps.

    Caught: ``return self.value`` changed to ``return self.value * float("nan")`` in ``Bias.compute``.
    """
    assert name in FEATURE_PIPELINES, f"add a pipeline for {name} to FEATURE_PIPELINES"
    torch.manual_seed(0)
    weight = tf.Weight("w", _rand(SRC_N, TGT_N) * 4, norm=8.0)
    pipeline = FEATURE_PIPELINES[name](weight)
    src = nodes.Input(n=SRC_N, traces=True)
    tgt = nodes.LIFNodes(n=TGT_N, traces=True)
    net = Network(dt=1.0, batch_size=BATCH)
    net.add_layer(src, name="in")
    net.add_layer(tgt, name="out")
    # Sub-features cannot be primed (KNOWN_BUGS): build without them, then set
    # the pipeline, as test_connections.py does.
    primable = [f for f in pipeline if isinstance(f, tf.AbstractFeature)]
    conn = topology.MulticompartmentConnection(
        source=src, target=tgt, device="cpu", pipeline=primable
    )
    conn.pipeline = pipeline
    net.add_connection(conn, source="in", target="out")
    batch = 1 if name in FEATURE_BATCH1_ONLY else BATCH
    if name in SUB_FEATURES:  # Network.run calls update(), which they lack
        s = _spikes((SRC_N,), batch)[0]
        net.layers["in"].set_batch_size(batch)
    else:
        net.run(inputs={"in": _spikes((SRC_N,), batch)}, time=STEPS)
        _check_layers(net, batch)
        s = src.s
    out = conn.compute(s)
    assert tuple(out.shape) == (batch, TGT_N)
    assert torch.isfinite(out).all()
    if name not in SUB_FEATURES:  # they also lack normalize() and reset
        conn.normalize()
        conn.reset_state_variables()


# --------------------------------------------------------------------------- #
# Monitors, reward functions and Network.run options                          #
# --------------------------------------------------------------------------- #

MONITOR_CLASSES = _public_classes(monitors, monitors.AbstractMonitor)
REWARD_CLASSES = _public_classes(reward, reward.AbstractReward)


def _small_net(**kwargs):
    torch.manual_seed(0)
    net = Network(dt=1.0, batch_size=BATCH, **kwargs)
    net.add_layer(nodes.Input(n=20, traces=True), name="in")
    net.add_layer(nodes.LIFNodes(n=10, traces=True, thresh=-60.0), name="out")
    conn = topology.Connection(
        net.layers["in"], net.layers["out"], update_rule=learning.MSTDP, nu=1e-2
    )
    net.add_connection(conn, source="in", target="out")
    return net, conn


@pytest.mark.parametrize("name", sorted(MONITOR_CLASSES))
def test_monitor_records_and_saves(name, tmp_path):
    """Every monitor records [time, batch, ...] tensors; NetworkMonitor saves both formats.

    Caught: ``[1:]`` removed from the fixed-time branch of ``NetworkMonitor.record`` (recording grows).
    """
    assert name in ("Monitor", "NetworkMonitor"), f"add a case for {name}"
    net, _ = _small_net()
    if name == "Monitor":
        mon = monitors.Monitor(net.layers["out"], ["s", "v"], time=STEPS)
        net.add_monitor(mon, name="m")
        spikes = _spikes((20,))
        net.run(inputs={"in": spikes}, time=STEPS, reward=1.0)
        assert tuple(mon.get("s").shape) == (STEPS, BATCH, 10)
        assert tuple(mon.get("v").shape) == (STEPS, BATCH, 10)
        assert torch.isfinite(mon.get("v")).all()
        return
    mon = monitors.NetworkMonitor(net, state_vars=["s", "v", "w"], time=STEPS)
    net.add_monitor(mon, name="m")
    spikes = _spikes((20,))
    net.run(inputs={"in": spikes}, time=STEPS, reward=1.0)
    rec = mon.get()
    assert torch.equal(rec["in"]["s"].bool(), spikes)
    assert tuple(rec["out"]["v"].shape) == (STEPS, BATCH, 10)
    assert tuple(rec[("in", "out")]["w"].shape) == (STEPS, 20, 10)
    mon.save(str(tmp_path / "rec.npz"), fmt="npz")
    mon.save(str(tmp_path / "rec.pt"), fmt="pickle")
    assert (tmp_path / "rec.npz").exists() and (tmp_path / "rec.pt").exists()
    # Without a fixed time the recording grows by one row per step.
    net, _ = _small_net()
    grow = monitors.NetworkMonitor(net, state_vars=["s", "w"])
    net.add_monitor(grow, name="g")
    net.run(inputs={"in": spikes}, time=STEPS, reward=1.0)
    assert tuple(grow.get()["out"]["s"].shape) == (STEPS, BATCH, 10)
    assert tuple(grow.get()[("in", "out")]["w"].shape) == (STEPS, 20, 10)
    grow.reset_state_variables()
    assert grow.get()["out"]["s"].numel() == 0


@pytest.mark.parametrize("name", sorted(REWARD_CLASSES))
def test_reward_fn_modulates_learning(name):
    """A reward_fn changes the reward the rule sees; update() moves its prediction.

    Caught: ``kwargs["reward"] = self.reward_fn.compute(**kwargs)`` removed from ``Network.run``.
    """
    plain, conn_a = _small_net()
    fn_net, conn_b = _small_net(reward_fn=REWARD_CLASSES[name])
    fn_net.reward_fn.update(accumulated_reward=torch.tensor(30.0), steps=10)
    spikes = _spikes((20,))
    plain.run(inputs={"in": spikes}, time=STEPS, reward=1.0)
    fn_net.run(inputs={"in": spikes}, time=STEPS, reward=1.0)
    assert not torch.equal(conn_a.w, conn_b.w)


def test_network_run_options():
    """Network.run honours injects_v, a_plus/a_minus (float and dict) and 1-D/2-D inputs.

    Caught: ``self.layers[l].v += inject_v`` removed from ``Network.run``.
    """
    # Voltage injection alone makes a silent layer spike.
    net, _ = _small_net()
    no_input = torch.zeros(STEPS, BATCH, 20, dtype=torch.bool)
    net.run(inputs={"in": no_input}, time=STEPS, reward=1.0)
    assert not net.layers["out"].s.any()
    net.reset_state_variables()
    m = monitors.Monitor(net.layers["out"], ["s"], time=STEPS)
    net.add_monitor(m, name="m")
    net.run(
        inputs={"in": no_input},
        time=STEPS,
        reward=1.0,
        injects_v={"out": torch.full((10,), 20.0)},
    )
    assert m.get("s").any()
    net.reset_state_variables()
    per_step = torch.full((STEPS, BATCH, 10), 20.0)
    net.run(
        inputs={"in": no_input}, time=STEPS, reward=1.0, injects_v={"out": per_step}
    )
    assert m.get("s").any()

    # a_plus = a_minus = 0 switches MSTDP off; a dict keyed by connection too.
    spikes = _spikes((20,))
    for a in (0.0, {("in", "out"): 0.0}):
        net, conn = _small_net()
        w0 = conn.w.clone()
        net.run(inputs={"in": spikes}, time=STEPS, reward=1.0, a_plus=a, a_minus=a)
        assert torch.equal(conn.w, w0)
    net, conn = _small_net()
    w0 = conn.w.clone()
    net.run(inputs={"in": spikes}, time=STEPS, reward=1.0)
    assert not torch.equal(conn.w, w0)

    # A [time, n] input becomes batch 1; an [n] input runs one step.
    net, _ = _small_net()
    net.run(inputs={"in": spikes[:, 0]}, time=STEPS, reward=1.0)
    assert net.batch_size == 1
    net.run(inputs={"in": spikes[0, 0]}, time=1, reward=1.0)
    assert tuple(net.layers["in"].s.shape) == (1, 20)


# --------------------------------------------------------------------------- #
# Known bugs on master, pinned as strict expected failures                    #
# --------------------------------------------------------------------------- #


def _local_connection_batch2():
    net, _, shape = _build("LocalConnection", batch=2)
    net.run(inputs={"in": _spikes(shape, 2)}, time=STEPS)


def _rmax_batch2():
    net, _, shape = _build("Connection", rule=learning.Rmax, batch=2)
    net.run(inputs={"in": _spikes(shape, 2)}, time=STEPS, reward=1.0)


def _sparse_pair(rule_name):
    def run():
        net, _, shape = _build(
            "SparseConnection",
            rule=RULE_CLASSES[rule_name],
            batch=1,
            wmin=0.0,
            wmax=1.0,
        )
        net.run(inputs={"in": _spikes(shape, 1)}, time=STEPS, reward=1.0)

    return run


def _maxpool_without_decay(name):
    def run():
        net, _, shape = _build(name)
        net.run(inputs={"in": _spikes(shape)}, time=STEPS)

    return run


def _maxpool_one_channel(name):
    dims = {"MaxPool1dConnection": 1, "MaxPool2dConnection": 2}.get(name, 3)
    spec = ((1,) + (4,) * dims, (1,) + (2,) * dims, {"kernel_size": 2, "stride": 2})

    def run():
        net, _, shape = _build(name, spec=spec, decay=0.5)
        net.run(inputs={"in": _spikes(shape)}, time=STEPS)

    return run


def _mean_field_batch_isolation():
    _, conn, _ = _build("MeanFieldConnection")
    s = torch.zeros(2, 20)
    s[1] = 1.0  # sample 0 silent, sample 1 all spiking
    out = conn.compute(s).expand(2, 10)
    assert torch.equal(out[0], torch.zeros(10)), "sample 1 leaked into sample 0"


def _mcc_mstdpet_batch2():
    net, _, shape = _build(
        "MulticompartmentConnection",
        learning_rule=mcc_learning.MSTDPET,
        nu=(1e-2, 1e-2),
    )
    net.run(inputs={"in": _spikes(shape)}, time=STEPS, reward=1.0)


def _mcc_tuple_shape():
    torch.manual_seed(0)
    src = nodes.Input(shape=(4,))
    tgt = nodes.LIFNodes(shape=(3,))
    net = Network(dt=1.0, batch_size=BATCH)
    net.add_layer(src, name="in")
    net.add_layer(tgt, name="out")
    weight = tf.Weight(name="w", value=torch.rand(4, 3))
    conn = topology.MulticompartmentConnection(
        source=src, target=tgt, device="cpu", pipeline=[weight]
    )
    net.add_connection(conn, source="in", target="out")
    net.run(inputs={"in": _spikes((4,))}, time=STEPS)


def _mean_field_feature():
    src, tgt = nodes.Input(n=SRC_N), nodes.LIFNodes(n=TGT_N)
    weight = tf.Weight("w", _rand(SRC_N, TGT_N))
    topology.MulticompartmentConnection(
        source=src, target=tgt, device="cpu", pipeline=[tf.MeanField(), weight]
    )


def _subfeature_in_constructor(cls):
    def run():
        src, tgt = nodes.Input(n=SRC_N), nodes.LIFNodes(n=TGT_N)
        weight = tf.Weight("w", _rand(SRC_N, TGT_N), norm=2.0)
        topology.MulticompartmentConnection(
            source=src,
            target=tgt,
            device="cpu",
            pipeline=[cls("sub", parent_feature=weight), weight],
        )

    return run


def _adaptation_batch2(cls):
    def run():
        src = nodes.Input(n=SRC_N, traces=True)
        tgt = nodes.LIFNodes(n=TGT_N, traces=True)
        net = Network(dt=1.0, batch_size=2)
        net.add_layer(src, name="in")
        net.add_layer(tgt, name="out")
        feat = cls("a", _rand(SRC_N, TGT_N), ann_values=_ann())
        conn = topology.MulticompartmentConnection(
            source=src, target=tgt, device="cpu", pipeline=[feat]
        )
        net.add_connection(conn, source="in", target="out")
        net.run(inputs={"in": _spikes((SRC_N,), 2)}, time=STEPS)

    return run


def _subfeature_network_run(cls):
    def run():
        src, tgt = nodes.Input(n=SRC_N), nodes.LIFNodes(n=TGT_N)
        net = Network(dt=1.0)
        net.add_layer(src, name="in")
        net.add_layer(tgt, name="out")
        weight = tf.Weight("w", _rand(SRC_N, TGT_N), norm=2.0)
        conn = topology.MulticompartmentConnection(
            source=src, target=tgt, device="cpu", pipeline=[weight]
        )
        conn.pipeline = [cls("sub", parent_feature=weight), weight]
        net.add_connection(conn, source="in", target="out")
        net.run(inputs={"in": _spikes((SRC_N,), 1)}, time=STEPS)

    return run


def _network_monitor_reset():
    net, _ = _small_net()
    mon = monitors.NetworkMonitor(net, state_vars=["s", "w"], time=STEPS)
    mon.reset_state_variables()


def _nodes_2d(name):
    return lambda: _run_nodes(name, NODE_SHAPE)


def _sparse_dc2015_learns():
    kwargs = {"n_inpt": 16, "n_neurons": 8, "batch_size": 1}
    kwargs.update(exc_thresh=-62.0, norm=None)  # spikes in 10 steps; no rescaling
    for sparse in (False, True):
        torch.manual_seed(0)
        w0 = _weights(
            MODEL_CLASSES["DiehlAndCook2015"](**kwargs).connections[("X", "Ae")]
        )
        net = _run_model("DiehlAndCook2015", {**kwargs, "sparse": sparse}, 1)
        w1 = _weights(net.connections[("X", "Ae")])
        assert not torch.equal(
            w1.reshape(w0.shape), w0
        ), f"sparse={sparse}: no learning"


def _sparse_dc2015_batch2():
    kwargs = {"n_inpt": 16, "n_neurons": 8, "sparse": True, "batch_size": 2}
    _run_model("DiehlAndCook2015", kwargs, 2)


# id -> (what is wrong and where, exception type, message fragment, reproducer).
KNOWN_BUGS = {
    "LocalConnection-batch2": (
        "LocalConnection.compute reshapes to target.shape without the batch "
        "dimension (topology.py, `a_post.view(*self.target.shape)`)",
        RuntimeError,
        "is invalid for input of size",
        _local_connection_batch2,
    ),
    "MeanFieldConnection-batch-mixing": (
        "MeanFieldConnection.compute returns s.float().mean() * w, the mean over "
        "the whole batch, so one sample's input depends on the other samples",
        AssertionError,
        "leaked",
        _mean_field_batch_isolation,
    ),
    "MeanField-feature": (
        "topology_features.MeanField.__init__ does not call the parent constructor, "
        "so the feature has no name and MulticompartmentConnection raises "
        "AttributeError when it indexes the pipeline (topology.py)",
        AttributeError,
        "has no attribute 'name'",
        _mean_field_feature,
    ),
    **{
        f"{cls.__name__}-in-constructor": (
            f"{cls.__name__} (a sub-feature) in the pipeline passed to "
            "MulticompartmentConnection raises AttributeError, because the "
            "constructor calls prime_feature, which sub-features do not have",
            AttributeError,
            "prime_feature",
            _subfeature_in_constructor(cls),
        )
        for cls in (tf.Normalization, tf.Updating)
    },
    **{
        f"{cls.__name__}-batch2": (
            f"{cls.__name__} fails for batch > 1 with an expand size mismatch "
            "(topology_features.py)",
            RuntimeError,
            "expanded size",
            _adaptation_batch2(cls),
        )
        for cls in (tf.AdaptationBaseSynapsHistory, tf.AdaptationBaseOtherSynaps)
    },
    **{
        f"{cls.__name__}-network-run": (
            f"{cls.__name__} in a pipeline breaks Network.run: "
            "MulticompartmentConnection.update calls update() on every pipeline "
            "entry and sub-features have none (nor normalize() or reset_state_variables())",
            AttributeError,
            "has no attribute 'update'",
            _subfeature_network_run(cls),
        )
        for cls in (tf.Normalization, tf.Updating)
    },
    "NetworkMonitor-reset-with-time": (
        "NetworkMonitor.reset_state_variables with time set looks connections up "
        "in network.layers instead of network.connections (monitors.py, last "
        "loop), so it raises KeyError; Network.reset_state_variables and a batch "
        "size change in Network.run call it",
        KeyError,
        "in",
        _network_monitor_reset,
    ),
    "MulticompartmentConnection-tuple-shape": (
        "MulticompartmentConnection.compute builds [batch] + target.shape, which "
        "raises TypeError when the target Nodes got its shape as a tuple",
        TypeError,
        "can only concatenate list",
        _mcc_tuple_shape,
    ),
    "Rmax-batch2": (
        "Rmax._connection_update flattens the batch with view(-1), so batch > 1 "
        "gives a size mismatch (learning.py)",
        RuntimeError,
        "must match the size",
        _rmax_batch2,
    ),
    "WeightDependentPostPre-SparseConnection": (
        "WeightDependentPostPre needs finite wmin and wmax; with them the update "
        "does add(sparse, dense), which torch does not support (learning.py)",
        RuntimeError,
        r"add\(sparse, dense\)",
        _sparse_pair("WeightDependentPostPre"),
    ),
    "DiehlAndCook-SparseConnection": (
        "DiehlAndCook needs a finite wmax; with it the update adds a dense "
        "tensor to the sparse weights and fails (learning.py)",
        RuntimeError,
        "dense-sparse addition",
        _sparse_pair("DiehlAndCook"),
    ),
    "MCC_learning.MSTDPET-batch2": (
        "MCC_learning.MSTDPET flattens source and target spikes with view(-1), so "
        "batch > 1 gives a size mismatch (MCC_learning.py)",
        RuntimeError,
        "must match the size",
        _mcc_mstdpet_batch2,
    ),
    "DiehlAndCook2015-sparse-no-learning": (
        "DiehlAndCook2015(sparse=True) never changes its input weights: "
        "MCC_LearningRule.update clamps a sparse feature by building a new tensor "
        "and binds it to rule.feature_value only (MCC_learning.py, the "
        "to_dense().clamp_().to_sparse() line), so Weight.value, which compute() "
        "reads, keeps its initial values. The dense model learns in the same run",
        AssertionError,
        "sparse=True: no learning",
        _sparse_dc2015_learns,
    ),
    "DiehlAndCook2015-sparse-batch2": (
        "DiehlAndCook2015(sparse=True) with batch_size > 1: "
        "MulticompartmentConnection.compute views the sparse feature as "
        "[source.n, target.n] (topology.py)",
        RuntimeError,
        "is invalid for input of size",
        _sparse_dc2015_batch2,
    ),
    **{
        f"{name}-no-decay": (
            f"{name} has no default decay; compute() multiplies None (topology.py)",
            TypeError,
            "NoneType",
            _maxpool_without_decay(name),
        )
        for name in sorted(MAXPOOL)
    },
    **{
        f"{name}-one-channel": (
            f"{name}.compute calls s.float().squeeze(), which drops a channel "
            "dimension of size 1, so the firing-rate update has the wrong shape "
            "for batch > 1 (topology.py)",
            RuntimeError,
            "doesn't match the broadcast shape",
            _maxpool_one_channel(name),
        )
        for name in sorted(MAXPOOL)
    },
}


# Fixed for 0.3.5. Each reproducer must now run without error, so the fix stays.
FIXED = [
    "LocalConnection-batch2",
    *(f"{name}-{case}" for name in MAXPOOL for case in ("no-decay", "one-channel")),
    "NetworkMonitor-reset-with-time",
    "MulticompartmentConnection-tuple-shape",
    "DiehlAndCook2015-sparse-no-learning",
    "DiehlAndCook2015-sparse-batch2",
    "Rmax-batch2",
    "MCC_learning.MSTDPET-batch2",
    "MeanFieldConnection-batch-mixing",
    "MeanField-feature",
    "WeightDependentPostPre-SparseConnection",
    "DiehlAndCook-SparseConnection",
    *(
        f"{c}-{case}"
        for c in ("Normalization", "Updating")
        for case in ("in-constructor", "network-run")
    ),
    *(
        f"{c}-batch2"
        for c in ("AdaptationBaseSynapsHistory", "AdaptationBaseOtherSynaps")
    ),
]
FIXED_BUGS = {bug_id: KNOWN_BUGS.pop(bug_id) for bug_id in FIXED}


@pytest.mark.parametrize("bug_id", sorted(FIXED_BUGS))
def test_fixed_bug_stays_fixed(bug_id):
    """Each former KNOWN_BUGS reproducer runs cleanly.

    Caught: restoring ``s.float().squeeze()`` in MaxPool2dConnection.compute fails
    MaxPool2dConnection-one-channel.
    """
    FIXED_BUGS[bug_id][3]()


def test_sparse_dc2015_learns_like_dense():
    """With one sample there is one weight copy, so sparse and dense weights learn
    identically (no normalisation; with it, sparse sums in another order).

    Caught: dropping the in-place copy_ in MCC_LearningRule.update (the sparse
    clamp) leaves the sparse weights at their initial values.
    """
    kwargs = {"n_inpt": 16, "n_neurons": 8, "batch_size": 1}
    kwargs.update(exc_thresh=-62.0, norm=None)
    learned = []
    for sparse in (False, True):
        net = _run_model("DiehlAndCook2015", {**kwargs, "sparse": sparse}, 1)
        w = _weights(net.connections[("X", "Ae")])
        learned.append((w.to_dense() if w.is_sparse else w).reshape(16, 8))
    assert torch.equal(learned[0], learned[1])


def test_mean_field_feature_keeps_samples_apart():
    """A MeanField feature uses each sample's own mean: the batch-2 output equals
    each sample computed alone.

    Caught: MeanField.compute taking s.float().mean() over the whole batch.
    """
    src, tgt = nodes.Input(n=SRC_N), nodes.LIFNodes(n=TGT_N)
    weight = tf.Weight("w", _rand(SRC_N, TGT_N))
    conn = topology.MulticompartmentConnection(
        source=src, target=tgt, device="cpu", pipeline=[tf.MeanField(), weight]
    )
    s = torch.tensor([[1.0] * SRC_N, [1.0] + [0.0] * (SRC_N - 1)])
    both = conn.compute(s)
    alone = torch.cat([conn.compute(s[i : i + 1]) for i in range(2)])
    assert not torch.equal(both[0], both[1])
    assert torch.equal(both, alone)


@pytest.mark.parametrize(
    "rule_name", ["PostPre", "WeightDependentPostPre", "DiehlAndCook"]
)
def test_sparse_connection_learns_like_dense(rule_name):
    """Rules on a bounded SparseConnection give the dense result: weight-dependent
    factors use a dense view of w and the bounds are applied in place.

    Caught: the sparse branch of LearningRule.update leaving w unbounded (PostPre
    fails; the weight-dependent rules never reach the bounds).
    """
    w0 = torch.rand(SRC_N, TGT_N, generator=torch.Generator().manual_seed(3))
    inputs = {"in": _spikes((SRC_N,), 1)}
    learned = []
    for cls, w in (
        (topology.Connection, w0),
        (topology.SparseConnection, w0.to_sparse()),
    ):
        torch.manual_seed(0)
        src = nodes.Input(n=SRC_N, traces=True)
        tgt = nodes.LIFNodes(n=TGT_N, traces=True, thresh=-60.0)
        net = Network(dt=1.0, batch_size=1)
        net.add_layer(src, name="in")
        net.add_layer(tgt, name="out")
        rule = RULE_CLASSES[rule_name]
        conn = cls(
            src, tgt, w=w.clone(), update_rule=rule, nu=(0.5, 0.5), wmin=0.0, wmax=1.0
        )
        net.add_connection(conn, source="in", target="out")
        net.run(inputs=inputs, time=STEPS)
        learned.append(conn.w.to_dense() if conn.w.is_sparse else conn.w)
    assert not torch.equal(learned[0], w0)
    assert learned[0].min() >= 0.0 and learned[0].max() <= 1.0
    assert torch.equal(learned[0], learned[1])


@pytest.mark.parametrize("bug_id", sorted(KNOWN_BUGS))
def test_known_bug_still_fails(bug_id):
    """Pins each known crash, with its error, so a fix is noticed.

    Caught: a ``return`` added as the first line of ``_rmax_batch2`` fails with DID NOT RAISE.
    """
    reason, exc_type, match, reproduce = KNOWN_BUGS[bug_id]
    with pytest.raises(exc_type, match=match):
        reproduce()
    pytest.xfail(reason)
