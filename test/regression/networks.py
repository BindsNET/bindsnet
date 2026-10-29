"""
Sixteen small seeded networks used by the golden-output and invariant tests.

Ported from the review tool's ``canaries.py`` (builders and the ``spikes`` input
generator only). Each builder takes ``dt`` and returns ``(network, make_inputs)``;
``make_inputs(steps, batch, gen)`` returns the ``inputs`` dict for ``Network.run``,
drawing every random number from the private generator ``gen``.

Determinism: ``build`` calls ``torch.manual_seed(0)`` before building (some
builders draw initial weights from the global generator), and ``inputs``
reseeds the global generator before every trial, because the network itself may
draw random numbers while it runs (``DiehlAndCookNodes`` with ``one_spike`` uses
``torch.multinomial``; ``SRM0Nodes`` fires stochastically).

Some networks run at dt = 0.5 ms: at dt = 1 a dropped or misplaced ``dt`` factor
multiplies by one and is invisible.
"""

import io
import lzma
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import torch

from bindsnet.network.monitors import Monitor


@dataclass
class Spec:
    build: Callable
    trials: List[Tuple[int, int, int]]  # (batch, steps, input seed)
    norm: bool = False  # a connection has weight normalisation
    run_kwargs: Dict = field(default_factory=dict)
    dt: float = 1.0


SPECS: Dict[str, Spec] = {}


def network(name, trials, norm=False, run_kwargs=None, dt=1.0):
    """Register a builder ``fn(dt)`` under ``name``."""

    def wrap(fn):
        SPECS[name] = Spec(fn, trials, norm, run_kwargs or {}, dt)
        return fn

    return wrap


def spikes(gen, steps, batch, shape, rate):
    """Bernoulli spike train of shape [steps, batch, *shape] from a private generator."""
    return (torch.rand(steps, batch, *shape, generator=gen) < rate).byte()


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #


def _dc2015(batch, dt):
    from bindsnet.models import DiehlAndCook2015

    net = DiehlAndCook2015(
        n_inpt=196, n_neurons=40, batch_size=batch, inpt_shape=(1, 14, 14), dt=dt
    )
    return net, lambda T, b, g: {"X": spikes(g, T, b, (1, 14, 14), 0.25)}


@network("dc2015_b1", trials=[(1, 60, 1), (1, 40, 2)], norm=True)
def dc2015_b1(dt):
    return _dc2015(1, dt)


@network("dc2015_b16", trials=[(16, 60, 1), (16, 40, 2)], norm=True)
def dc2015_b16(dt):
    return _dc2015(16, dt)


def _dense_postpre(batch, dt):
    from bindsnet.learning import PostPre
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection

    net = Network(dt=dt, batch_size=batch)
    net.add_layer(Input(n=100, traces=True), "in")
    net.add_layer(LIFNodes(n=50, traces=True), "out")
    net.add_connection(
        Connection(
            net.layers["in"],
            net.layers["out"],
            nu=(1e-3, 1e-2),
            update_rule=PostPre,
            wmin=0.0,
            wmax=1.0,
            norm=20.0,
        ),
        "in",
        "out",
    )
    return net, lambda T, b, g: {"in": spikes(g, T, b, (100,), 0.04)}


@network("conn_postpre_b1", trials=[(1, 60, 1), (1, 40, 2)], norm=True)
def conn_postpre_b1(dt):
    return _dense_postpre(1, dt)


@network("conn_postpre_b8", trials=[(8, 60, 1), (8, 40, 2)], norm=True)
def conn_postpre_b8(dt):
    return _dense_postpre(8, dt)


@network("conn_postpre_dt05", trials=[(1, 80, 1), (1, 60, 2)], norm=True, dt=0.5)
def conn_postpre_dt05(dt):
    return _dense_postpre(1, dt)


@network("local2d_postpre", trials=[(1, 60, 1), (1, 40, 2)])
def local2d_postpre(dt):
    from bindsnet.learning import PostPre
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import LocalConnection2D

    net = Network(dt=dt)
    net.add_layer(Input(shape=[1, 20, 20], traces=True), "in")
    net.add_layer(LIFNodes(shape=[8, 4, 4], traces=True), "out")
    net.add_connection(
        LocalConnection2D(
            net.layers["in"],
            net.layers["out"],
            kernel_size=8,
            stride=4,
            n_filters=8,
            nu=(1e-3, 1e-2),
            update_rule=PostPre,
            wmin=0.0,
            wmax=1.0,
        ),
        "in",
        "out",
    )
    return net, lambda T, b, g: {"in": spikes(g, T, b, (1, 20, 20), 0.06)}


def _mstdp(dt):
    from bindsnet.learning import MSTDP, MSTDPET
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection

    net = Network(dt=dt)
    net.add_layer(Input(n=100), "in")
    net.add_layer(LIFNodes(n=30), "out1")
    net.add_layer(LIFNodes(n=30), "out2")
    for target, rule in (("out1", MSTDP), ("out2", MSTDPET)):
        net.add_connection(
            Connection(
                net.layers["in"],
                net.layers[target],
                nu=1e-2,
                update_rule=rule,
                wmin=0.0,
                wmax=1.0,
            ),
            "in",
            target,
        )
    return net, lambda T, b, g: {"in": spikes(g, T, b, (100,), 0.04)}


@network("mstdp_mstdpet", trials=[(1, 60, 1), (1, 40, 2)], run_kwargs={"reward": 1.0})
def mstdp_mstdpet(dt):
    return _mstdp(dt)


# MSTDPET keeps Florian's dt (eq. 2.7); at dt = 1 dropping it changes nothing.
@network(
    "mstdp_mstdpet_dt05",
    trials=[(1, 80, 1), (1, 60, 2)],
    run_kwargs={"reward": 1.0},
    dt=0.5,
)
def mstdp_mstdpet_dt05(dt):
    return _mstdp(dt)


# The third trial changes the batch size (Network.run resizes the layers).
@network("csrm_window", trials=[(1, 30, 1), (1, 20, 2), (3, 20, 3)])
def csrm_window(dt):
    from bindsnet.network import Network
    from bindsnet.network.nodes import CSRMNodes, Input
    from bindsnet.network.topology import Connection

    net = Network(dt=dt)
    net.add_layer(Input(n=20), "in")
    net.add_layer(CSRMNodes(n=10, res_window_size=5, ref_window_size=3), "out")
    net.add_connection(
        Connection(net.layers["in"], net.layers["out"], w=4.0 * torch.rand(20, 10)),
        "in",
        "out",
    )
    return net, lambda T, b, g: {"in": spikes(g, T, b, (20,), 0.3).float()}


def _meanfield(dt, **mf_kwargs):
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import MeanFieldConnection

    net = Network(dt=dt)
    net.add_layer(Input(n=20), "in")
    net.add_layer(LIFNodes(n=10), "out")
    net.add_connection(
        MeanFieldConnection(net.layers["in"], net.layers["out"], **mf_kwargs),
        "in",
        "out",
    )
    return net, lambda T, b, g: {"in": spikes(g, T, b, (20,), 0.3)}


# weight_decay reaches the learning rule only if it is forwarded by name (#816).
@network("meanfield_decay", trials=[(1, 60, 1), (1, 40, 2)])
def meanfield_decay(dt):
    return _meanfield(dt, w=torch.full((10,), 20.0), weight_decay=0.01)


@network("meanfield_norm", trials=[(1, 60, 1), (1, 40, 2)], norm=True)
def meanfield_norm(dt):
    return _meanfield(dt, w=torch.linspace(5.0, 25.0, 10), norm=100.0)


# bindsnet's own encoders; scalar nu, clamp, unclamp and injects_v are
# Network.run options that no other network passes.
def _clamps():
    on = torch.zeros(20, dtype=torch.bool)
    off = torch.zeros(20, dtype=torch.bool)
    on[:2], off[2:4] = True, True
    return {
        "clamp": {"out": on},
        "unclamp": {"out": off},
        "injects_v": {"out": torch.full((20,), 0.5)},
    }


def _encoders(dt):
    from bindsnet.encoding import bernoulli, poisson
    from bindsnet.learning import PostPre
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection

    net = Network(dt=dt)
    net.add_layer(Input(n=64, traces=True), "in_p")
    net.add_layer(Input(n=32, traces=True), "in_b")
    net.add_layer(LIFNodes(n=20, traces=True), "out")
    for src in ("in_p", "in_b"):
        net.add_connection(
            Connection(
                net.layers[src],
                net.layers["out"],
                nu=1e-2,  # one number: LearningRule expands it to (pre, post)
                update_rule=PostPre,
                wmin=0.0,
                wmax=1.0,
            ),
            src,
            "out",
        )

    def inputs(T, b, g):
        rates = 64.0 * torch.rand(b, 64, generator=g)  # Hz
        probs = torch.rand(b, 32, generator=g)
        return {
            "in_p": poisson(datum=rates, time=T * dt, dt=dt),
            "in_b": bernoulli(datum=probs, time=T * dt, dt=dt, max_prob=0.3),
        }

    return net, inputs


@network("encode_clamp", trials=[(1, 60, 1), (1, 40, 2)], run_kwargs=_clamps)
def encode_clamp(dt):
    return _encoders(dt)


@network(
    "encode_clamp_dt05", trials=[(1, 80, 1), (1, 60, 2)], run_kwargs=_clamps, dt=0.5
)
def encode_clamp_dt05(dt):
    return _encoders(dt)


# Conv2dConnection, IFNodes and IzhikevichNodes.
@network("conv2d_if_izh", trials=[(1, 60, 1), (1, 40, 2)])
def conv2d_if_izh(dt):
    from bindsnet.learning import PostPre
    from bindsnet.network import Network
    from bindsnet.network.nodes import IFNodes, Input, IzhikevichNodes
    from bindsnet.network.topology import Connection, Conv2dConnection

    net = Network(dt=dt)
    net.add_layer(Input(shape=[1, 12, 12], traces=True), "in")
    net.add_layer(IFNodes(shape=[4, 8, 8], traces=True, thresh=-60.0), "conv")
    net.add_layer(IzhikevichNodes(n=16), "izh")
    net.add_connection(
        Conv2dConnection(
            net.layers["in"],
            net.layers["conv"],
            kernel_size=5,
            nu=(1e-3, 1e-2),
            update_rule=PostPre,
            wmin=0.0,
            wmax=1.0,
        ),
        "in",
        "conv",
    )
    net.add_connection(
        Connection(net.layers["conv"], net.layers["izh"], w=0.5 * torch.rand(256, 16)),
        "conv",
        "izh",
    )
    return net, lambda T, b, g: {"in": spikes(g, T, b, (1, 12, 12), 0.2)}


# Hebbian, WeightDependentPostPre (with AdaptiveLIFNodes) and Rmax (with SRM0Nodes,
# which fire stochastically from the global generator, reseeded per trial).
def _rules(dt):
    from bindsnet.learning import Hebbian, Rmax, WeightDependentPostPre
    from bindsnet.network import Network
    from bindsnet.network.nodes import AdaptiveLIFNodes, Input, LIFNodes, SRM0Nodes
    from bindsnet.network.topology import Connection

    net = Network(dt=dt)
    net.add_layer(Input(n=50, traces=True), "in")
    net.add_layer(Input(n=50, traces=True, traces_additive=True), "in_add")
    net.add_layer(AdaptiveLIFNodes(n=10, traces=True), "alif")
    net.add_layer(LIFNodes(n=10, traces=True), "heb")
    net.add_layer(SRM0Nodes(n=10, traces=True), "srm")
    for src, tgt, rule, nu in (
        ("in", "alif", WeightDependentPostPre, (1e-3, 1e-2)),
        ("in", "heb", Hebbian, (1e-4, 1e-4)),
        ("in_add", "srm", Rmax, 1e-2),
    ):
        net.add_connection(
            Connection(
                net.layers[src],
                net.layers[tgt],
                nu=nu,
                update_rule=rule,
                wmin=0.0,
                wmax=1.0,
                w=0.3 * torch.rand(50, 10),
            ),
            src,
            tgt,
        )
    return net, lambda T, b, g: {
        "in": spikes(g, T, b, (50,), 0.1),
        "in_add": spikes(g, T, b, (50,), 0.1),
    }


@network("rules_mix", trials=[(1, 60, 1), (1, 40, 2)], run_kwargs={"reward": 1.0})
def rules_mix(dt):
    return _rules(dt)


@network(
    "rules_mix_dt05",
    trials=[(1, 80, 1), (1, 60, 2)],
    run_kwargs={"reward": 1.0},
    dt=0.5,
)
def rules_mix_dt05(dt):
    return _rules(dt)


# --------------------------------------------------------------------------- #
# Running and recording
# --------------------------------------------------------------------------- #

PREFIX = "_reg_"


def deterministic():
    """CPU determinism settings required for exact comparisons (see
    ``cpu_pins.py``). Fails when the capability pin was applied too late."""
    capability = torch.backends.cpu.get_cpu_capability()
    if capability != "AVX2":
        raise RuntimeError(
            f"PyTorch CPU capability is {capability}, not AVX2: run through pytest "
            "(test/conftest.py) or regen.py, which pin it before torch starts"
        )
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    torch.backends.mkldnn.enabled = False


def run_kwargs(spec):
    kw = spec.run_kwargs
    return kw() if callable(kw) else dict(kw)


def build(name, learning=True, dt=None):
    """Seed, build network ``name``, attach monitors on every layer.

    Monitors record ``s``, ``v`` (when present), ``x`` (when the layer keeps
    traces) and ``theta`` (when present) at every step.
    """
    spec = SPECS[name]
    torch.manual_seed(0)
    net, make = spec.build(spec.dt if dt is None else dt)
    net.train(learning)
    for lname, layer in list(net.layers.items()):
        names = ["s"]
        for var in ("v", "theta"):
            if isinstance(getattr(layer, var, None), torch.Tensor):
                names.append(var)
        if getattr(layer, "traces", False):
            names.append("x")
        net.add_monitor(Monitor(layer, names), name=PREFIX + lname)
    return net, make


def inputs(make, steps, batch, seed):
    """Inputs from a private generator; also reseeds the global generator."""
    torch.manual_seed(1000 + seed)
    gen = torch.Generator().manual_seed(seed)
    return make(steps, batch, gen)


def run(net, spec, x, extra=None):
    """One ``Network.run`` call over all steps of the inputs ``x``."""
    steps = next(iter(x.values())).size(0)
    kw = run_kwargs(spec)
    kw.update(extra or {})
    net.run(inputs=x, time=steps * spec.dt, **kw)


def feature_tensors(net):
    """Tensors held by each connection's pipeline features (``value``, ``range``...)."""
    out = {}
    for (src, tgt), conn in net.connections.items():
        for i, feature in enumerate(getattr(conn, "pipeline", None) or []):
            for attr, val in sorted(vars(feature).items()):
                if isinstance(val, torch.Tensor) and not attr.startswith("_"):
                    t = val.to_dense() if val.is_sparse else val
                    out[f"{src}_to_{tgt}.{i}.{attr}"] = t
    return out


def collect(net, prefix=""):
    """Every monitored recording (emptying the monitors) and the state_dict."""
    out = {}
    for mname, mon in net.monitors.items():
        if mname.startswith(PREFIX):
            for var in mon.state_vars:
                t = mon.get(var).detach().cpu()
                if var == "s":
                    t = t.bool()
                out[f"{prefix}{mname[len(PREFIX):]}.{var}"] = t
    # MulticompartmentConnection keeps its weights in pipeline features, which are
    # not submodules, so they are missing from the state_dict; record them here.
    for key, t in feature_tensors(net).items():
        out[f"{prefix}feature.{key}"] = t.detach().cpu().clone()
    seen = set()  # layers also appear under each connection's source and target
    for key, t in net.state_dict().items():
        t = t.detach()
        if t.is_sparse:
            t = t.to_dense()
        ident = (t.data_ptr(), tuple(t.shape), t.dtype)
        if t.numel() and ident in seen:
            continue
        seen.add(ident)
        out[f"{prefix}state.{key}"] = t.cpu().clone()
    return out


def run_standard(name):
    """Build, run every trial (reset between trials), record after each trial.

    Returns ``(tensors, errors, margins)``. A trial that raises is recorded as
    ``errors["trial<i>"] = "<type>: <message>"``; what its monitors and the
    state_dict hold at that moment is still recorded, and later trials are not
    run. ``margins`` is the smallest threshold margin per layer over all trials
    (see ``threshold_margins``).
    """
    spec = SPECS[name]
    net, make = build(name)
    out, errors, margins = {}, {}, {}
    for i, (batch, steps, seed) in enumerate(spec.trials):
        if i:
            net.reset_state_variables()
        try:
            run(net, spec, inputs(make, steps, batch, seed))
        except Exception as e:  # noqa: BLE001 - a crash is a result to pin
            errors[f"trial{i}"] = f"{type(e).__name__}: {e}"
        rec = collect(net, f"trial{i}.")
        for lname, m in threshold_margins(net, rec, f"trial{i}.").items():
            margins[lname] = min(m, margins.get(lname, float("inf")))
        out.update(rec)
        if errors:
            break
    return out, errors, margins


def threshold_margins(net, rec, prefix):
    """Smallest distance between voltage and threshold at the spike check.

    For every layer with ``v`` and ``thresh`` (``SRM0Nodes`` excluded: it fires
    stochastically, not by threshold), over all steps and neurons where the
    recorded ``v`` is the value that was compared with the threshold: steps where
    the neuron did not spike (spiking neurons are reset before recording), plus
    the spiking steps of ``IzhikevichNodes`` and ``CSRMNodes``, which do not reset
    on the step they spike. The adaptive threshold ``theta`` is taken back to its
    value at the check by removing that step's ``theta_plus * s.sum(0)``.
    """
    from bindsnet.network.nodes import CSRMNodes, IzhikevichNodes, SRM0Nodes

    margins = {}
    for lname, layer in net.layers.items():
        if isinstance(layer, SRM0Nodes) or not hasattr(layer, "thresh"):
            continue
        key = f"{prefix}{lname}."
        if key + "v" not in rec:
            continue
        v, s = rec[key + "v"].double(), rec[key + "s"]
        if not v.numel():  # the trial crashed before its first step
            continue
        thresh = torch.as_tensor(layer.thresh).double()
        if key + "theta" in rec:
            theta = rec[key + "theta"].double()
            if layer.learning:
                theta = theta - layer.theta_plus.double() * s.double().sum(1)
            thresh = thresh + theta.unsqueeze(1)
        no_reset = isinstance(layer, (IzhikevichNodes, CSRMNodes))
        mask = torch.ones_like(s) if no_reset else ~s
        d = (v - thresh).abs()[mask]
        margins[lname] = float(d.min()) if d.numel() else float("inf")
    return margins


# --------------------------------------------------------------------------- #
# Golden files
# --------------------------------------------------------------------------- #

GOLDEN_DIR = Path(__file__).resolve().parent / "golden"
MARGIN = 1e-5  # smallest allowed distance between v and threshold (step 6)
FLOAT_ATOL = 1e-6  # floats may differ at about 1e-8 across CPUs; spikes may not


def golden_path(name):
    return GOLDEN_DIR / f"{name}.pt.xz"


def save_golden(name, payload):
    """Write ``payload`` with ``torch.save``, compressed with xz (lzma)."""
    buf = io.BytesIO()
    torch.save(payload, buf)
    GOLDEN_DIR.mkdir(exist_ok=True)
    golden_path(name).write_bytes(lzma.compress(buf.getvalue(), preset=9))


def load_golden(name):
    raw = lzma.decompress(golden_path(name).read_bytes())
    return torch.load(io.BytesIO(raw), weights_only=True)


def compare(expected, actual):
    """List of human-readable differences between two ``{key: tensor}`` dicts.

    Spikes and every other non-float tensor: ``torch.equal``. Floats:
    ``torch.allclose(atol=FLOAT_ATOL, rtol=0)``.
    """
    problems = []
    for key in sorted(set(expected) - set(actual)):
        problems.append(f"{key}: missing from this run")
    for key in sorted(set(actual) - set(expected)):
        problems.append(f"{key}: not in the golden file")
    for key in sorted(set(expected) & set(actual)):
        e, a = expected[key], actual[key]
        if e.shape != a.shape or e.dtype != a.dtype:
            problems.append(
                f"{key}: golden {e.dtype}{list(e.shape)}, now {a.dtype}{list(a.shape)}"
            )
        elif e.is_floating_point():
            if not torch.allclose(e, a, atol=FLOAT_ATOL, rtol=0, equal_nan=True):
                diff = (e - a).abs().nan_to_num(float("inf")).max().item()
                problems.append(f"{key}: max abs diff {diff:.3e} > {FLOAT_ATOL}")
        elif not torch.equal(e, a):
            n = int((e != a).sum())
            problems.append(f"{key}: {n} of {e.numel()} elements differ (exact)")
    return problems
