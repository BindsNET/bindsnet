"""
Invariant tests: the code compared with itself, so they are exact on any CPU and
need no golden file. Parametrised over the sixteen networks in ``networks.py``.

Each test's docstring names the deliberate wrong change it was shown to catch
(applied, run, reverted on 2026-10-24).
"""

import io

import networks as N
import pytest
import torch

from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import SRM0Nodes

NAMES = list(N.SPECS)
SEED_A, SEED_B = 11, 12  # input seeds of the two trials used below


def _first_trial(name):
    spec = N.SPECS[name]
    batch, steps, _ = spec.trials[0]
    return spec, batch, steps


def _disable_norm(net):
    """Turn off weight normalisation (``Network.run`` renormalises after every call)."""
    for conn in net.connections.values():
        if hasattr(conn, "norm"):
            conn.norm = None
        for feature in getattr(conn, "pipeline", None) or []:
            if hasattr(feature, "norm"):
                feature.norm = None


def _match_batch(net, batch):
    """Resize every layer to ``batch`` the way ``Network.run`` does on its first call
    (``DiehlAndCook2015`` builds its layers with batch 1 whatever ``batch_size`` is)."""
    if net.batch_size != batch:
        net.batch_size = batch
        for layer in net.layers.values():
            layer.set_batch_size(batch)


def _differences(a, b):
    """Keys whose tensors are not ``torch.equal`` (or exist on one side only)."""
    keys = sorted(set(a) | set(b))
    return [
        k
        for k in keys
        if k not in a
        or k not in b
        or a[k].shape != b[k].shape
        or not torch.equal(a[k], b[k])
    ]


def _assert_equal(a, b, what):
    bad = _differences(a, b)
    assert not bad, f"{what}: {len(bad)} tensors differ: {bad[:10]}"


# --------------------------------------------------------------------------- #
# 1. run(T) == run(T/2) twice
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
def test_split_run_equals_single_run(name):
    """run(T) equals two run(T/2) calls (weight normalisation turned off).
    Caught: Network.run resetting every layer at the start of each call."""
    spec, batch, steps = _first_trial(name)

    def go(split):
        net, make = N.build(name)
        _disable_norm(net)
        x = N.inputs(make, steps, batch, SEED_A)
        parts = (
            [
                {k: v[: steps // 2] for k, v in x.items()},
                {k: v[steps // 2 :] for k, v in x.items()},
            ]
            if split
            else [x]
        )
        for part in parts:
            N.run(net, spec, part)
        return N.collect(net)

    _assert_equal(go(False), go(True), f"{name} run(T) vs 2 x run(T/2)")


# --------------------------------------------------------------------------- #
# 2 and 3. reset_state_variables() equals a fresh network
# --------------------------------------------------------------------------- #


def _reset_path(name, learning):
    spec, batch, steps = _first_trial(name)
    net, make = N.build(name, learning=learning)
    N.run(net, spec, N.inputs(make, steps, batch, SEED_A))
    net.reset_state_variables()
    N.run(net, spec, N.inputs(make, steps, batch, SEED_B))
    return N.collect(net), net


@pytest.mark.parametrize("name", NAMES)
def test_reset_equals_fresh_learning_off(name):
    """Learning off: trial, reset, trial equals a fresh network running the second
    trial. Caught: LIFNodes.reset_state_variables not zeroing refrac_count."""
    spec, batch, steps = _first_trial(name)
    after_reset, _ = _reset_path(name, learning=False)

    net, make = N.build(name, learning=False)
    for conn in net.connections.values():  # the first run() of the reset path
        conn.normalize()  # normalised the weights once at its end
    N.run(net, spec, N.inputs(make, steps, batch, SEED_B))
    _assert_equal(after_reset, N.collect(net), f"{name} reset vs fresh")


@pytest.mark.parametrize("name", NAMES)
def test_reset_equals_fresh_learning_on(name):
    """Learning on: after reset, the second trial equals a fresh network that loaded
    the first trial's state_dict. Caught: PostPre caching the previous step's post
    trace on the rule object."""
    spec, batch, steps = _first_trial(name)
    after_reset, _ = _reset_path(name, learning=True)

    first, make = N.build(name)
    N.run(first, spec, N.inputs(make, steps, batch, SEED_A))
    net, make = N.build(name)
    _match_batch(net, batch)
    net.load_state_dict(first.state_dict())
    net.reset_state_variables()
    N.run(net, spec, N.inputs(make, steps, batch, SEED_B))
    _assert_equal(after_reset, N.collect(net), f"{name} reset vs fresh+state_dict")


# --------------------------------------------------------------------------- #
# 4. state_dict round trip
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
def test_state_dict_round_trip(name):
    """Learning off: a network rebuilt from another seed that loads a saved
    state_dict continues exactly like the original. Caught: LIFNodes keeping
    refrac_count as a plain attribute instead of a buffer."""
    spec, batch, steps = _first_trial(name)
    net, make = N.build(name, learning=False)
    N.run(net, spec, N.inputs(make, steps, batch, SEED_A))
    N.collect(net)  # empty the monitors

    buf = io.BytesIO()
    torch.save(net.state_dict(), buf)
    buf.seek(0)
    torch.manual_seed(12345)  # a different build seed: nothing may come from it
    spec_build = N.SPECS[name].build
    other, _ = spec_build(spec.dt)
    other.train(False)
    for lname, mon in net.monitors.items():
        other.add_monitor(
            Monitor(other.layers[lname[len(N.PREFIX) :]], mon.state_vars), name=lname
        )
    _match_batch(other, batch)
    other.load_state_dict(torch.load(buf, weights_only=True))

    # inputs() also reseeds the global generator, which SRM0Nodes draw from.
    N.run(net, spec, N.inputs(make, steps, batch, SEED_B))
    N.run(other, spec, N.inputs(make, steps, batch, SEED_B))
    _assert_equal(N.collect(net), N.collect(other), f"{name} state_dict round trip")


# --------------------------------------------------------------------------- #
# 5. learning off leaves the weights alone
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("name", NAMES)
def test_learning_off_keeps_weights(name):
    """learning=False leaves every connection's w and b bit-identical (weight
    normalisation turned off). Caught: AbstractConnection.update ignoring learning."""
    spec, batch, steps = _first_trial(name)
    net, make = N.build(name, learning=False)
    _disable_norm(net)

    def params():
        out = {
            f"{src}->{tgt}.{pname}": p.detach().clone()
            for (src, tgt), conn in net.connections.items()
            for pname, p in conn.named_parameters()
        }
        out.update({k: t.clone() for k, t in N.feature_tensors(net).items()})
        return out

    before = params()
    assert before, f"{name}: no connection parameters found"
    N.run(net, spec, N.inputs(make, steps, batch, SEED_A))
    _assert_equal(before, params(), f"{name} weights after a run with learning off")


# --------------------------------------------------------------------------- #
# 6. batch 1 versus batch N of the same sample
# --------------------------------------------------------------------------- #

BATCH_N = 3
# Floats are compared with a relative tolerance as well as the absolute one:
# batched kernels sum in a different order, which moves a float32 value by a
# couple of units in the last place (measured: CSRMNodes v off by 7.6e-6 at
# |v| = 53.8, relative 2.0e-7, csrm_window batch 1 vs 3). CLAUDE.md allows
# summation-order differences of about 1e-7 for batch > 1, stated here.
BATCH_RTOL = 1e-6


@pytest.mark.parametrize("name", NAMES)
def test_batch_of_copies_equals_batch_one(name):
    """Learning off: every sample of a batch of N copies equals the batch-1 run;
    spikes exact, floats within atol 1e-6 plus rtol 1e-6 (summation order). one_spike is turned off and SRM0Nodes layers
    are skipped, because both draw random numbers per sample. Caught:
    Connection.compute summing presynaptic spikes over the batch."""
    spec, _, steps = _first_trial(name)
    x1 = N.inputs(N.build(name)[1], steps, 1, SEED_A)
    xn = {k: v.repeat(1, BATCH_N, *([1] * (v.dim() - 2))) for k, v in x1.items()}

    def go(x):
        net, _ = N.build(name, learning=False)
        for layer in net.layers.values():
            if getattr(layer, "one_spike", False):
                layer.one_spike = False  # random tie-break: one draw per sample
        N.run(net, spec, x)
        stochastic = {  # fire from random draws, one per sample
            ln for ln, layer in net.layers.items() if isinstance(layer, SRM0Nodes)
        }
        return {
            k: v
            for k, v in N.collect(net).items()
            if not k.startswith(("state.", "feature."))
            and k.split(".")[0] not in stochastic
            and v.dim() > 1
        }

    one, many = go(x1), go(xn)
    problems = []
    for key, ref in one.items():
        if ref.size(1) != 1 or many[key].size(1) != BATCH_N:
            continue  # not a per-sample tensor (e.g. theta)
        for i in range(BATCH_N):
            got = many[key][:, i : i + 1]
            ok = (
                torch.allclose(got, ref, atol=1e-6, rtol=BATCH_RTOL)
                if ref.is_floating_point()
                else torch.equal(got, ref)
            )
            if not ok:
                problems.append(f"{key}[sample {i}]")
    assert not problems, f"{name}: batch of copies differs: {problems[:10]}"


# --------------------------------------------------------------------------- #
# 7. dt scaling of the learning rules
# --------------------------------------------------------------------------- #

TC_TRACE = 20.0  # Nodes default trace time constant, ms


def _postpre_dw(dt, t_pre, t_post, nu, total=40.0):
    """Weight change of one synapse for one pre and one post spike (times in ms).

    The post spike is forced with a per-step ``clamp``; the weight (0.5 mV) is far
    too small to make the LIF neuron fire by itself.
    """
    from bindsnet.learning import PostPre
    from bindsnet.network import Network
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection

    net = Network(dt=dt)
    net.add_layer(Input(n=1, traces=True, tc_trace=TC_TRACE), "pre")
    net.add_layer(LIFNodes(n=1, traces=True, tc_trace=TC_TRACE), "post")
    conn = Connection(
        net.layers["pre"],
        net.layers["post"],
        nu=nu,
        update_rule=PostPre,
        w=torch.full((1, 1), 0.5),
        wmin=-10.0,
        wmax=10.0,
    )
    net.add_connection(conn, "pre", "post")
    steps = int(round(total / dt))
    pre = torch.zeros(steps, 1, 1, dtype=torch.bool)
    post = torch.zeros(steps, 1, dtype=torch.bool)
    pre[int(round(t_pre / dt))] = True
    post[int(round(t_post / dt))] = True
    net.run(inputs={"pre": pre}, time=total, clamp={"post": post})
    return float(conn.w.item()) - 0.5


@pytest.mark.parametrize("order", ["pre_then_post", "post_then_pre"])
def test_postpre_has_no_dt_factor(order):
    """PostPre weight change per spike pair is the same at dt = 1 and dt = 0.5
    and equals nu * exp(-gap / tc_trace) (CLAUDE.md: pair STDP has no dt factor).
    Caught: PostPre._connection_update multiplying its update by dt."""
    nu = (1e-2, 2e-2)
    gap = 6.0  # ms between the two spikes
    t_first, t_second = 10.0, 10.0 + gap
    if order == "pre_then_post":
        times = dict(t_pre=t_first, t_post=t_second)
        expected = nu[1] * torch.exp(torch.tensor(-gap / TC_TRACE)).item()
    else:
        times = dict(t_pre=t_second, t_post=t_first)
        expected = -nu[0] * torch.exp(torch.tensor(-gap / TC_TRACE)).item()
    for dt in (1.0, 0.5):
        dw = _postpre_dw(dt, nu=nu, **times)
        assert dw == pytest.approx(expected, rel=1e-5), f"dt={dt}: dw={dw}"


def test_mstdpet_weight_step_scales_with_dt():
    """MSTDPET's weight step from a stored eligibility trace is
    nu * dt * r * exp(-dt / tc_e) * z (Florian eq. 2.7 with eq. 2.8, no new spikes),
    so the ratio between dt = 0.5 and dt = 1 is 0.5 * exp(0.5 / tc_e).
    Caught: MSTDPET._connection_update without the dt factor."""
    from bindsnet.learning import MSTDPET
    from bindsnet.network.nodes import Input, LIFNodes
    from bindsnet.network.topology import Connection

    nu, tc_e, z0, reward = 1e-2, 25.0, 0.8, 1.0

    def step(dt):
        src, tgt = Input(n=1), LIFNodes(n=1)
        src.batch_size = tgt.batch_size = 1
        src.s = torch.zeros(1, 1, dtype=torch.bool)
        tgt.s = torch.zeros(1, 1, dtype=torch.bool)
        conn = Connection(
            src, tgt, nu=nu, update_rule=MSTDPET, tc_e_trace=tc_e, wmin=-10.0, wmax=10.0
        )
        conn.dt = dt
        with torch.no_grad():
            conn.w.zero_()
        conn.update(reward=reward)  # creates zero traces; no spikes, no change
        conn.update_rule.eligibility_trace.fill_(z0)
        conn.update(reward=reward)
        return float(conn.w.item())

    for dt in (1.0, 0.5):
        expected = nu * dt * reward * torch.exp(torch.tensor(-dt / tc_e)).item() * z0
        assert step(dt) == pytest.approx(expected, rel=1e-5), f"dt={dt}"
    ratio = step(0.5) / step(1.0)
    assert ratio == pytest.approx(
        0.5 * torch.exp(torch.tensor(0.5 / tc_e)).item(), rel=1e-5
    )
