"""
Regression tests for the performance / memory changes to the per-timestep hot
paths. Every test pins the optimised code to the plain formula it replaced, so a
future change that alters the numerics is caught here.

Covered:

* ``MulticompartmentConnection`` can be moved with ``network.to(device)``
  (fixes a ``_apply`` signature mismatch that crashed the Diehl & Cook model).
* ``PostPre`` / ``Hebbian`` on dense ``Connection`` (classic and
  multicompartment) use one fused ``addmm_`` instead of materialising the
  ``[batch, source.n, target.n]`` outer product.
* ``LearningRule`` skips the ``w *= 1.0`` multiply when no weight decay is set.
* Local-connection rules scale rows directly instead of multiplying by an
  ``[n, n]`` identity matrix every timestep.
* Reward-modulated rules cache ``exp(-dt / tc)`` and their default learning
  rate tensors.
* Neuron models update their state buffers in place (same operations, same
  order) instead of re-assigning module attributes every step.
* ``rank_order`` encoding is vectorised.
"""

import pytest
import torch

from bindsnet.encoding import rank_order
from bindsnet.learning import MSTDP, MSTDPET, Hebbian, PostPre, WeightDependentPostPre
from bindsnet.learning import MCC_learning
from bindsnet.learning.learning import (
    _cached_decay,
    _dense_outer_update_ok,
    _reward_rates,
    _row_scale,
)
from bindsnet.models import DiehlAndCook2015
from bindsnet.network import Network
from bindsnet.network.nodes import (
    AdaptiveLIFNodes,
    CurrentLIFNodes,
    DiehlAndCookNodes,
    Input,
    IzhikevichNodes,
    LIFNodes,
    SRM0Nodes,
)
from bindsnet.network.topology import (
    Connection,
    LocalConnection1D,
    LocalConnection2D,
    LocalConnection3D,
    MulticompartmentConnection,
)
from bindsnet.network.topology_features import Weight


def _dense_net(rule, batch_size=1, n_in=40, n_out=25, dt=1.0, seed=0, **kwargs):
    torch.manual_seed(seed)
    net = Network(dt=dt, batch_size=batch_size)
    net.add_layer(Input(n=n_in, traces=True), "in")
    net.add_layer(LIFNodes(n=n_out, traces=True), "out")
    conn = Connection(
        net.layers["in"],
        net.layers["out"],
        nu=kwargs.pop("nu", (1e-2, 2e-2)),
        update_rule=rule,
        wmin=0.0,
        wmax=1.0,
        **kwargs,
    )
    net.add_connection(conn, "in", "out")
    return net, conn


def _warm_up(net, steps=15, seed=1):
    """Run a few steps so traces and spikes are non-trivial, then return the
    snapshot needed by the reference formulas."""
    torch.manual_seed(seed)
    b = net.batch_size
    n_in = net.layers["in"].n
    inp = torch.bernoulli(0.4 * torch.rand(steps, b, n_in)).byte()
    net.run(inputs={"in": inp}, time=steps * net.dt)
    # Make sure both sides have spikes and traces in the snapshot.
    net.layers["in"].s = torch.bernoulli(0.5 * torch.ones(b, n_in)).bool()
    net.layers["out"].s = torch.bernoulli(
        0.5 * torch.ones(b, net.layers["out"].n)
    ).bool()
    net.layers["in"].x = torch.rand(b, n_in)
    net.layers["out"].x = torch.rand(b, net.layers["out"].n)
    return net


def _reference_outer(source, target, batch_size, reduction):
    """The un-fused formula: outer products over the batch, then the batch
    reduction (``squeeze`` for batch 1, ``sum`` otherwise)."""
    source_s = source.s.view(batch_size, -1).unsqueeze(2).float()
    source_x = source.x.view(batch_size, -1).unsqueeze(2)
    target_s = target.s.view(batch_size, -1).unsqueeze(1).float()
    target_x = target.x.view(batch_size, -1).unsqueeze(1)
    return (
        reduction(torch.bmm(source_s, target_x), dim=0),  # pre: s_pre x x_post
        reduction(torch.bmm(source_x, target_s), dim=0),  # post: x_pre x s_post
    )


class TestMulticompartmentDeviceMove:
    def test_to_cpu_works(self):
        net = DiehlAndCook2015(n_inpt=16, n_neurons=4, inpt_shape=(1, 4, 4))
        net.to("cpu")  # crashed before: _apply() got an unexpected ``recurse``
        net.run(inputs={"X": torch.zeros(5, 1, 1, 4, 4).byte()}, time=5)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_to_cuda_moves_features(self):
        net = DiehlAndCook2015(n_inpt=16, n_neurons=4, inpt_shape=(1, 4, 4))
        net.to("cuda")
        for conn in net.connections.values():
            assert conn.pipeline[0].value.is_cuda
        net.run(inputs={"X": torch.zeros(5, 1, 1, 4, 4).byte().cuda()}, time=5)


class TestFusedOuterProductRules:
    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_postpre_matches_reference(self, batch_size):
        net, conn = _dense_net(PostPre, batch_size)
        _warm_up(net)
        rule = conn.update_rule
        w0 = conn.w.detach().clone()
        pre, post = _reference_outer(
            conn.source, conn.target, batch_size, rule.reduction
        )
        expected = (w0 - pre * rule.nu[0] + post * rule.nu[1]).clamp_(0.0, 1.0)

        conn.update(learning=True)

        if batch_size == 1:
            assert torch.equal(conn.w, expected)
        else:
            assert torch.allclose(conn.w, expected, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("batch_size", [1, 4])
    def test_hebbian_matches_reference(self, batch_size):
        net, conn = _dense_net(Hebbian, batch_size)
        _warm_up(net)
        rule = conn.update_rule
        w0 = conn.w.detach().clone()
        pre, post = _reference_outer(
            conn.source, conn.target, batch_size, rule.reduction
        )
        expected = (w0 + rule.nu[0] * pre + rule.nu[1] * post).clamp_(0.0, 1.0)

        conn.update(learning=True)

        if batch_size == 1:
            assert torch.equal(conn.w, expected)
        else:
            assert torch.allclose(conn.w, expected, rtol=1e-6, atol=1e-7)

    def test_fused_path_predicate(self):
        # The fused ``addmm_`` path is only taken for dense float32 weights,
        # scalar learning rates and the two default batch reductions.
        net, conn = _dense_net(PostPre, 1)
        rule = conn.update_rule
        assert _dense_outer_update_ok(rule, conn.w)
        assert not _dense_outer_update_ok(rule, conn.w.to(torch.float64))
        assert not _dense_outer_update_ok(rule, conn.w.to_sparse())
        rule.reduction = torch.mean
        assert not _dense_outer_update_ok(rule, conn.w)
        rule.reduction = torch.sum
        rule.nu = torch.stack([torch.rand(40, 25), torch.rand(40, 25)])
        assert not _dense_outer_update_ok(rule, conn.w)

    def test_custom_reduction_falls_back_and_matches(self):
        net, conn = _dense_net(PostPre, 4, reduction=torch.mean)
        _warm_up(net)
        rule = conn.update_rule
        assert rule.reduction is torch.mean
        w0 = conn.w.detach().clone()
        pre, post = _reference_outer(conn.source, conn.target, 4, torch.mean)
        expected = (w0 - pre * rule.nu[0] + post * rule.nu[1]).clamp_(0.0, 1.0)

        conn.update(learning=True)
        assert torch.allclose(conn.w, expected, rtol=1e-6, atol=1e-7)

    @pytest.mark.parametrize("dt", [1.0, 0.5])
    def test_mcc_postpre_matches_reference(self, dt):
        torch.manual_seed(0)
        net = Network(dt=dt)
        net.add_layer(Input(n=40, traces=True), "in")
        net.add_layer(LIFNodes(n=25, traces=True), "out")
        conn = MulticompartmentConnection(
            net.layers["in"],
            net.layers["out"],
            device="cpu",
            pipeline=[
                Weight(
                    "w",
                    0.3 * torch.rand(40, 25),
                    range=[0.0, 1.0],
                    nu=(1e-2, 2e-2),
                    learning_rule=MCC_learning.PostPre,
                )
            ],
        )
        net.add_connection(conn, "in", "out")
        _warm_up(net)
        rule = conn.pipeline[0].learning_rule
        w0 = conn.pipeline[0].value.detach().clone()
        pre, post = _reference_outer(conn.source, conn.target, 1, rule.reduction)
        expected = (w0 - pre * rule.nu[0] * dt + post * rule.nu[1] * dt).clamp_(
            0.0, 1.0
        )

        conn.update(learning=True)
        assert torch.equal(conn.pipeline[0].value, expected)

    def test_mcc_hebbian_matches_reference(self):
        torch.manual_seed(0)
        net = Network(dt=1.0)
        net.add_layer(Input(n=40, traces=True), "in")
        net.add_layer(LIFNodes(n=25, traces=True), "out")
        conn = MulticompartmentConnection(
            net.layers["in"],
            net.layers["out"],
            device="cpu",
            pipeline=[
                Weight(
                    "w",
                    0.3 * torch.rand(40, 25),
                    range=[0.0, 1.0],
                    nu=(1e-2, 2e-2),
                    learning_rule=MCC_learning.Hebbian,
                )
            ],
        )
        net.add_connection(conn, "in", "out")
        _warm_up(net)
        rule = conn.pipeline[0].learning_rule
        w0 = conn.pipeline[0].value.detach().clone()
        pre, post = _reference_outer(conn.source, conn.target, 1, rule.reduction)
        expected = (w0 + rule.nu[0] * pre + rule.nu[1] * post).clamp_(0.0, 1.0)

        conn.update(learning=True)
        assert torch.equal(conn.pipeline[0].value, expected)


class TestWeightDecay:
    def test_no_decay_leaves_weights_untouched(self):
        net, conn = _dense_net(PostPre, 1)
        assert conn.update_rule.weight_decay == 1.0
        w0 = conn.w.detach().clone()
        # No spikes and no traces: the only thing that could change ``w`` is decay.
        conn.update(learning=True)
        assert torch.equal(conn.w, w0)

    def test_decay_still_applied(self):
        net, conn = _dense_net(PostPre, 1, weight_decay=0.01)
        w0 = conn.w.detach().clone()
        conn.update(learning=True)
        assert torch.equal(conn.w, w0 * (1.0 - 0.01))


class TestLocalConnectionRules:
    def test_row_scale_equals_diagonal_bmm(self):
        torch.manual_seed(0)
        vec = torch.rand(3, 17, 1)
        mat = torch.rand(3, 17, 9)
        reference = torch.bmm(vec * torch.eye(17), mat)
        assert torch.equal(_row_scale(vec, mat), reference)
        assert torch.equal(_row_scale(vec.squeeze(2), mat), reference)

    @staticmethod
    def _local2d_net(rule, batch_size):
        torch.manual_seed(0)
        net = Network(dt=1.0, batch_size=batch_size)
        net.add_layer(Input(shape=[2, 8, 8], traces=True), "in")
        net.add_layer(LIFNodes(shape=[3, 3, 3], traces=True), "out")
        conn = LocalConnection2D(
            net.layers["in"],
            net.layers["out"],
            kernel_size=4,
            stride=2,
            n_filters=3,
            nu=(1e-2, 2e-2),
            update_rule=rule,
            wmin=0.0,
            wmax=1.0,
        )
        net.add_connection(conn, "in", "out")
        torch.manual_seed(1)
        inp = torch.bernoulli(0.4 * torch.rand(10, batch_size, 2, 8, 8)).byte()
        net.run(inputs={"in": inp}, time=10)
        net.layers["in"].s = torch.bernoulli(
            0.5 * torch.ones(batch_size, 2, 8, 8)
        ).bool()
        net.layers["out"].s = torch.bernoulli(
            0.5 * torch.ones(batch_size, 3, 3, 3)
        ).bool()
        net.layers["in"].x = torch.rand(batch_size, 2, 8, 8)
        net.layers["out"].x = torch.rand(batch_size, 3, 3, 3)
        return net, conn

    @staticmethod
    def _reference_local2d(conn, batch_size):
        """The previous implementation: diagonal matrices built with ``torch.eye``
        and multiplied with ``bmm``."""
        kh, kw = conn.kernel_size
        sh, sw = conn.stride
        c_in = conn.source.shape[0]
        n_out = conn.n_filters * conn.conv_size[0] * conn.conv_size[1]

        def unfold(t):
            return (
                t.float()
                .unfold(-2, kh, sh)
                .unfold(-2, kw, sw)
                .reshape(
                    batch_size, conn.conv_size[0] * conn.conv_size[1], c_in * kh * kw
                )
                .repeat(1, conn.n_filters, 1)
            )

        target_x = conn.target.x.reshape(batch_size, n_out, 1) * torch.eye(n_out)
        target_s = conn.target.s.float().reshape(batch_size, n_out, 1) * torch.eye(
            n_out
        )
        source_s, source_x = unfold(conn.source.s), unfold(conn.source.x)
        red = conn.update_rule.reduction
        pre = red(torch.bmm(target_x, source_s), dim=0).view(conn.w.size())
        post = red(torch.bmm(target_s, source_x), dim=0).view(conn.w.size())
        return pre, post

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_postpre_local2d_matches_diag_reference(self, batch_size):
        net, conn = self._local2d_net(PostPre, batch_size)
        nu = conn.update_rule.nu
        w0 = conn.w.detach().clone()
        pre, post = self._reference_local2d(conn, batch_size)
        expected = (w0 - nu[0] * pre + nu[1] * post).clamp_(0.0, 1.0)
        conn.update(learning=True)
        assert torch.equal(conn.w, expected)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_hebbian_local2d_matches_diag_reference(self, batch_size):
        net, conn = self._local2d_net(Hebbian, batch_size)
        nu = conn.update_rule.nu
        w0 = conn.w.detach().clone()
        pre, post = self._reference_local2d(conn, batch_size)
        expected = (w0 + nu[0] * pre + nu[1] * post).clamp_(0.0, 1.0)
        conn.update(learning=True)
        assert torch.equal(conn.w, expected)

    @pytest.mark.parametrize("batch_size", [1, 2])
    def test_weight_dependent_local2d_matches_diag_reference(self, batch_size):
        net, conn = self._local2d_net(WeightDependentPostPre, batch_size)
        nu = conn.update_rule.nu
        w0 = conn.w.detach().clone()
        pre, post = self._reference_local2d(conn, batch_size)
        update = -nu[0] * pre * (w0 - conn.wmin) + nu[1] * post * (conn.wmax - w0)
        expected = (w0 + update).clamp_(0.0, 1.0)
        conn.update(learning=True)
        assert torch.equal(conn.w, expected)

    @pytest.mark.parametrize(
        "rule", [PostPre, Hebbian, WeightDependentPostPre, MSTDP, MSTDPET]
    )
    @pytest.mark.parametrize("dim", [1, 2, 3])
    def test_all_local_rules_run(self, rule, dim):
        torch.manual_seed(0)
        net = Network(dt=1.0, batch_size=2)
        if dim == 1:
            in_shape, out_shape = [2, 12], [3, 5]
            conn_cls, k = LocalConnection1D, 4
        elif dim == 2:
            in_shape, out_shape = [2, 8, 8], [3, 3, 3]
            conn_cls, k = LocalConnection2D, 4
        else:
            in_shape, out_shape = [2, 6, 6, 6], [2, 2, 2, 2]
            conn_cls, k = LocalConnection3D, 4
        net.add_layer(Input(shape=in_shape, traces=True), "in")
        net.add_layer(LIFNodes(shape=out_shape, traces=True), "out")
        conn = conn_cls(
            net.layers["in"],
            net.layers["out"],
            kernel_size=k,
            stride=2,
            n_filters=out_shape[0],
            nu=(1e-2, 2e-2),
            update_rule=rule,
            wmin=0.0,
            wmax=1.0,
        )
        net.add_connection(conn, "in", "out")
        inp = torch.bernoulli(0.4 * torch.rand(20, 2, *in_shape)).byte()
        kw = {"reward": 0.5} if rule in (MSTDP, MSTDPET) else {}
        net.run(inputs={"in": inp}, time=20, **kw)
        assert torch.isfinite(conn.w).all()
        assert (conn.w >= 0).all() and (conn.w <= 1).all()
        if rule in (MSTDP, MSTDPET):
            # The traces are kept as vectors now, not diagonal matrices.
            r = conn.update_rule
            assert r.p_minus.shape == (2, net.layers["out"].n, 1)
            assert r.eligibility.shape == (2, *conn.w.shape)


class TestRewardRuleCaching:
    def test_cached_decay_matches_formula_and_tracks_changes(self):
        net, conn = _dense_net(MSTDP, 1, dt=0.5)
        rule = conn.update_rule
        d = _cached_decay(rule, "tc_plus")
        assert torch.equal(d, torch.exp(-0.5 / rule.tc_plus))
        assert _cached_decay(rule, "tc_plus") is d  # reused, not recomputed
        rule.tc_plus = torch.tensor(7.0)
        assert torch.equal(
            _cached_decay(rule, "tc_plus"), torch.exp(-0.5 / torch.tensor(7.0))
        )
        conn.dt = 2.0
        assert torch.equal(
            _cached_decay(rule, "tc_plus"), torch.exp(-2.0 / torch.tensor(7.0))
        )

    def test_reward_rates_defaults_reused_and_overrides_honoured(self):
        net, conn = _dense_net(MSTDP, 1)
        rule = conn.update_rule
        ap, am = _reward_rates(rule, {}, torch.device("cpu"))
        assert ap.item() == 1.0 and am.item() == -1.0
        ap2, am2 = _reward_rates(rule, {}, torch.device("cpu"))
        assert ap2 is ap and am2 is am
        ap3, am3 = _reward_rates(
            rule, {"a_plus": 0.3, "a_minus": -0.2}, torch.device("cpu")
        )
        assert ap3.item() == pytest.approx(0.3) and am3.item() == pytest.approx(-0.2)

    @pytest.mark.parametrize("rule", [MSTDP, MSTDPET])
    def test_reward_rules_match_step_by_step_reference(self, rule):
        # From-scratch Florian (2007) reference with the default one-step lag,
        # run alongside the network on identical spike trains.
        torch.manual_seed(0)
        n_in, n_out, T, dt = 12, 7, 25, 0.5
        net = Network(dt=dt)
        net.add_layer(Input(n=n_in), "in")
        net.add_layer(LIFNodes(n=n_out), "out")
        conn = Connection(
            net.layers["in"],
            net.layers["out"],
            nu=1e-2,
            update_rule=rule,
            wmin=-5.0,
            wmax=5.0,
            w=0.5 * torch.rand(n_in, n_out),
        )
        net.add_connection(conn, "in", "out")
        w_ref = conn.w.detach().clone()
        r = conn.update_rule
        tc_plus, tc_minus, tc_e = float(r.tc_plus), float(r.tc_minus), 25.0
        inp = torch.bernoulli(0.4 * torch.rand(T, n_in)).byte()
        p_plus, p_minus = torch.zeros(n_in), torch.zeros(n_out)
        elig, e_trace = torch.zeros(n_in, n_out), torch.zeros(n_in, n_out)
        reward = 0.7
        for t in range(T):
            net.run(inputs={"in": inp[t : t + 1]}, time=dt, reward=reward)
            pre = net.layers["in"].s.view(-1).float()
            post = net.layers["out"].s.view(-1).float()
            if rule is MSTDP:
                w_ref += 1e-2 * reward * elig
            else:
                e_trace = e_trace * torch.exp(torch.tensor(-dt / tc_e)) + elig / tc_e
                w_ref += 1e-2 * dt * reward * e_trace
            p_plus = p_plus * torch.exp(torch.tensor(-dt / tc_plus)) + pre
            p_minus = p_minus * torch.exp(torch.tensor(-dt / tc_minus)) - post
            elig = torch.outer(p_plus, post) + torch.outer(pre, p_minus)
            w_ref.clamp_(-5.0, 5.0)
            assert torch.allclose(conn.w, w_ref, rtol=1e-5, atol=1e-6), t


class TestNodesInPlace:
    def test_lif_matches_explicit_simulation(self):
        torch.manual_seed(0)
        n, T, dt = 30, 40, 0.5
        layer = LIFNodes(n=n, traces=True, lbound=-70.0)
        net = Network(dt=dt)
        net.add_layer(Input(n=n), "in")
        net.add_layer(layer, "out")
        w = torch.diag(20.0 * torch.ones(n))
        net.add_connection(Connection(net.layers["in"], layer, w=w), "in", "out")
        inp = torch.bernoulli(0.6 * torch.rand(T, n)).byte()

        v = layer.rest * torch.ones(1, n)
        refrac = torch.zeros(1, n)
        x_trace = torch.zeros(1, n)
        decay = torch.exp(-torch.tensor(dt) / layer.tc_decay)
        trace_decay = torch.exp(-torch.tensor(dt) / layer.tc_trace)
        for t in range(T):
            net.run(inputs={"in": inp[t : t + 1]}, time=dt)
            # Synchronous update: the layer sees the input spikes of step t-1.
            prev = inp[t - 1].float() if t > 0 else torch.zeros(n)
            inj = (prev @ w).unsqueeze(0)
            v = decay * (v - layer.rest) + layer.rest
            inj.masked_fill_(refrac > 0, 0.0)
            refrac -= torch.tensor(dt)
            v += inj
            s = v >= layer.thresh
            refrac.masked_fill_(s, layer.refrac)
            v.masked_fill_(s, layer.reset)
            v.masked_fill_(v < -70.0, -70.0)
            x_trace *= trace_decay
            x_trace.masked_fill_(s, 1.0)
            assert torch.equal(layer.s, s), t
            assert torch.equal(layer.v, v), t
            assert torch.equal(layer.x, x_trace), t

    def test_diehl_and_cook_matches_explicit_simulation(self):
        torch.manual_seed(0)
        n, T = 20, 40
        layer = DiehlAndCookNodes(n=n, one_spike=False, theta_plus=0.05)
        net = Network(dt=1.0)
        net.add_layer(Input(n=n), "in")
        net.add_layer(layer, "out")
        w = torch.diag(15.0 * torch.ones(n))
        net.add_connection(Connection(net.layers["in"], layer, w=w), "in", "out")
        inp = torch.bernoulli(0.7 * torch.rand(T, n)).byte()

        v = layer.rest * torch.ones(1, n)
        refrac = torch.zeros(1, n)
        theta = torch.zeros(n)
        decay = torch.exp(-torch.tensor(1.0) / layer.tc_decay)
        theta_decay = torch.exp(-torch.tensor(1.0) / layer.tc_theta_decay)
        for t in range(T):
            net.run(inputs={"in": inp[t : t + 1]}, time=1)
            prev = inp[t - 1].float() if t > 0 else torch.zeros(n)
            inj = (prev @ w).unsqueeze(0)
            v = decay * (v - layer.rest) + layer.rest
            theta *= theta_decay
            v += (refrac <= 0).float() * inj
            refrac -= torch.tensor(1.0)
            s = v >= layer.thresh + theta
            refrac.masked_fill_(s, layer.refrac)
            v.masked_fill_(s, layer.reset)
            theta += layer.theta_plus * s.float().sum(0)
            assert torch.equal(layer.s, s), t
            assert torch.equal(layer.v, v), t
            assert torch.equal(layer.theta, theta), t

    @pytest.mark.parametrize(
        "node",
        [LIFNodes, CurrentLIFNodes, AdaptiveLIFNodes, DiehlAndCookNodes, SRM0Nodes],
    )
    def test_state_buffers_keep_identity_and_registration(self, node):
        torch.manual_seed(0)
        net = Network(dt=1.0)
        net.add_layer(Input(n=10), "in")
        layer = node(n=6, traces=True)
        net.add_layer(layer, "out")
        net.add_connection(
            Connection(net.layers["in"], layer, w=torch.rand(10, 6)), "in", "out"
        )
        v_before = layer.v
        net.run(inputs={"in": torch.ones(5, 10).byte()}, time=5)
        # In-place updates: same tensor object, still a registered buffer.
        assert layer.v is v_before
        assert "v" in dict(layer.named_buffers())
        assert torch.isfinite(layer.v).all()

    def test_izhikevich_runs_finite(self):
        torch.manual_seed(0)
        net = Network(dt=1.0)
        net.add_layer(Input(n=10), "in")
        layer = IzhikevichNodes(n=10, excitatory=0.8)
        net.add_layer(layer, "out")
        net.add_connection(
            Connection(net.layers["in"], layer, w=5.0 * torch.rand(10, 10)), "in", "out"
        )
        net.run(
            inputs={"in": torch.bernoulli(0.5 * torch.rand(30, 10)).byte()}, time=30
        )
        assert torch.isfinite(layer.v).all() and torch.isfinite(layer.u).all()


class TestRankOrderEncoding:
    def test_matches_loop_reference(self):
        torch.manual_seed(0)
        time = 25
        datum = torch.rand(6, 9) * torch.bernoulli(torch.full((6, 9), 0.7))
        datum[0, 0] = 1.0  # a guaranteed maximum -> earliest spike
        out = rank_order(datum.clone(), time=time, dt=1.0)

        # Previous per-neuron loop implementation.
        d = datum.flatten().clone()
        d /= d.max()
        times = torch.zeros(d.numel())
        times[d != 0] = 1 / d[d != 0]
        times *= time / times.max()
        times = torch.ceil(times).long()
        ref = torch.zeros(time, d.numel()).byte()
        for i in range(d.numel()):
            if 0 < times[i] < time:
                ref[times[i] - 1, i] = 1
        assert torch.equal(out, ref.reshape(time, 6, 9))
        assert out.dtype == torch.uint8
