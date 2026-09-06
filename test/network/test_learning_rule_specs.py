# language=rst
"""
Validation of the weight-modifying learning rules against the equations of the
papers they implement. Each reference below is a from-scratch transcription of
the cited equations, driven with the spike trains actually produced by the
network, so the tests check the rule (not the neuron model) step by step.

Sources
-------
* Pair-based trace STDP (``PostPre``, ``WeightDependentPostPre``, ``Hebbian``):
  Morrison, Diesmann & Gerstner (2008), *Biol. Cybern.* 98:459-478,
  Sect. 4.1, eqs. (11)-(15):

  .. math::

     \\dot x_j = -x_j/\\tau_x + \\sum_f \\delta(t - t_j^f) \\qquad (11)

     \\dot y_i = -y_i/\\tau_y + \\sum_f \\delta(t - t_i^f) \\qquad (12)

     \\Delta w_{ij}^+(t_i^f) = F_+(w_{ij})\\, x_j(t_i^f) \\qquad (13)

     \\Delta w_{ij}^-(t_j^f) = -F_-(w_{ij})\\, y_i(t_j^f) \\qquad (14)

  with ``j`` presynaptic (``source``) and ``i`` postsynaptic (``target``).
  ``PostPre`` is the additive case :math:`F_+ = \\nu_\\text{post}`,
  :math:`F_- = \\nu_\\text{pre}`; ``WeightDependentPostPre`` is the soft-bounded
  (multiplicative) case :math:`F_+ = \\nu_\\text{post}(w_\\max - w)`,
  :math:`F_- = \\nu_\\text{pre}(w - w_\\min)`. Traces follow Sect. 2.3: the
  accumulating trace (``traces_additive=True``) adds 1 per spike; the saturating
  trace with :math:`A = 1` (``traces_additive=False``) resets to 1 on each spike.
  ``Hebbian`` is the same trace machinery with both terms positive (BindsNET's
  own definition; no paper equation).

* Reward-modulated STDP (``MSTDP``, ``MSTDPET``): Florian (2007), *Neural
  Comput.* 19:1468-1502, discrete-time eqs. (3.9)-(3.12) and (2.7)-(2.8). Those
  rules are validated in ``test_mstdp_florian.py``; this file only records the
  equation numbers and the timing convention:
  :math:`w(t+\\delta t) = w(t) + \\gamma\\, r(t+\\delta t)\\, \\zeta(t)` (3.9), i.e.
  the reward supplied at a step multiplies the eligibility of the *previous*
  step (the default ``zero_lag=False``).

* R-max (``Rmax``): Vasilaki, Fremaux, Urbanczik, Senn & Gerstner (2009),
  *PLoS Comput. Biol.* 5(12):e1000586, eqs. (7), (8) and (13):

  .. math::

     \\dot w_{ij} = \\alpha (R - b)\\, \\delta(t - t_\\text{hit})\\, e_{ij}(t) \\qquad (7)

     \\dot e_{ij} = -e_{ij}/\\tau_e + \\frac{g'}{g}\\Big[Y_i(t) -
       \\frac{\\rho_i(t)}{1 + \\tau_c\\,\\rho_i(t)}\\Big] \\sum_f \\epsilon(t - t_j^f)
       \\qquad (8)

     \\rho_i = g(u_i) = \\rho_0 \\exp\\big((u_i - u_0)/\\Delta u\\big) \\qquad (13)

  :math:`\\tau_c = 0` is the strict policy-gradient rule and
  :math:`\\tau_c \\to \\infty` the naive Hebbian rule (Vasilaki et al., p. 3-4).
  For the exponential :math:`g` of (13), :math:`g'/g = 1/\\Delta u` is a constant
  absorbed into the learning rate (ibid.). BindsNET discretises (8) with a
  forward-Euler decay and the per-step spike probability
  :math:`p = 1 - e^{-\\rho\\,\\delta t}` in place of :math:`\\rho\\,\\delta t`.

Spikes forced with the ``clamp`` argument of ``Network.run`` (and spikes removed with
``unclamp``) are applied inside ``Nodes.forward`` before the spike trace is updated,
so they are seen by the learning rules exactly like naturally generated spikes.
``TestClampEntersTraces`` pins that; the STDP-window test fires the post-synaptic
neuron through a teacher input and repeats it with ``clamp``.
"""

import math

import pytest
import torch

from bindsnet.learning import Hebbian, PostPre, Rmax, WeightDependentPostPre
from bindsnet.learning import MCC_learning
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, SRM0Nodes
from bindsnet.network.topology import Connection, MulticompartmentConnection
from bindsnet.network.topology_features import Weight

TOL = 1e-5


def _trace_step(trace, spikes, decay, additive):
    """Morrison et al. (2008) Sect. 2.3 trace, one time step."""
    trace = trace * decay
    if additive:
        return trace + spikes
    return torch.where(spikes > 0, torch.ones_like(trace), trace)


def _pair_stdp_reference(
    pre, post, w0, nu_pre, nu_post, dt, tc_trace, additive, f, wmin, wmax
):
    """
    Morrison et al. (2008) eqs. (11)-(14) in discrete time. ``pre``/``post`` are
    ``[T, n]`` 0/1 spike arrays; ``f(w)`` returns ``(F_minus(w), F_plus(w))``.
    Traces are updated before the weight change of the same step, matching the
    network's order (``Nodes.forward`` then ``Connection.update``).
    """
    w = w0.clone()
    x = torch.zeros(pre.shape[1])
    y = torch.zeros(post.shape[1])
    decay = math.exp(-dt / tc_trace)
    hist = []
    for t in range(pre.shape[0]):
        x = _trace_step(x, pre[t], decay, additive)
        y = _trace_step(y, post[t], decay, additive)
        f_minus, f_plus = f(w)
        w = w - nu_pre * f_minus * torch.outer(pre[t], y)  # (14): pre spike
        w = w + nu_post * f_plus * torch.outer(x, post[t])  # (13): post spike
        w = w.clamp(wmin, wmax)  # hard bounds, applied every step as in BindsNET
        hist.append(w.clone())
    return torch.stack(hist)


def _run_pair_rule(rule, dt, additive, wmin, wmax, seed=0, T=40, n_in=12, n_out=6):
    torch.manual_seed(seed)
    net = Network(dt=dt)
    net.add_layer(
        Input(n=n_in, traces=True, traces_additive=additive, tc_trace=20.0), "in"
    )
    net.add_layer(
        LIFNodes(n=n_out, traces=True, traces_additive=additive, tc_trace=20.0),
        "out",
    )
    w0 = 0.5 * torch.rand(n_in, n_out)
    conn = Connection(
        net.layers["in"],
        net.layers["out"],
        w=w0.clone(),
        nu=(1e-2, 3e-2),
        update_rule=rule,
        wmin=wmin,
        wmax=wmax,
    )
    net.add_connection(conn, "in", "out")
    # Strong drive so the LIF layer actually spikes.
    torch.manual_seed(seed + 1)
    pre = torch.bernoulli(0.5 * torch.ones(T, n_in))
    net.layers["out"].thresh.fill_(-60.0)
    w_hist, post_hist = [], []
    for t in range(T):
        net.run(inputs={"in": pre[t : t + 1].byte()}, time=dt)
        post_hist.append(net.layers["out"].s.view(-1).float().clone())
        w_hist.append(conn.w.detach().clone())
    post = torch.stack(post_hist)
    assert post.sum() > 0, "post-synaptic layer never spiked; test is vacuous"
    return w0, pre, post, torch.stack(w_hist), conn


class TestPairSTDPMorrison2008:
    @pytest.mark.parametrize("dt", [1.0, 0.5])
    @pytest.mark.parametrize("additive", [False, True])
    def test_postpre_is_additive_pair_stdp(self, dt, additive):
        w0, pre, post, w_b, conn = _run_pair_rule(PostPre, dt, additive, 0.0, 1.0)
        w_ref = _pair_stdp_reference(
            pre,
            post,
            w0,
            1e-2,
            3e-2,
            dt,
            20.0,
            additive,
            lambda w: (1.0, 1.0),
            0.0,
            1.0,
        )
        assert (w_b - w_ref).abs().max().item() < TOL

    @pytest.mark.parametrize("additive", [False, True])
    def test_weight_dependent_is_soft_bounded_pair_stdp(self, additive):
        wmin, wmax = 0.0, 1.0
        w0, pre, post, w_b, conn = _run_pair_rule(
            WeightDependentPostPre, 1.0, additive, wmin, wmax
        )
        w_ref = _pair_stdp_reference(
            pre,
            post,
            w0,
            1e-2,
            3e-2,
            1.0,
            20.0,
            additive,
            lambda w: (w - wmin, wmax - w),
            wmin,
            wmax,
        )
        assert (w_b - w_ref).abs().max().item() < TOL

    def test_hebbian_both_terms_potentiate(self):
        w0, pre, post, w_b, conn = _run_pair_rule(Hebbian, 1.0, False, 0.0, 1.0)
        w = w0.clone()
        x = torch.zeros(pre.shape[1])
        y = torch.zeros(post.shape[1])
        decay = math.exp(-1.0 / 20.0)
        hist = []
        for t in range(pre.shape[0]):
            x = _trace_step(x, pre[t], decay, False)
            y = _trace_step(y, post[t], decay, False)
            w = w + 1e-2 * torch.outer(pre[t], y) + 3e-2 * torch.outer(x, post[t])
            hist.append(w.clamp(0.0, 1.0).clone())
            w = w.clamp(0.0, 1.0)
        assert (w_b - torch.stack(hist)).abs().max().item() < TOL

    @pytest.mark.parametrize("drive", ["teacher", "clamp"])
    def test_stdp_window_sign_and_shape(self, drive):
        # Morrison (2008) eq. (10): pre-before-post potentiates by
        # F_+ exp(-|dt|/tau_+); post-before-pre depresses by F_- exp(-|dt|/tau_-).
        # The post neuron is fired either by a strong "teacher" input or by the
        # ``clamp`` argument of ``Network.run``; both must give the same window.
        tc = 20.0
        for delta in (1, 5, 15):
            for pre_first in (True, False):
                net = Network(dt=1.0)
                net.add_layer(Input(n=1, traces=True, tc_trace=tc), "in")
                net.add_layer(Input(n=1), "teacher")
                net.add_layer(LIFNodes(n=1, traces=True, tc_trace=tc), "out")
                conn = Connection(
                    net.layers["in"],
                    net.layers["out"],
                    w=torch.zeros(1, 1),  # no drive from the plastic synapse
                    nu=(1.0, 1.0),
                    update_rule=PostPre,
                    wmin=-10.0,
                    wmax=10.0,
                )
                net.add_connection(conn, "in", "out")
                net.add_connection(
                    Connection(
                        net.layers["teacher"],
                        net.layers["out"],
                        w=torch.full((1, 1), 100.0),
                    ),
                    "teacher",
                    "out",
                )
                T = 40
                pre = torch.zeros(T, 1, 1)
                teach = torch.zeros(T, 1, 1)
                if pre_first:
                    pre[10, 0, 0] = 1
                    teach[10 + delta - 1, 0, 0] = 1  # post fires one step later
                else:
                    teach[10 - 1, 0, 0] = 1
                    pre[10 + delta, 0, 0] = 1
                post_times = []
                for t in range(T):
                    if drive == "teacher":
                        net.run(
                            inputs={"in": pre[t].byte(), "teacher": teach[t].byte()},
                            time=1,
                        )
                    else:
                        # Force the post spike at the step the teacher would
                        # have fired it (one step after the teacher spike).
                        force = (
                            teach[t - 1, 0].bool() if t > 0 else torch.zeros(1).bool()
                        )
                        net.run(
                            inputs={
                                "in": pre[t].byte(),
                                "teacher": torch.zeros(1, 1).byte(),
                            },
                            time=1,
                            clamp={"out": force},
                        )
                    if net.layers["out"].s.any():
                        post_times.append(t)
                assert post_times == [10 + delta if pre_first else 10], post_times
                change = conn.w.item()
                expected = math.exp(-delta / tc) * (1 if pre_first else -1)
                assert abs(change - expected) < 1e-5, (delta, pre_first, change)

    def test_diehl_and_cook_2015_rule_is_not_postpre(self):
        # Diehl & Cook (2015), Methods "Learning": weights change only on
        # postsynaptic spikes, Delta w = eta (x_pre - x_tar) (w_max - w)^mu.
        # BindsNET's ``PostPre`` (used by ``DiehlAndCook2015``) instead has no
        # x_tar term and a depression term on presynaptic spikes. Pin that fact
        # so the deviation stays documented: a lone presynaptic spike changes
        # the weight under PostPre (it would not under Diehl & Cook's rule).
        net = Network(dt=1.0)
        net.add_layer(Input(n=1, traces=True), "in")
        net.add_layer(LIFNodes(n=1, traces=True), "out")
        conn = Connection(
            net.layers["in"],
            net.layers["out"],
            w=torch.full((1, 1), 0.5),
            nu=(1.0, 1.0),
            update_rule=PostPre,
            wmin=0.0,
            wmax=1.0,
        )
        net.add_connection(conn, "in", "out")
        net.layers["out"].x.fill_(0.3)  # a lingering post-synaptic trace
        pre = torch.zeros(3, 1, 1)
        pre[1, 0, 0] = 1
        net.run(inputs={"in": pre.byte()}, time=3)
        assert conn.w.item() < 0.5  # depressed by the pre spike alone


class TestMulticompartmentRulesMatchClassic:
    """The ``MCC_learning`` PostPre / Hebbian must apply the same equations as the
    classic rules (at ``dt = 1``, where the MCC rule's extra ``dt`` factor is 1)."""

    @pytest.mark.parametrize(
        "rule_pair", [(PostPre, MCC_learning.PostPre), (Hebbian, MCC_learning.Hebbian)]
    )
    def test_same_weights_step_by_step(self, rule_pair):
        classic, mcc = rule_pair
        torch.manual_seed(0)
        w0 = 0.5 * torch.rand(12, 6)
        torch.manual_seed(1)
        pre = torch.bernoulli(0.5 * torch.ones(40, 12)).byte()

        def build(use_mcc):
            net = Network(dt=1.0)
            net.add_layer(Input(n=12, traces=True), "in")
            net.add_layer(LIFNodes(n=6, traces=True, thresh=-60.0), "out")
            if use_mcc:
                conn = MulticompartmentConnection(
                    net.layers["in"],
                    net.layers["out"],
                    device="cpu",
                    pipeline=[
                        Weight(
                            "w",
                            w0.clone(),
                            range=[0.0, 1.0],
                            nu=(1e-2, 3e-2),
                            learning_rule=mcc,
                        )
                    ],
                )
            else:
                conn = Connection(
                    net.layers["in"],
                    net.layers["out"],
                    w=w0.clone(),
                    nu=(1e-2, 3e-2),
                    update_rule=classic,
                    wmin=0.0,
                    wmax=1.0,
                )
            net.add_connection(conn, "in", "out")
            return net, conn

        net_a, conn_a = build(False)
        net_b, conn_b = build(True)
        for t in range(40):
            net_a.run(inputs={"in": pre[t : t + 1]}, time=1)
            net_b.run(inputs={"in": pre[t : t + 1]}, time=1)
            wa, wb = conn_a.w, conn_b.pipeline[0].value
            assert (wa - wb).abs().max().item() < TOL, t


class TestRmaxVasilaki2009:
    @staticmethod
    def _build(tc_c, dt=1.0, seed=0, n_in=10, n_out=4):
        torch.manual_seed(seed)
        net = Network(dt=dt)
        net.add_layer(
            Input(n=n_in, traces=True, traces_additive=True, tc_trace=10.0), "in"
        )
        net.add_layer(SRM0Nodes(n=n_out, tc_decay=10.0, thresh=-55.0), "out")
        conn = Connection(
            net.layers["in"],
            net.layers["out"],
            w=2.0 * torch.rand(n_in, n_out),
            nu=1e-2,
            update_rule=Rmax,
            wmin=-10.0,
            wmax=10.0,
            tc_c=tc_c,
            tc_e_trace=25.0,
        )
        net.add_connection(conn, "in", "out")
        return net, conn

    def test_srm0_escape_rate_eq13(self):
        # rho = rho_0 exp((u - theta)/Delta u), spike probability 1 - exp(-rho dt).
        net, conn = self._build(5.0, dt=0.5)
        layer = net.layers["out"]
        torch.manual_seed(3)
        net.run(
            inputs={"in": torch.bernoulli(0.5 * torch.ones(5, 10)).byte()},
            time=2.5,
            reward=0.0,
        )
        rho = layer.rho_0 * torch.exp((layer.v - layer.thresh) / layer.d_thresh)
        assert torch.allclose(layer.rho, rho)
        assert torch.allclose(layer.s_prob, 1.0 - torch.exp(-rho * 0.5))

    @pytest.mark.parametrize("tc_c", [0.0, 5.0, 1e9])
    @pytest.mark.parametrize("dt", [1.0, 0.5])
    def test_eligibility_and_update_eq7_eq8(self, tc_c, dt):
        net, conn = self._build(tc_c, dt=dt)
        src, tgt = net.layers["in"], net.layers["out"]
        w_ref = conn.w.detach().clone()
        e = torch.zeros_like(w_ref)
        torch.manual_seed(4)
        T = 30
        pre = torch.bernoulli(0.5 * torch.ones(T, 10)).byte()
        rewards = torch.randn(T)
        for t in range(T):
            net.run(inputs={"in": pre[t : t + 1]}, time=dt, reward=rewards[t].item())
            # Observed post-synaptic factors of this step (Y_i and rho_i dt as p_i).
            Y = tgt.s.view(-1).float()
            p = tgt.s_prob.view(-1)
            eps = src.x.view(-1)  # sum_f epsilon(t - t_j^f): additive pre trace
            # Eq. (8), forward Euler in dt, with rho dt -> p and tau_c rho -> (tau_c/dt) p.
            post_factor = Y - p / (1.0 + (tc_c / dt) * p)
            e = e * (1.0 - dt / 25.0) + torch.outer(eps, post_factor)
            # Eq. (7): the reward of this step gates the eligibility of this step.
            w_ref = (w_ref + 1e-2 * rewards[t] * e).clamp(-10.0, 10.0)
            assert (conn.w - w_ref).abs().max().item() < TOL, t

    def test_tc_c_limits(self):
        # tau_c = 0: post factor is Y - p (policy gradient, mean-zero in
        # expectation); tau_c -> inf: post factor is Y (naive Hebbian).
        net, conn = self._build(0.0)
        rule = conn.update_rule
        tgt = net.layers["out"]
        tgt.s = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
        tgt.s_prob = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        net.layers["in"].x = torch.ones(1, 10)
        conn.update(reward=1.0, learning=True)
        expected = torch.tensor([1.0, 0.0, 1.0, 0.0]) - tgt.s_prob.view(-1)
        assert torch.allclose(rule.eligibility_trace[0], expected)

        net, conn = self._build(1e12)
        rule = conn.update_rule
        tgt = net.layers["out"]
        tgt.s = torch.tensor([[1, 0, 1, 0]], dtype=torch.bool)
        tgt.s_prob = torch.tensor([[0.2, 0.4, 0.6, 0.8]])
        net.layers["in"].x = torch.ones(1, 10)
        conn.update(reward=1.0, learning=True)
        assert torch.allclose(
            rule.eligibility_trace[0], torch.tensor([1.0, 0.0, 1.0, 0.0]), atol=1e-6
        )


class TestClampEntersTraces:
    """``clamp`` / ``unclamp`` spikes must be reflected in the spike traces."""

    @staticmethod
    def _layer_net():
        net = Network(dt=1.0)
        net.add_layer(Input(n=3), "in")
        net.add_layer(LIFNodes(n=3, traces=True, tc_trace=20.0), "out")
        net.add_connection(
            Connection(net.layers["in"], net.layers["out"], w=torch.zeros(3, 3)),
            "in",
            "out",
        )
        return net

    def test_clamped_spike_sets_trace(self):
        net = self._layer_net()
        mask = torch.tensor([True, False, False])
        net.run(inputs={"in": torch.zeros(1, 3).byte()}, time=1, clamp={"out": mask})
        assert torch.equal(net.layers["out"].s.view(-1), mask)
        assert torch.equal(net.layers["out"].x.view(-1), mask.float())
        # And it decays afterwards like any other spike.
        net.run(inputs={"in": torch.zeros(1, 3).byte()}, time=1)
        assert torch.allclose(
            net.layers["out"].x.view(-1), mask.float() * math.exp(-1.0 / 20.0)
        )

    def test_time_indexed_clamp(self):
        net = self._layer_net()
        mask = torch.zeros(4, 3, dtype=torch.bool)
        mask[2, 1] = True
        net.run(inputs={"in": torch.zeros(4, 3).byte()}, time=4, clamp={"out": mask})
        x = net.layers["out"].x.view(-1)
        assert x[1].item() == pytest.approx(math.exp(-1.0 / 20.0))
        assert x[0].item() == 0.0 and x[2].item() == 0.0

    def test_unclamped_spike_leaves_no_trace(self):
        net = self._layer_net()
        net.layers["out"].thresh.fill_(-64.0)
        net.run(
            inputs={"in": torch.ones(1, 3).byte()},
            time=1,
            injects_v={"out": 100.0 * torch.ones(3)},
            unclamp={"out": torch.tensor([False, False, True])},
        )
        s = net.layers["out"].s.view(-1)
        assert s[0] and s[1] and not s[2]
        assert torch.equal(net.layers["out"].x.view(-1), s.float())
