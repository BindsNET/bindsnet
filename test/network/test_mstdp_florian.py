# language=rst
"""
Validation tests for the reward-modulated learning rules ``MSTDP`` and
``MSTDPET`` against a from-scratch implementation of the discrete
`Florian (2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_
equations.

Motivated by issue #217 / #140 (P-traces had been written as a state
assignment instead of an exponentially-decaying trace).  These tests pin the
behaviour of the ``Connection``, ``LocalConnection`` and ``Conv`` paths for
both rules against a from-scratch Florian reference (point eligibility, weight
update, batch handling, STDP sign, and the ``zero_lag`` timing option).

Reference (i = presynaptic/source, j = postsynaptic/target):
    P+_i(t) = P+_i(t-dt) * exp(-dt/tc_plus)  + a_plus  * pre_i(t)
    P-_j(t) = P-_j(t-dt) * exp(-dt/tc_minus) + a_minus * post_j(t)
    zeta_ij(t) = P+_i(t) * post_j(t) + pre_i(t) * P-_j(t)
    MSTDP:   dw_ij(t) = nu * r(t) * zeta_ij(t)
    MSTDPET: e_ij(t)  = e_ij(t-dt)*exp(-dt/tc_e) + zeta_ij(t-dt)/tc_e
             dw_ij(t) = nu * dt * r(t) * e_ij(t)
"""

import itertools

import pytest
import torch

from bindsnet.learning import MSTDP, MSTDPET
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import (
    Connection,
    Conv3dConnection,
    LocalConnection1D,
    LocalConnection2D,
    LocalConnection3D,
    Conv1dConnection,
    Conv2dConnection,
)

TOL = 1e-5


# --------------------------------------------------------------------------- #
# From-scratch Florian (2007) reference implementations.                      #
# --------------------------------------------------------------------------- #
def _florian_mstdp(
    pre, post, rewards, nu, dt, tc_plus, tc_minus, a_plus=1.0, a_minus=-1.0, lag=True
):
    T, npre, npost = pre.shape[0], pre.shape[1], post.shape[1]
    p_plus = torch.zeros(npre)
    p_minus = torch.zeros(npost)
    elig = torch.zeros(npre, npost)
    w = torch.zeros(npre, npost)
    w_hist, pplus_hist = [], []
    for t in range(T):
        if lag:
            w = w + nu * rewards[t] * elig
        p_plus = p_plus * torch.exp(torch.tensor(-dt / tc_plus)) + a_plus * pre[t]
        p_minus = p_minus * torch.exp(torch.tensor(-dt / tc_minus)) + a_minus * post[t]
        elig = torch.outer(p_plus, post[t]) + torch.outer(pre[t], p_minus)
        if not lag:
            w = w + nu * rewards[t] * elig
        w_hist.append(w.clone())
        pplus_hist.append(p_plus.clone())
    return torch.stack(w_hist), torch.stack(pplus_hist)


def _florian_mstdpet(
    pre, post, rewards, nu, dt, tc_plus, tc_minus, tc_e, a_plus=1.0, a_minus=-1.0
):
    T, npre, npost = pre.shape[0], pre.shape[1], post.shape[1]
    p_plus = torch.zeros(npre)
    p_minus = torch.zeros(npost)
    elig = torch.zeros(npre, npost)
    etrace = torch.zeros(npre, npost)
    w = torch.zeros(npre, npost)
    w_hist = []
    for t in range(T):
        etrace = etrace * torch.exp(torch.tensor(-dt / tc_e)) + elig / tc_e
        w = w + nu * dt * rewards[t] * etrace
        p_plus = p_plus * torch.exp(torch.tensor(-dt / tc_plus)) + a_plus * pre[t]
        p_minus = p_minus * torch.exp(torch.tensor(-dt / tc_minus)) + a_minus * post[t]
        elig = torch.outer(p_plus, post[t]) + torch.outer(pre[t], p_minus)
        w_hist.append(w.clone())
    return torch.stack(w_hist)


# --------------------------------------------------------------------------- #
# Helpers to drive a Connection with explicit, controlled spike trains.       #
# --------------------------------------------------------------------------- #
def _make_conn(npre, npost, rule, batch=1, nu=1e-2, dt=1.0, **kw):
    src = Input(n=npre)
    tgt = LIFNodes(n=npost)
    src.batch_size = batch
    tgt.batch_size = batch
    src.s = torch.zeros(batch, npre, dtype=torch.bool)
    tgt.s = torch.zeros(batch, npost, dtype=torch.bool)
    conn = Connection(src, tgt, nu=nu, update_rule=rule, reduction=torch.sum, **kw)
    conn.dt = dt
    with torch.no_grad():
        conn.w.zero_()
    return conn


def _drive(conn, pre, post, rewards):
    """pre/post: [T, batch, n]; returns weight history [T, npre, npost]."""
    T, batch = pre.shape[0], pre.shape[1]
    w_hist, pplus_hist, elig_hist = [], [], []
    for t in range(T):
        conn.source.s = pre[t].bool().view(batch, -1)
        conn.target.s = post[t].bool().view(batch, -1)
        conn.update(reward=rewards[t])
        w_hist.append(conn.w.detach().clone())
        ur = conn.update_rule
        pplus_hist.append(ur.p_plus.detach().clone())
        elig_hist.append(ur.eligibility.detach().clone())
    return torch.stack(w_hist), torch.stack(pplus_hist), torch.stack(elig_hist)


class TestMSTDPFlorian:
    """Numerical validation of MSTDP / MSTDPET on ``Connection``."""

    def test_pplus_trace_is_exponential_decay(self):
        # Regression for issue #140: a single pre-spike must leave an
        # exponentially-decaying P+ trace, not a divergent state update.
        T, dt, tc = 40, 1.0, 20.0
        pre = torch.zeros(T, 1, 1)
        pre[0, 0, 0] = 1.0
        post = torch.zeros(T, 1, 1)
        rew = torch.ones(T)
        conn = _make_conn(1, 1, MSTDP, dt=dt, tc_plus=tc, tc_minus=tc)
        _, pplus, _ = _drive(conn, pre, post, rew)
        expected = torch.exp(-torch.arange(T).float() / tc)
        assert (pplus.view(T) - expected).abs().max().item() < TOL

    def test_mstdp_matches_florian(self):
        torch.manual_seed(1)
        T, npre, npost = 60, 4, 3
        pre = torch.bernoulli(torch.full((T, 1, npre), 0.2))
        post = torch.bernoulli(torch.full((T, 1, npost), 0.2))
        rew = torch.randn(T) * 0.5 + 1.0
        nu, dt, tcp, tcm = 1e-2, 1.0, 20.0, 22.0
        conn = _make_conn(npre, npost, MSTDP, nu=nu, dt=dt, tc_plus=tcp, tc_minus=tcm)
        w_b, _, _ = _drive(conn, pre, post, rew)
        w_f, _ = _florian_mstdp(pre[:, 0], post[:, 0], rew, nu, dt, tcp, tcm, lag=True)
        assert (w_b - w_f).abs().max().item() < TOL

    def test_mstdpet_matches_florian(self):
        torch.manual_seed(2)
        T, npre, npost = 60, 4, 3
        pre = torch.bernoulli(torch.full((T, 1, npre), 0.2))
        post = torch.bernoulli(torch.full((T, 1, npost), 0.2))
        rew = torch.randn(T) * 0.5 + 1.0
        nu, dt, tcp, tcm, tce = 1e-2, 1.0, 20.0, 22.0, 25.0
        conn = _make_conn(
            npre,
            npost,
            MSTDPET,
            nu=nu,
            dt=dt,
            tc_plus=tcp,
            tc_minus=tcm,
            tc_e_trace=tce,
        )
        w_b, _, _ = _drive(conn, pre, post, rew)
        w_f = _florian_mstdpet(pre[:, 0], post[:, 0], rew, nu, dt, tcp, tcm, tce)
        assert (w_b - w_f).abs().max().item() < TOL

    def test_stdp_causality_sign(self):
        # pre-before-post must potentiate; post-before-pre must depress.
        T, dt, tc = 10, 1.0, 20.0

        def elig_at_second_spike(pre_first):
            pre = torch.zeros(T, 1, 1)
            post = torch.zeros(T, 1, 1)
            if pre_first:
                pre[2, 0, 0] = 1.0
                post[5, 0, 0] = 1.0
            else:
                post[2, 0, 0] = 1.0
                pre[5, 0, 0] = 1.0
            conn = _make_conn(1, 1, MSTDP, nu=1.0, dt=dt, tc_plus=tc, tc_minus=tc)
            _, _, elig = _drive(conn, pre, post, torch.ones(T))
            return elig[5].view(()).item()

        assert elig_at_second_spike(pre_first=True) > 0.0
        assert elig_at_second_spike(pre_first=False) < 0.0

    def test_zero_lag_matches_florian_no_lag(self):
        # With zero_lag=True, MSTDP must reproduce the un-lagged Florian update.
        torch.manual_seed(3)
        T, npre, npost = 50, 4, 3
        pre = torch.bernoulli(torch.full((T, 1, npre), 0.2))
        post = torch.bernoulli(torch.full((T, 1, npost), 0.2))
        rew = torch.randn(T) * 0.5 + 1.0
        nu, dt, tcp, tcm = 1e-2, 1.0, 20.0, 22.0
        conn = _make_conn(
            npre, npost, MSTDP, nu=nu, dt=dt, tc_plus=tcp, tc_minus=tcm, zero_lag=True
        )
        w_b, _, _ = _drive(conn, pre, post, rew)
        w_f, _ = _florian_mstdp(pre[:, 0], post[:, 0], rew, nu, dt, tcp, tcm, lag=False)
        assert (w_b - w_f).abs().max().item() < TOL

    @pytest.mark.parametrize("rule", [MSTDP, MSTDPET])
    def test_batch_equals_sum_of_samples(self, rule):
        # Batched update with reduction=sum must equal the sum of the
        # independent per-sample updates.
        torch.manual_seed(7)
        T, npre, npost, B = 40, 5, 4, 3
        pre = torch.bernoulli(torch.full((T, B, npre), 0.25))
        post = torch.bernoulli(torch.full((T, B, npost), 0.25))
        rew = torch.ones(T)

        conn = _make_conn(npre, npost, rule, batch=B)
        w_batch, _, _ = _drive(conn, pre, post, rew)

        w_sum = torch.zeros_like(w_batch[-1])
        for b in range(B):
            cb = _make_conn(npre, npost, rule, batch=1)
            wb, _, _ = _drive(cb, pre[:, b : b + 1], post[:, b : b + 1], rew)
            w_sum = w_sum + wb[-1]

        assert (w_batch[-1] - w_sum).abs().max().item() < TOL
        assert w_batch[-1].abs().sum().item() > 0.0  # actually learned

    def test_conv3d_runs_without_dtype_error(self):
        # Regression: conv3d source spikes must be cast to float (issue: bmm
        # dtype mismatch).  Both rules should run a few steps without error.
        for rule in (MSTDP, MSTDPET):
            src = Input(shape=[1, 8, 8, 8])
            tgt = LIFNodes(shape=[4, 6, 6, 6])
            src.batch_size = 1
            tgt.batch_size = 1
            conn = Conv3dConnection(
                src, tgt, kernel_size=3, stride=1, nu=1e-2, update_rule=rule
            )
            from bindsnet.network import Network

            net = Network(dt=1.0, batch_size=1)
            net.add_layer(src, name="in")
            net.add_layer(tgt, name="out")
            net.add_connection(conn, source="in", target="out")
            inp = torch.bernoulli(torch.rand(5, 1, 1, 8, 8, 8)).byte()
            net.run(inputs={"in": inp}, time=5, reward=1.0)  # must not raise


def _drive_shaped(conn, pre, post):
    """Drive a conv/local connection with explicitly-shaped spike trains.

    pre/post: [T, batch, *node_shape]. Returns (weight_change, [eligibility...]).
    """
    T = pre.shape[0]
    w0 = conn.w.detach().clone()
    eligs = []
    for t in range(T):
        conn.source.s = pre[t].bool()
        conn.target.s = post[t].bool()
        conn.update(reward=1.0)
        eligs.append(conn.update_rule.eligibility.detach().clone())
    return conn.w.detach() - w0, eligs


_LOCAL_CASES = [
    (LocalConnection1D, [1, 12], [4, 10]),
    (LocalConnection2D, [1, 8, 8], [4, 6, 6]),
    (LocalConnection3D, [1, 6, 6, 6], [4, 4, 4, 4]),
]


def _make_local(lc, in_shape, out_shape, rule, batch):
    src = Input(shape=in_shape)
    tgt = LIFNodes(shape=out_shape)
    src.batch_size = batch
    tgt.batch_size = batch
    src.s = torch.zeros(batch, *in_shape, dtype=torch.bool)
    tgt.s = torch.zeros(batch, *out_shape, dtype=torch.bool)
    conn = lc(
        src,
        tgt,
        kernel_size=3,
        stride=1,
        n_filters=4,
        nu=1e-2,
        update_rule=rule,
        reduction=torch.sum,
    )
    conn.dt = 1.0
    with torch.no_grad():
        conn.w.zero_()
    return conn


class TestLocalMSTDPET:
    """LocalConnection MSTDPET is validated against the (sound) Local MSTDP.

    The local-connection trace init uses ``.repeat`` (a copy), so it does not
    suffer the conv overlap-aliasing bug; these variants are correct.
    """

    @pytest.mark.parametrize("lc,in_shape,out_shape", _LOCAL_CASES)
    def test_local_mstdpet_learns(self, lc, in_shape, out_shape):
        torch.manual_seed(0)
        T = 25
        pre = torch.bernoulli(torch.full((T, 1, *in_shape), 0.3))
        post = torch.bernoulli(torch.full((T, 1, *out_shape), 0.3))
        conn = _make_local(lc, in_shape, out_shape, MSTDPET, 1)
        dW, _ = _drive_shaped(conn, pre, post)
        assert dW.abs().sum().item() > 0.0

    @pytest.mark.parametrize("lc,in_shape,out_shape", _LOCAL_CASES)
    def test_local_mstdpet_eligibility_matches_mstdp(self, lc, in_shape, out_shape):
        # MSTDPET must compute exactly the same point eligibility as MSTDP.
        torch.manual_seed(1)
        T = 20
        pre = torch.bernoulli(torch.full((T, 1, *in_shape), 0.3))
        post = torch.bernoulli(torch.full((T, 1, *out_shape), 0.3))
        _, e_et = _drive_shaped(
            _make_local(lc, in_shape, out_shape, MSTDPET, 1), pre, post
        )
        _, e_mp = _drive_shaped(
            _make_local(lc, in_shape, out_shape, MSTDP, 1), pre, post
        )
        err = max((a - b).abs().max().item() for a, b in zip(e_et, e_mp))
        assert err < TOL

    @pytest.mark.parametrize("lc,in_shape,out_shape", _LOCAL_CASES)
    def test_local_mstdpet_batch_equals_sum(self, lc, in_shape, out_shape):
        torch.manual_seed(2)
        T, B = 20, 3
        pre = torch.bernoulli(torch.full((T, B, *in_shape), 0.3))
        post = torch.bernoulli(torch.full((T, B, *out_shape), 0.3))
        dW_b, _ = _drive_shaped(
            _make_local(lc, in_shape, out_shape, MSTDPET, B), pre, post
        )
        dW_s = torch.zeros_like(dW_b)
        for b in range(B):
            d, _ = _drive_shaped(
                _make_local(lc, in_shape, out_shape, MSTDPET, 1),
                pre[:, b : b + 1],
                post[:, b : b + 1],
            )
            dW_s = dW_s + d
        assert (dW_b - dW_s).abs().max().item() < TOL


import itertools

_CONV_CASES = [
    (1, Conv1dConnection, [2, 8], [3, 6]),
    (2, Conv2dConnection, [1, 8, 8], [3, 6, 6]),
    (3, Conv3dConnection, [1, 6, 6, 6], [2, 4, 4, 4]),
]


def _conv_florian_ref(Pp, Pm, pre, post, wshape, dim, stride=1):
    """Batch-summed n-d Florian conv point eligibility, computed from scratch.

    eligibility[o, i, *k] = sum_b sum_p  P+[b,i, p*stride+k] * post[b,o,p]
                                       + pre[b,i, p*stride+k] * P-[b,o,p]
    """
    Co, Ci = wshape[0], wshape[1]
    ksz = wshape[2:]
    B = pre.shape[0]
    Lout = post.shape[2:]
    ref = torch.zeros(*wshape)
    for b in range(B):
        for co in range(Co):
            for ci in range(Ci):
                for k in itertools.product(*[range(x) for x in ksz]):
                    acc = 0.0
                    for p in itertools.product(*[range(x) for x in Lout]):
                        idx = tuple(p[d] * stride + k[d] for d in range(dim))
                        acc += (
                            Pp[(b, ci) + idx] * post[(b, co) + p]
                            + pre[(b, ci) + idx] * Pm[(b, co) + p]
                        )
                    ref[(co, ci) + k] += acc
    return ref


def _make_conv(conn_cls, in_shape, out_shape, rule, batch, tc=20.0):
    src = Input(shape=in_shape)
    tgt = LIFNodes(shape=out_shape)
    src.batch_size = batch
    tgt.batch_size = batch
    src.s = torch.zeros(batch, *in_shape, dtype=torch.bool)
    tgt.s = torch.zeros(batch, *out_shape, dtype=torch.bool)
    conn = conn_cls(
        src,
        tgt,
        kernel_size=3,
        stride=1,
        nu=1e-2,
        update_rule=rule,
        tc_plus=tc,
        tc_minus=tc,
    )
    conn.dt = 1.0
    with torch.no_grad():
        conn.w.zero_()
    return conn


class TestConvMSTDP:
    """Conv MSTDP/MSTDPET validated against the from-scratch Florian conv rule.

    Traces are kept in neuron space and the eligibility is the convolution
    weight-gradient, so it is correct for any channels/stride and supports
    batches (the previous im2col-space trace double-counted overlapping
    kernel positions).
    """

    @pytest.mark.parametrize("dim,conn_cls,in_shape,out_shape", _CONV_CASES)
    def test_conv_mstdp_eligibility_matches_florian(
        self, dim, conn_cls, in_shape, out_shape
    ):
        torch.manual_seed(0)
        tc = 20.0
        conn = _make_conv(conn_cls, in_shape, out_shape, MSTDP, 1, tc=tc)
        T = 5
        Pp = torch.zeros(1, *in_shape)
        Pm = torch.zeros(1, *out_shape)
        decay = torch.exp(torch.tensor(-1.0 / tc))
        max_err = 0.0
        for _ in range(T):
            pre = torch.bernoulli(torch.full((1, *in_shape), 0.5))
            post = torch.bernoulli(torch.full((1, *out_shape), 0.5))
            conn.source.s = pre.bool()
            conn.target.s = post.bool()
            conn.update(reward=1.0)
            Pp = Pp * decay + pre
            Pm = Pm * decay - post
            ref = _conv_florian_ref(Pp, Pm, pre, post, tuple(conn.w.shape), dim)
            max_err = max(
                max_err, (conn.update_rule.eligibility - ref).abs().max().item()
            )
        assert max_err < 1e-4  # float32 accumulation tolerance

    @pytest.mark.parametrize("dim,conn_cls,in_shape,out_shape", _CONV_CASES)
    def test_conv_mstdpet_eligibility_matches_mstdp(
        self, dim, conn_cls, in_shape, out_shape
    ):
        torch.manual_seed(1)
        T = 12
        pre = torch.bernoulli(torch.full((T, 1, *in_shape), 0.3))
        post = torch.bernoulli(torch.full((T, 1, *out_shape), 0.3))
        _, e_et = _drive_shaped(
            _make_conv(conn_cls, in_shape, out_shape, MSTDPET, 1), pre, post
        )
        _, e_mp = _drive_shaped(
            _make_conv(conn_cls, in_shape, out_shape, MSTDP, 1), pre, post
        )
        err = max((a - b).abs().max().item() for a, b in zip(e_et, e_mp))
        assert err < TOL

    @pytest.mark.parametrize("dim,conn_cls,in_shape,out_shape", _CONV_CASES)
    @pytest.mark.parametrize("rule", [MSTDP, MSTDPET])
    def test_conv_learns(self, dim, conn_cls, in_shape, out_shape, rule):
        torch.manual_seed(2)
        T = 25
        pre = torch.bernoulli(torch.full((T, 1, *in_shape), 0.3))
        post = torch.bernoulli(torch.full((T, 1, *out_shape), 0.3))
        dW, _ = _drive_shaped(
            _make_conv(conn_cls, in_shape, out_shape, rule, 1), pre, post
        )
        assert dW.abs().sum().item() > 0.0

    @pytest.mark.parametrize("dim,conn_cls,in_shape,out_shape", _CONV_CASES)
    @pytest.mark.parametrize("rule", [MSTDP, MSTDPET])
    def test_conv_batch_equals_sum(self, dim, conn_cls, in_shape, out_shape, rule):
        # Conv eligibility is batch-summed, so a batched run equals the sum of
        # the per-sample runs (scalar reward).
        torch.manual_seed(3)
        T, B = 15, 3
        pre = torch.bernoulli(torch.full((T, B, *in_shape), 0.3))
        post = torch.bernoulli(torch.full((T, B, *out_shape), 0.3))
        dW_b, _ = _drive_shaped(
            _make_conv(conn_cls, in_shape, out_shape, rule, B), pre, post
        )
        dW_s = torch.zeros_like(dW_b)
        for b in range(B):
            d, _ = _drive_shaped(
                _make_conv(conn_cls, in_shape, out_shape, rule, 1),
                pre[:, b : b + 1],
                post[:, b : b + 1],
            )
            dW_s = dW_s + d
        assert (dW_b - dW_s).abs().max().item() < 1e-4
