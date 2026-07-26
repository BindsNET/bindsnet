import torch
import math

from bindsnet.learning import (
    MSTDP,
    MSTDPET,
    Hebbian,
    NoOp,
    PostPre,
    Rmax,
    WeightDependentPostPre,
)
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, SRM0Nodes
from bindsnet.network.topology import *
import bindsnet.learning.MCC_learning as mcc
import bindsnet.network.topology_features as tf


class TestConnection:
    """
    Tests all stable groups of neurons / nodes.
    """

    def __init__(self):
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
        else:
            self.device = torch.device("cpu:0")
        print(f"Using device '{self.device}' for the test")

    def test_transfer(self):
        if not torch.cuda.is_available():
            return

        connection_types = [
            Connection,
            Conv2dConnection,
            MaxPool2dConnection,
            LocalConnection,
            MeanFieldConnection,
            SparseConnection,
        ]
        args = [[], [3], [3], [3, 1, 1], [], []]
        kwargs = [{}, {}, {}, {}, {}, {"sparsity": 0.9}]
        for conn_type, args, kwargs in zip(connection_types, args, kwargs):
            l_a = LIFNodes(shape=[1, 28, 28])
            l_b = LIFNodes(shape=[1, 26, 26])
            connection = conn_type(l_a, l_b, *args, **kwargs)

            connection.to()

            connection_tensors = [
                k
                for k, v in connection.state_dict().items()
                if isinstance(v, torch.Tensor) and not "." in k
            ]

            print(
                "State dict in {} : {}".format(
                    conn_type, connection.state_dict().keys()
                )
            )
            print("__dict__ in {} : {}".format(conn_type, connection.__dict__.keys()))
            print("Tensors in {} : {}".format(conn_type, connection_tensors))

            tensor_devs = [getattr(connection, k).device for k in connection_tensors]
            print(
                "Tensor devices {}".format(list(zip(connection_tensors, tensor_devs)))
            )

            for d in tensor_devs:
                print(d, d == torch.device("cuda:0"))
                assert d == torch.device("cuda:0")

    def test_weights(self, conn_type, shape_a, shape_b, shape_w, *args, **kwargs):
        print("Testing:", conn_type)
        time = 100
        weights = [None, torch.Tensor(*shape_w)]
        wmins = [
            -np.inf,
            0,
            torch.zeros(*shape_w),
            torch.zeros(*shape_w).masked_fill(
                torch.bernoulli(torch.rand(*shape_w)) == 1, -np.inf
            ),
        ]
        wmaxes = [
            np.inf,
            0,
            torch.ones(*shape_w),
            torch.randn(*shape_w).masked_fill(
                torch.bernoulli(torch.rand(*shape_w)) == 1, np.inf
            ),
        ]
        update_rule = kwargs.get("update_rule", None)
        for w in weights:
            for wmin in wmins:
                for wmax in wmaxes:
                    ### Conditional checks ###
                    # WeightDependentPostPre does not handle infinite ranges
                    if (
                        (torch.tensor(wmin, dtype=torch.float32) == -np.inf).any()
                        or (torch.tensor(wmax, dtype=torch.float32) == np.inf).any()
                    ) and update_rule == WeightDependentPostPre:
                        continue

                    # Rmax only supported for Connection & LocalConnection
                    elif (
                        not (conn_type == Connection or conn_type == LocalConnection)
                        and update_rule == Rmax
                    ):
                        return

                    # SparseConnection isn't supported for wmin\\wmax
                    elif (conn_type == SparseConnection) and not (
                        (torch.tensor(wmin, dtype=torch.float32) == -np.inf).all()
                        and (torch.tensor(wmax, dtype=torch.float32) == np.inf).all()
                    ):
                        continue

                    print(
                        f"- w: {type(w).__name__}, "
                        f"wmin: {type(wmax).__name__}, wmax: {type(wmax).__name__}"
                    )
                    if kwargs.get("update_rule") == Rmax:
                        l_a = SRM0Nodes(
                            shape=shape_a, traces=True, traces_additive=True
                        )
                        l_b = SRM0Nodes(
                            shape=shape_b, traces=True, traces_additive=True
                        )
                    else:
                        l_a = LIFNodes(shape=shape_a, traces=True, traces_additive=True)
                        l_b = LIFNodes(shape=shape_b, traces=True, traces_additive=True)

                    ### Create network ###
                    network = Network(dt=1.0)
                    network.add_layer(
                        Input(n=100, traces=True, traces_additive=True), name="input"
                    )
                    network.add_layer(l_a, name="a")
                    network.add_layer(l_b, name="b")

                    network.add_connection(
                        conn_type(l_a, l_b, w=w, wmin=wmin, wmax=wmax, *args, **kwargs),
                        source="a",
                        target="b",
                    )
                    network.add_connection(
                        Connection(
                            wmin=0,
                            wmax=1,
                            source=network.layers["input"],
                            target=network.layers["a"],
                            **kwargs,
                        ),
                        source="input",
                        target="a",
                    )

                    ### Run network ###
                    network.run(
                        inputs={"input": torch.bernoulli(torch.rand(time, 100)).byte()},
                        time=time,
                        reward=1,
                    )


class TestMultiCompartmentConnection:

    def __init__(self):
        self.device = torch.device("cpu")
        print(f"Using device '{self.device}' for the MCC test")

    # ----------------------------------------------------------------------- #
    # Helpers                                                                 #
    # ----------------------------------------------------------------------- #

    def _make_mcc(self, pipeline, src_n, tgt_n, batch=1, sparse_compute=False):
        """Build (and prime) a standalone MCC: Input(src_n) -> LIFNodes(tgt_n)."""
        src = Input(n=src_n, traces=True)
        tgt = LIFNodes(n=tgt_n, traces=True)
        # batch_size is None until a Network sets it; the learning rules need it.
        src.batch_size = batch
        tgt.batch_size = batch
        conn = MulticompartmentConnection(
            source=src,
            target=tgt,
            device=self.device,
            pipeline=pipeline,
            sparse_compute=sparse_compute,
        )
        conn.dt = 1.0  # not set until added to a Network; rules read connection.dt
        return conn

    def _reference_expansion(self, pipeline, s, tgt_n):
        """Pre-collapse pipeline features """
        b, src = s.shape
        x = s.view(b, src, 1).expand(b, src, tgt_n).clone().float()
        for f in pipeline:
            op = getattr(f, "op", "mul")
            if isinstance(f, tf.Degradation):
                v = (
                    f.degrade_function(f.value)
                    if f.degrade_function is not None
                    else f.value
                )
            else:
                v = f.value
            if torch.is_tensor(v):
                v = v.float()
            if op == "mul":
                x = x * v
            elif op == "add":
                x = x + v
            else:  # "sub"
                x = x - v
        return x.sum(dim=1)

    def _learning_conn(self, rule, w0, nu, rng=(-1.0, 1.0)):
        """MCC with a single learnable Weight; returns (connection, feature)."""
        src_n, tgt_n = w0.shape
        conn = self._make_mcc(
            [tf.Weight(name="w", value=w0.clone(), learning_rule=rule, nu=nu, range=rng)],
            src_n,
            tgt_n,
            batch=1,
        )
        return conn, conn.pipeline[0]

    def _mstdp_seq(self):
        return [
            (torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, 0.0]), 0.0),
            (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([1.0, 0.0]), 1.0),
            (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0]), 1.0),
            (torch.tensor([0.0, 0.0, 0.0]), torch.tensor([0.0, 0.0]), 1.0),
        ]

    def _mstdp_reference(self, w0, seq, nu0, dt, tc_plus=20.0, tc_minus=20.0, rng=(-1.0, 1.0)):
        w = w0.clone().float()
        src_n, tgt_n = w.shape
        p_plus, p_minus = torch.zeros(src_n), torch.zeros(tgt_n)
        elig = torch.zeros(src_n, tgt_n)
        dp, dm = math.exp(-dt / tc_plus), math.exp(-dt / tc_minus)
        for src_s, tgt_s, reward in seq:
            w = w + nu0 * reward * elig
            p_plus = dp * p_plus + src_s
            p_minus = dm * p_minus - tgt_s
            elig = torch.outer(p_plus, tgt_s) + torch.outer(src_s, p_minus)
            w = torch.clamp(w, rng[0], rng[1])
        return w

    def _mstdpet_reference(
        self, w0, seq, nu0, dt, tc_plus=20.0, tc_minus=20.0, tc_e=25.0, rng=(-1.0, 1.0)
    ):
        w = w0.clone().float()
        src_n, tgt_n = w.shape
        p_plus, p_minus = torch.zeros(src_n), torch.zeros(tgt_n)
        elig = torch.zeros(src_n, tgt_n)
        elig_tr = torch.zeros(src_n, tgt_n)
        for src_s, tgt_s, reward in seq:
            elig_tr = elig_tr * math.exp(-dt / tc_e) + elig / tc_e
            w = w + nu0 * dt * reward * elig_tr
            p_plus = p_plus * math.exp(-dt / tc_plus) + src_s
            p_minus = p_minus * math.exp(-dt / tc_minus) - tgt_s
            elig = torch.outer(p_plus, tgt_s) + torch.outer(src_s, p_minus)
            w = torch.clamp(w, rng[0], rng[1])
        return w

    # ----------------------------------------------------------------------- #
    # Individual feature outputs                                              #
    # ----------------------------------------------------------------------- #

    def test_weight_feature_output(self):
        s = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0]])
        w = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        conn = self._make_mcc([tf.Weight(name="w", value=w.clone())], 3, 2, batch=2)
        assert torch.allclose(conn.compute(s), s @ w, atol=1e-6)

    def test_mask_feature_output(self):
        s = torch.tensor([[1.0, 1.0, 1.0]])
        w = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        mask = torch.tensor([[True, False], [True, True], [False, True]])
        conn = self._make_mcc(
            [tf.Weight(name="w", value=w.clone()), tf.Mask(name="m", value=mask.clone())],
            3,
            2,
        )
        assert torch.allclose(conn.compute(s), s @ (w * mask.float()), atol=1e-6)

    def test_bias_feature_output(self):
        # Bias is additive per-synapse; after the source-sum it adds bias.sum(0).
        s = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 0.0]])
        w = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        bias = torch.tensor([[0.5, -0.5], [0.1, 0.2], [0.0, 1.0]])
        conn = self._make_mcc(
            [tf.Weight(name="w", value=w.clone()), tf.Bias(name="b", value=bias.clone())],
            3,
            2,
            batch=2,
        )
        assert torch.allclose(conn.compute(s), s @ w + bias.sum(0), atol=1e-6)

    def test_intensity_feature_output(self):
        # Intensity's value is a per-synapse [src, tgt] tensor (a constant 2.0 here).
        s = torch.tensor([[1.0, 0.0, 1.0]])
        w = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        intensity = torch.full((3, 2), 2.0)
        conn = self._make_mcc(
            [
                tf.Weight(name="w", value=w.clone()),
                tf.Intensity(name="i", value=intensity.clone(), range=(-5.0, 5.0)),
            ],
            3,
            2,
        )
        assert torch.allclose(conn.compute(s), s @ (w * intensity), atol=1e-6)

    def test_degradation_feature_output(self):
        # Degradation subtracts degrade_function(value) per-synapse -> -sum(0).
        s = torch.tensor([[1.0, 1.0, 0.0]])
        w = torch.tensor([[0.4, 0.2], [0.3, 0.4], [0.5, 0.6]])
        deg = torch.tensor([[0.2, 0.4], [0.6, 0.8], [0.1, 0.3]])
        conn = self._make_mcc(
            [
                tf.Weight(name="w", value=w.clone()),
                tf.Degradation(
                    name="d", value=deg.clone(), degrade_function=lambda v: v * 0.5
                ),
            ],
            3,
            2,
        )
        assert torch.allclose(conn.compute(s), s @ w - (0.5 * deg).sum(0), atol=1e-6)

    def test_probability_feature_deterministic_bounds(self):
        # bernoulli(1) == 1 (always passes); bernoulli(0) == 0 (always blocked).
        s = torch.tensor([[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]])
        w = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]])
        passes = self._make_mcc(
            [
                tf.Probability(name="p", value=torch.ones(3, 2)),
                tf.Weight(name="w", value=w.clone()),
            ],
            3,
            2,
            batch=2,
        )
        assert torch.allclose(passes.compute(s), s @ w, atol=1e-6)
        blocked = self._make_mcc(
            [
                tf.Probability(name="p", value=torch.zeros(3, 2)),
                tf.Weight(name="w", value=w.clone()),
            ],
            3,
            2,
            batch=2,
        )
        assert torch.allclose(blocked.compute(s), torch.zeros(2, 2), atol=1e-6)

    def test_adaptation_features_output(self):
        for cls in (tf.AdaptationBaseSynapsHistory, tf.AdaptationBaseOtherSynaps):
            src_n, tgt_n = 3, 2
            s = torch.tensor([[1.0, 0.0, 1.0]])
            feat = cls(
                name="a",
                value=torch.zeros(src_n, tgt_n),
                ann_values=[torch.zeros(1, 1), torch.zeros(1, 1)],
            )
            conn = self._make_mcc([feat], src_n, tgt_n, batch=1)
            out = conn.compute(s)
            assert torch.allclose(feat.value.float(), torch.ones(src_n, tgt_n))
            assert torch.allclose(out, s @ torch.ones(src_n, tgt_n), atol=1e-6)

    # ----------------------------------------------------------------------- #
    # Combined pipelines == pre-collapse expansion                            #
    # ----------------------------------------------------------------------- #

    def test_pipeline_matches_expansion(self):
        torch.manual_seed(0)
        src_n, tgt_n, batch = 4, 3, 2
        s = (torch.rand(batch, src_n) > 0.4).float()
        w = torch.randn(src_n, tgt_n)
        w2 = torch.randn(src_n, tgt_n)
        mask = torch.rand(src_n, tgt_n) > 0.5
        bias = torch.randn(src_n, tgt_n) * 0.2
        deg = torch.rand(src_n, tgt_n)

        pipelines = {
            "weight": [tf.Weight(name="w", value=w.clone())],
            "weight+mask": [
                tf.Weight(name="w", value=w.clone()),
                tf.Mask(name="m", value=mask.clone()),
            ],
            "weight+bias": [
                tf.Weight(name="w", value=w.clone()),
                tf.Bias(name="b", value=bias.clone()),
            ],
            "weight+bias+degradation": [
                tf.Weight(name="w", value=w.clone()),
                tf.Bias(name="b", value=bias.clone()),
                tf.Degradation(
                    name="d", value=deg.clone(), degrade_function=lambda v: v * 0.3
                ),
            ],
            "weight+intensity+bias": [
                tf.Weight(name="w", value=w.clone()),
                tf.Intensity(
                    name="i", value=torch.full((src_n, tgt_n), 1.5), range=(-5.0, 5.0)
                ),
                tf.Bias(name="b", value=bias.clone()),
            ],
            "weight,bias,weight,bias": [
                tf.Weight(name="w", value=w.clone()),
                tf.Bias(name="b", value=bias.clone()),
                tf.Weight(name="w2", value=w2.clone()),
                tf.Bias(name="b2", value=(bias * 0.5).clone()),
            ],
        }

        for name, pipe in pipelines.items():
            conn = self._make_mcc(pipe, src_n, tgt_n, batch=batch)
            out = conn.compute(s)
            ref = self._reference_expansion(pipe, s, tgt_n)
            assert torch.allclose(
                out, ref, atol=1e-5
            ), f"{name}: {(out - ref).abs().max()}"

    # ----------------------------------------------------------------------- #
    # Sparse activity / sparse_compute == dense                               #
    # ----------------------------------------------------------------------- #

    def test_sparse_compute_matches_dense(self):
        torch.manual_seed(1)
        src_n, tgt_n, batch = 6, 4, 3
        w = torch.randn(src_n, tgt_n)
        bias = torch.randn(src_n, tgt_n) * 0.2
        deg = torch.rand(src_n, tgt_n)

        def pipes():
            return [
                [tf.Weight(name="w", value=w.clone())],
                [
                    tf.Weight(name="w", value=w.clone()),
                    tf.Bias(name="b", value=bias.clone()),
                ],
                [
                    tf.Weight(name="w", value=w.clone()),
                    tf.Degradation(
                        name="d", value=deg.clone(), degrade_function=lambda v: v * 0.4
                    ),
                ],
            ]

        spike_sets = [
            (torch.rand(batch, src_n) > 0.5).float(),
            torch.zeros(batch, src_n),  # empty-spike edge case
        ]
        for dense_pipe, sparse_pipe in zip(pipes(), pipes()):
            for s in spike_sets:
                dense = self._make_mcc(dense_pipe, src_n, tgt_n, batch=batch).compute(s)
                sparse = self._make_mcc(
                    sparse_pipe, src_n, tgt_n, batch=batch, sparse_compute=True
                ).compute(s)
                assert torch.allclose(dense, sparse, atol=1e-5)

    # ----------------------------------------------------------------------- #
    # MCC learning rules                                                      #
    # ----------------------------------------------------------------------- #

    def test_noop_leaves_weight_unchanged(self):
        w0 = torch.rand(3, 2)
        conn, feat = self._learning_conn(mcc.NoOp, w0, nu=(0.1, 0.1))
        conn.source.s = torch.ones(1, 3)
        conn.target.s = torch.ones(1, 2)
        feat.learning_rule.update(reward=1.0)
        assert torch.allclose(feat.value, w0)

    def test_postpre_predictable(self):
        # PostPre: dW = -nu0 * outer(src_s, tgt_x) + nu1 * outer(src_x, tgt_s).
        w0 = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.0, -0.2]])
        conn, feat = self._learning_conn(mcc.PostPre, w0, nu=(0.1, 0.2))
        conn.source.s = torch.tensor([[1.0, 0.0, 1.0]])
        conn.target.s = torch.tensor([[1.0, 1.0]])
        conn.source.x = torch.tensor([[0.3, 0.7, 0.2]])
        conn.target.x = torch.tensor([[0.5, 0.4]])
        feat.learning_rule.update()
        dw = -0.1 * torch.outer(conn.source.s[0], conn.target.x[0]) + 0.2 * torch.outer(
            conn.source.x[0], conn.target.s[0]
        )
        expected = torch.clamp(w0 + dw, -1.0, 1.0)
        assert torch.allclose(feat.value, expected, atol=1e-6)
        assert not torch.allclose(feat.value, w0)  # sanity: it actually changed

    def test_learning_respects_range_clamp(self):
        # Post-only potentiation of +5 per synapse must clamp to the range max (1.0).
        w0 = torch.full((2, 2), 0.9)
        conn, feat = self._learning_conn(mcc.PostPre, w0, nu=(0.0, 5.0), rng=(-1.0, 1.0))
        conn.source.s = torch.zeros(1, 2)
        conn.target.s = torch.ones(1, 2)
        conn.source.x = torch.ones(1, 2)
        conn.target.x = torch.zeros(1, 2)
        feat.learning_rule.update()
        assert torch.allclose(feat.value, torch.ones(2, 2))

    def test_mstdp_predictable(self):
        w0 = torch.zeros(3, 2)
        conn, feat = self._learning_conn(mcc.MSTDP, w0, nu=(0.5, 0.0))
        seq = self._mstdp_seq()
        for src_v, tgt_v, reward in seq:
            conn.source.s = src_v.unsqueeze(0)
            conn.target.s = tgt_v.unsqueeze(0)
            feat.learning_rule.update(reward=reward)
        expected = self._mstdp_reference(w0, seq, nu0=0.5, dt=1.0)
        assert torch.allclose(feat.value, expected, atol=1e-5)
        assert not torch.allclose(feat.value, w0)

    def test_mstdpet_predictable(self):
        w0 = torch.zeros(3, 2)
        conn, feat = self._learning_conn(mcc.MSTDPET, w0, nu=(0.5, 0.5))
        seq = self._mstdp_seq()
        for src_v, tgt_v, reward in seq:
            conn.source.s = src_v.unsqueeze(0)
            conn.target.s = tgt_v.unsqueeze(0)
            feat.learning_rule.update(reward=reward)
        expected = self._mstdpet_reference(w0, seq, nu0=0.5, dt=1.0)
        assert torch.allclose(feat.value, expected, atol=1e-5)
        assert not torch.allclose(feat.value, w0)

    def test_hebbian_predictable(self):
        # Hebbian: dW = nu0 * outer(src_s, tgt_x) + nu1 * outer(src_x, tgt_s), then
        # clamped. Both pre- and post-synaptic terms are positive (contrast PostPre,
        # which subtracts the pre term).
        w0 = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.0, -0.2]])
        conn, feat = self._learning_conn(mcc.Hebbian, w0, nu=(0.1, 0.2))
        conn.source.s = torch.tensor([[1.0, 0.0, 1.0]])
        conn.target.s = torch.tensor([[1.0, 1.0]])
        conn.source.x = torch.tensor([[0.3, 0.7, 0.2]])
        conn.target.x = torch.tensor([[0.5, 0.4]])
        feat.learning_rule.update()
        dw = 0.1 * torch.outer(conn.source.s[0], conn.target.x[0]) + 0.2 * torch.outer(
            conn.source.x[0], conn.target.s[0]
        )
        expected = torch.clamp(w0 + dw, -1.0, 1.0)
        assert torch.allclose(feat.value, expected, atol=1e-6)
        assert not torch.allclose(feat.value, w0)  # sanity: it actually changed


if __name__ == "__main__":
    # MulticompartmentConnection + MCC learning-rule tests.
    mcc_tester = TestMultiCompartmentConnection()
    mcc_tests = [
        mcc_tester.test_weight_feature_output,
        mcc_tester.test_mask_feature_output,
        mcc_tester.test_bias_feature_output,
        mcc_tester.test_intensity_feature_output,
        mcc_tester.test_degradation_feature_output,
        mcc_tester.test_probability_feature_deterministic_bounds,
        mcc_tester.test_meanfield_not_implemented,
        mcc_tester.test_adaptation_features_output,
        mcc_tester.test_pipeline_matches_expansion,
        mcc_tester.test_sparse_compute_matches_dense,
        mcc_tester.test_noop_leaves_weight_unchanged,
        mcc_tester.test_postpre_predictable,
        mcc_tester.test_learning_respects_range_clamp,
        mcc_tester.test_mstdp_predictable,
        mcc_tester.test_mstdpet_predictable,
        mcc_tester.test_hebbian_predictable,
    ]
    for mcc_test in mcc_tests:
        mcc_test()
        print(f"  PASSED: {mcc_test.__name__}")
    print(f"All {len(mcc_tests)} MulticompartmentConnection tests passed.")

    tester = TestConnection()

    # tester.test_transfer()

    # Connections with learning ability
    conn_types = [Connection, SparseConnection, Conv2dConnection, LocalConnection]
    args = [
        [[100], [50], (100, 50)],
        [[100], [50], (100, 50)],
        [[1, 28, 28], [1, 26, 26], (1, 1, 3, 3), 3],
        [[1, 28, 28], [1, 26, 26], (784, 676), 3, 1, 1],
    ]
    for update_rule in (Hebbian, PostPre, WeightDependentPostPre, MSTDP, MSTDPET, Rmax):
        print("Learning Rule:", update_rule)
        for conn_type, arg in zip(conn_types, args):
            tester.test_weights(conn_type, nu=1e-2, update_rule=update_rule, *arg)

    # Other connections
    # Note: Does not include MaxPool2dConnection because this connection
    # does not utilize weights and wmin/wmax
    conn_types = [MeanFieldConnection]
    args = [[[1, 28, 28], [1, 26, 26], (1, 26), 3, 1]]
    for conn_type, arg in zip(conn_types, args):
        tester.test_weights(conn_type, decay=1, update_rule=NoOp, *arg)
