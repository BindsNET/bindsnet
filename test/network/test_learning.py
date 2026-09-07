import pytest
import torch

from bindsnet.learning import (
    MSTDP,
    MSTDPET,
    Hebbian,
)
from bindsnet.learning import MCC_learning as mcc
from bindsnet.learning import PostPre, Rmax, WeightDependentPostPre
from bindsnet.network import Network
from bindsnet.network import topology_features as tf
from bindsnet.network.nodes import CSRMNodes, Input, LIFNodes, SRM0Nodes
from bindsnet.network.topology import (
    Connection,
    Conv2dConnection,
    MulticompartmentConnection,
)


class TestLearningRules:
    """
    Tests all stable learning rules for compatible ``Connection`` types.
    """

    def test_hebbian(self):
        # Connection test
        network = Network(dt=1.0)
        network.add_layer(Input(n=100, traces=True), name="input")
        network.add_layer(LIFNodes(n=100, traces=True), name="output")
        network.add_connection(
            Connection(
                source=network.layers["input"],
                target=network.layers["output"],
                nu=1e-2,
                update_rule=Hebbian,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 100)).byte()}, time=250
        )

        # Conv2dConnection test
        network = Network(dt=1.0)
        network.add_layer(Input(shape=[1, 10, 10], traces=True), name="input")
        network.add_layer(LIFNodes(shape=[32, 8, 8], traces=True), name="output")
        network.add_connection(
            Conv2dConnection(
                source=network.layers["input"],
                target=network.layers["output"],
                kernel_size=3,
                stride=1,
                nu=1e-2,
                update_rule=Hebbian,
            ),
            source="input",
            target="output",
        )
        # shape is [time, batch, channels, height, width]
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 1, 1, 10, 10)).byte()},
            time=250,
        )

    def test_post_pre(self):
        # Connection test
        network = Network(dt=1.0)
        network.add_layer(Input(n=100, traces=True), name="input")
        network.add_layer(LIFNodes(n=100, traces=True), name="output")
        network.add_connection(
            Connection(
                source=network.layers["input"],
                target=network.layers["output"],
                nu=1e-2,
                update_rule=PostPre,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 100)).byte()}, time=250
        )

        network2 = Network(dt=1.0)
        network2.add_layer(Input(n=100, traces=True), name="input")
        network2.add_layer(CSRMNodes(n=100, traces=True), name="output")
        network2.add_connection(
            Connection(
                source=network2.layers["input"],
                target=network2.layers["output"],
                nu=1e-2,
                update_rule=PostPre,
            ),
            source="input",
            target="output",
        )
        network2.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 100)).byte()}, time=250
        )

        # Conv2dConnection test
        network = Network(dt=1.0)
        network.add_layer(Input(shape=[1, 10, 10], traces=True), name="input")
        network.add_layer(LIFNodes(shape=[32, 8, 8], traces=True), name="output")
        network.add_connection(
            Conv2dConnection(
                source=network.layers["input"],
                target=network.layers["output"],
                kernel_size=3,
                stride=1,
                nu=1e-2,
                update_rule=PostPre,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 1, 1, 10, 10)).byte()},
            time=250,
        )

    def test_weight_dependent_post_pre(self):
        # Connection test
        network = Network(dt=1.0)
        network.add_layer(Input(n=100, traces=True), name="input")
        network.add_layer(LIFNodes(n=100, traces=True), name="output")
        network.add_connection(
            Connection(
                source=network.layers["input"],
                target=network.layers["output"],
                nu=1e-2,
                update_rule=WeightDependentPostPre,
                wmin=-1,
                wmax=1,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 100)).byte()}, time=250
        )

        # Conv2dConnection test
        network = Network(dt=1.0)
        network.add_layer(Input(shape=[1, 10, 10], traces=True), name="input")
        network.add_layer(LIFNodes(shape=[32, 8, 8], traces=True), name="output")
        network.add_connection(
            Conv2dConnection(
                source=network.layers["input"],
                target=network.layers["output"],
                kernel_size=3,
                stride=1,
                nu=1e-2,
                update_rule=WeightDependentPostPre,
                wmin=-1,
                wmax=1,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 1, 1, 10, 10)).byte()},
            time=250,
        )

    def test_mstdp(self):
        # Connection test
        network = Network(dt=1.0)
        network.add_layer(Input(n=100), name="input")
        network.add_layer(LIFNodes(n=100), name="output")
        network.add_connection(
            Connection(
                source=network.layers["input"],
                target=network.layers["output"],
                nu=1e-2,
                update_rule=MSTDP,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 100)).byte()},
            time=250,
            reward=1.0,
        )

        # Conv2dConnection test
        network = Network(dt=1.0)
        network.add_layer(Input(shape=[1, 10, 10]), name="input")
        network.add_layer(LIFNodes(shape=[32, 8, 8]), name="output")
        network.add_connection(
            Conv2dConnection(
                source=network.layers["input"],
                target=network.layers["output"],
                kernel_size=3,
                stride=1,
                nu=1e-2,
                update_rule=MSTDP,
            ),
            source="input",
            target="output",
        )

        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 1, 1, 10, 10)).byte()},
            time=250,
            reward=1.0,
        )

    def test_mstdpet(self):
        # Connection test
        network = Network(dt=1.0)
        network.add_layer(Input(n=100), name="input")
        network.add_layer(LIFNodes(n=100), name="output")
        network.add_connection(
            Connection(
                source=network.layers["input"],
                target=network.layers["output"],
                nu=1e-2,
                update_rule=MSTDPET,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 100)).byte()},
            time=250,
            reward=1.0,
        )

        # Conv2dConnection test
        network = Network(dt=1.0)
        network.add_layer(Input(shape=[1, 10, 10]), name="input")
        network.add_layer(LIFNodes(shape=[32, 8, 8]), name="output")
        network.add_connection(
            Conv2dConnection(
                source=network.layers["input"],
                target=network.layers["output"],
                kernel_size=3,
                stride=1,
                nu=1e-2,
                update_rule=MSTDPET,
            ),
            source="input",
            target="output",
        )

        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 1, 1, 10, 10)).byte()},
            time=250,
            reward=1.0,
        )

    def test_rmax(self):
        # Connection test
        network = Network(dt=1.0)
        network.add_layer(Input(n=100, traces=True, traces_additive=True), name="input")
        network.add_layer(SRM0Nodes(n=100), name="output")
        network.add_connection(
            Connection(
                source=network.layers["input"],
                target=network.layers["output"],
                nu=1e-2,
                update_rule=Rmax,
            ),
            source="input",
            target="output",
        )
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(250, 100)).byte()},
            time=250,
            reward=1.0,
        )


class TestLearningRuleReset:
    """
    ``network.reset_state_variables()`` must clear every variable a
    ``MulticompartmentConnection`` learning rule carries between time steps.

    Regression for #777: the reset never reached the learning rules at all,
    because every feature overrode ``reset_state_variables`` with a bare
    ``pass``, and the rules that did get called cleared only part of their
    state.
    """

    @staticmethod
    def _build(rule, n=8, **rule_kwargs):
        """A one-connection network whose single Weight uses ``rule``."""
        network = Network(dt=1.0)
        network.add_layer(Input(n=n, traces=True), name="input")
        network.add_layer(LIFNodes(n=n, traces=True), name="output")
        weight = tf.Weight(
            name="w",
            value=torch.rand(n, n),
            range=[0.0, 1.0],
            nu=(1e-2, 1e-2),
            learning_rule=rule,
        )
        connection = MulticompartmentConnection(
            source=network.layers["input"],
            target=network.layers["output"],
            device="cpu",
            pipeline=[weight],
            **rule_kwargs,
        )
        network.add_connection(connection, source="input", target="output")
        return network, connection.pipeline[0].learning_rule

    @staticmethod
    def _drive(network, n=8, time=100, seed=0):
        torch.manual_seed(seed)
        network.run(
            inputs={"input": torch.bernoulli(torch.rand(time, n)).byte()},
            time=time,
            reward=1.0,
        )

    def test_reset_reaches_the_learning_rule(self):
        # The bug behind #777: MSTDPET's reset was never invoked, so even the
        # two variables it did clear survived a network reset.
        network, rule = self._build(mcc.MSTDPET)
        self._drive(network)
        assert rule.eligibility_trace.abs().sum() > 0  # sanity: state exists
        network.reset_state_variables()
        assert torch.all(rule.eligibility_trace == 0)

    def test_mstdpet_reset_clears_all_state(self):
        # Reported by @saachigoyall in #777: p_plus, p_minus and the
        # moving-average buffer were left untouched.
        network, rule = self._build(
            mcc.MSTDPET, average_update=5, continues_update=True
        )
        self._drive(network)
        assert rule.p_plus.abs().sum() > 0
        assert rule.p_minus.abs().sum() > 0
        assert rule.average_buffer.abs().sum() > 0

        network.reset_state_variables()

        assert torch.all(rule.eligibility == 0)
        assert torch.all(rule.eligibility_trace == 0)
        assert torch.all(rule.p_plus == 0)
        assert torch.all(rule.p_minus == 0)
        assert torch.all(rule.average_buffer == 0)
        assert rule.average_buffer_index == 0

    def test_mstdp_reset_clears_all_state(self):
        # MSTDP's reset was a bare ``return``, clearing nothing.
        network, rule = self._build(mcc.MSTDP, average_update=5, continues_update=True)
        self._drive(network)
        assert rule.p_plus.abs().sum() > 0
        assert rule.p_minus.abs().sum() > 0

        network.reset_state_variables()

        assert rule.eligibility is None or torch.all(rule.eligibility == 0)
        assert torch.all(rule.p_plus == 0)
        assert torch.all(rule.p_minus == 0)
        assert torch.all(rule.average_buffer == 0)
        assert rule.average_buffer_index == 0

    def test_mstdp_reset_clears_fast_path_spike_lag(self):
        # The fast path keeps the previous step's spikes for its rank-1 update.
        # Left in place, the first step of a new episode pairs with the last
        # step of the old one.
        network, rule = self._build(mcc.MSTDP)
        self._drive(network)
        assert rule._prev_source_s is not None
        assert rule._prev_target_s is not None

        network.reset_state_variables()

        assert rule._prev_source_s is None
        assert rule._prev_target_s is None

    @pytest.mark.parametrize(
        "rule", [mcc.MSTDP, mcc.MSTDPET, mcc.PostPre, mcc.Hebbian, mcc.DiehlAndCook]
    )
    def test_reset_before_first_run_does_not_raise(self, rule):
        # Some of this state is built lazily on the first update, because only
        # then are the batch size and device known. Resetting a network before
        # running it must still work.
        network, rule_obj = self._build(rule)
        network.reset_state_variables()
        assert rule_obj is not None

    def test_postpre_reset_clears_average_buffers(self):
        # PostPre's reset was a bare ``return``; both buffers survived.
        network, rule = self._build(
            mcc.PostPre, average_update=5, continues_update=True
        )
        self._drive(network)

        network.reset_state_variables()

        assert torch.all(rule.average_buffer_pre == 0)
        assert torch.all(rule.average_buffer_post == 0)
        assert rule.average_buffer_index_pre == 0
        assert rule.average_buffer_index_post == 0

    @pytest.mark.parametrize(
        "rule", [mcc.MSTDP, mcc.MSTDPET, mcc.PostPre, mcc.Hebbian, mcc.DiehlAndCook]
    )
    def test_episodes_are_independent_after_reset(self, rule):
        # The symptom #777 reported: with a reset between them, two identical
        # episodes must produce identical weights. Before the fix the second
        # episode started from the first one's leftover state.
        network, _ = self._build(rule)
        feature = network.connections[("input", "output")].pipeline[0]
        w0 = feature.value.clone()

        self._drive(network, seed=1)
        after_first = feature.value.clone()

        network.reset_state_variables()
        with torch.no_grad():
            feature.value.copy_(w0)
        self._drive(network, seed=1)
        after_second = feature.value.clone()

        assert not torch.allclose(after_first, w0)  # sanity: learning happened
        assert torch.allclose(after_first, after_second, atol=1e-6)
