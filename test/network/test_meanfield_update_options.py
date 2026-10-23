import pytest
import torch

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MeanFieldConnection


@pytest.mark.parametrize("weight_decay", [0.0, 0.1, 1.0])
@pytest.mark.parametrize("shared", [False, True])
def test_meanfield_weight_decay_during_network_run(weight_decay, shared):
    network = Network()
    source, target = Input(n=2), LIFNodes(n=3)
    network.add_layer(source, "X")
    network.add_layer(target, "Y")
    initial = torch.tensor(0.5) if shared else torch.tensor([0.25, 0.5, 0.75])
    connection = MeanFieldConnection(
        source, target, w=initial.clone(), weight_decay=weight_decay
    )
    network.add_connection(connection, "X", "Y")
    parameter = connection.w
    # No STDP rule or normalization: the common learning-rule decay acts alone.
    for step in range(1, 4):
        network.run({"X": torch.zeros(1, 1, 2)}, time=1)
        expected = initial * (1 - weight_decay) ** step
        torch.testing.assert_close(connection.w, expected)
        assert connection.w is parameter


@pytest.mark.parametrize(
    "reduction,expected",
    [(torch.sum, [5.0, 7.0, 9.0]), (torch.mean, [2.5, 3.5, 4.5])],
)
def test_meanfield_accepts_update_reduction(reduction, expected):
    connection = MeanFieldConnection(
        Input(n=2),
        LIFNodes(n=3),
        w=torch.tensor(0.5),
        weight_decay=0.1,
        reduction=reduction,
    )
    updates = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    actual = connection.update_rule.reduction(updates, dim=0)
    torch.testing.assert_close(actual, torch.tensor(expected))
