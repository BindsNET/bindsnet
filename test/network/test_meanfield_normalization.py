import pytest
import torch

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MeanFieldConnection


@pytest.mark.parametrize("weight_shape", [(), (4,), (2, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("norm", [None, 0.0, 2.0])
def test_meanfield_normalization_preserves_parameter(weight_shape, dtype, norm):
    target_shape = weight_shape or (3,)
    weights = (
        torch.arange(1, 5, dtype=dtype).reshape(weight_shape)
        if weight_shape
        else torch.tensor(0.25, dtype=dtype)
    )
    connection = MeanFieldConnection(
        Input(n=2),
        LIFNodes(shape=target_shape),
        w=weights.clone(),
        w_dtype=dtype,
        norm=norm,
    )
    parameter = connection.w
    state_keys = set(connection.state_dict())
    expected = weights.clone()
    if norm is not None:
        expected = (
            weights * (norm / 10)
            if weight_shape
            else torch.tensor(norm / 3, dtype=dtype)
        )

    connection.normalize()
    assert connection.w is parameter
    assert isinstance(connection.w, torch.nn.Parameter)
    assert connection.w.shape == weights.shape
    assert set(connection.state_dict()) == state_keys
    torch.testing.assert_close(connection.w, expected)
    spikes = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    torch.testing.assert_close(connection.compute(spikes), 0.75 * expected)
    connection.normalize()
    torch.testing.assert_close(connection.w, expected)


@pytest.mark.parametrize("norm", [0.0, 2.0])
def test_network_run_normalizes_meanfield_connection(norm):
    network = Network(dt=1.0)
    source, target = Input(n=2), LIFNodes(n=3)
    network.add_layer(source, name="source")
    network.add_layer(target, name="target")
    connection = MeanFieldConnection(source, target, w=torch.tensor(0.25), norm=norm)
    network.add_connection(connection, source="source", target="target")
    for _ in range(2):
        network.run(inputs={"source": torch.ones(2, 1, 2)}, time=2)
        torch.testing.assert_close(connection.w, torch.tensor(norm / target.n))
        assert torch.isfinite(target.v).all()


@pytest.mark.parametrize(
    "weights,expected",
    [
        ([-1.0, -2.0], [-2 / 3, -4 / 3]),
        ([1.0, -1.0], [1.0, -1.0]),
        ([1.0, -0.999999], [2 / 1.999999, -1.999998 / 1.999999]),
    ],
)
def test_meanfield_normalization_preserves_signed_weights(weights, expected):
    connection = MeanFieldConnection(
        Input(n=2), LIFNodes(n=2), w=torch.tensor(weights), norm=2.0
    )
    for _ in range(2):
        connection.normalize()
        assert torch.isfinite(connection.w).all()
        torch.testing.assert_close(connection.w, torch.tensor(expected))


@pytest.mark.parametrize("value", [-0.25, 0.25])
@pytest.mark.parametrize("target_shape", [(3,), (2, 2)])
def test_meanfield_shared_and_per_target_weights_agree(value, target_shape):
    source, target = Input(n=2), LIFNodes(shape=target_shape)
    shared = MeanFieldConnection(source, target, w=torch.tensor(value), norm=2.0)
    per_target = MeanFieldConnection(
        source, target, w=torch.full(target_shape, value), norm=2.0
    )
    spikes = torch.tensor([[0.0, 1.0], [1.0, 1.0]])
    for _ in range(2):
        shared.normalize()
        per_target.normalize()
        expected = torch.full(target_shape, (2.0 if value > 0 else -2.0) / target.n)
        torch.testing.assert_close(shared.w.expand(target_shape), expected)
        torch.testing.assert_close(per_target.w, expected)
        torch.testing.assert_close(
            shared.compute(spikes).expand(target_shape), per_target.compute(spikes)
        )
