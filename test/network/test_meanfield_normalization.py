import pytest
import torch

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MeanFieldConnection


@pytest.mark.parametrize("weight_shape", [(), (4,), (2, 2)])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("norm", [None, 2.0])
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
        expected = weights * 0.2 if weight_shape else torch.tensor(2.0, dtype=dtype)

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


def test_network_run_normalizes_meanfield_connection():
    network = Network(dt=1.0)
    source, target = Input(n=2), LIFNodes(n=3)
    network.add_layer(source, name="source")
    network.add_layer(target, name="target")
    connection = MeanFieldConnection(source, target, w=torch.tensor(0.25), norm=2.0)
    network.add_connection(connection, source="source", target="target")
    network.run(inputs={"source": torch.ones(2, 1, 2)}, time=2)
    torch.testing.assert_close(connection.w, torch.tensor(2.0))
    assert torch.isfinite(target.v).all()
