import pytest
import torch

from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import (
    Conv1dConnection,
    Conv2dConnection,
    Conv3dConnection,
    LocalConnection1D,
    LocalConnection2D,
    LocalConnection3D,
)

CONNECTIONS = [
    (Conv1dConnection, 1),
    (Conv2dConnection, 2),
    (Conv3dConnection, 3),
    (LocalConnection1D, 1),
    (LocalConnection2D, 2),
    (LocalConnection3D, 3),
]


def _connection(connection_type, dimensions, dtype, norm):
    return connection_type(
        Input(shape=(1, *((4,) * dimensions))),
        LIFNodes(shape=(2, *((3,) * dimensions))),
        kernel_size=2,
        stride=1,
        n_filters=2,
        norm=norm,
        w_dtype=dtype,
    )


@pytest.mark.parametrize(("connection_type", "dimensions"), CONNECTIONS)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("norm", [None, 0.0, 2.0])
def test_zero_kernel_normalization(connection_type, dimensions, dtype, norm):
    connection = _connection(connection_type, dimensions, dtype, norm)
    kernel_elements = 2**dimensions
    rows = connection.w.numel() // kernel_elements
    ramp = torch.arange(1, kernel_elements + 1, dtype=dtype)
    weights = ramp.repeat(rows, 1)
    weights[0].zero_()
    with torch.no_grad():
        connection.w.copy_(weights.reshape_as(connection.w))
    parameter = connection.w
    expected = weights.clone()
    if norm is not None:
        expected *= norm / (kernel_elements * (kernel_elements + 1) / 2)
    for _ in range(2):
        connection.normalize()
        assert connection.w is parameter
        torch.testing.assert_close(connection.w.reshape(rows, -1), expected)
    if dtype == torch.float32:
        output = connection.compute(torch.ones((1, 1, *((4,) * dimensions))))
        assert torch.isfinite(output).all()


@pytest.mark.parametrize(("connection_type", "dimensions"), CONNECTIONS)
def test_nonzero_signed_cancellation_is_not_a_zero_kernel(connection_type, dimensions):
    connection = _connection(connection_type, dimensions, torch.float32, 2.0)
    with torch.no_grad():
        weights = connection.w.reshape(-1, 2**dimensions)
        weights[0].zero_()
        weights[0, 0], weights[0, 1] = 1.0, -1.0
    connection.normalize()
    actual = connection.w.reshape(-1, 2**dimensions)
    assert torch.isposinf(actual[0, 0])
    assert torch.isneginf(actual[0, 1])
