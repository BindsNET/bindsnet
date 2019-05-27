import torch

from bindsnet.network.nodes import (
    LIFNodes,
)

from bindsnet.network.topology import AbstractConnection


class TestConnection:
    """
    Tests all stable groups of neurons / nodes.
    """

    def test_transfer(self):
        for conn_type in Connection.__subclasses__():
            l_a = LIFNodes(10)
            l_b = LIFNodes(10)
            connection = conn_type(l_a, l_b)

            connection.to(torch.device('cuda:0'))

            connection_tensors = [k for k, v in connection.state_dict().items() if
                    isinstance(v, torch.Tensor)]

            tensor_devs = [getattr(connection,k).device for k in connection_tensors]

            print("State dict in {} : {}".format(nodes,
                connection.state_dict().keys()))
            print("__dict__ in {} : {}".format(nodes,
                connection.__dict__.keys()))
            print("Tensors in {} : {}".format(nodes, connection_tensors))
            print("Tensor devices {}".format(list(zip(connection_tensors,
                tensor_devs))))

            for d in tensor_devs:
                print(d, d==torch.device('cuda:0'))
                assert d == torch.device('cuda:0')


if __name__ == "__main__":
    tester = TestConnections()

    tester.test_transfer()
