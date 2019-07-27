import torch

from bindsnet.network.nodes import LIFNodes

from bindsnet.network.topology import *


class TestConnection:
    """
    Tests all stable groups of neurons / nodes.
    """

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

            connection.to(torch.device("cuda:0"))

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


if __name__ == "__main__":
    tester = TestConnection()

    tester.test_transfer()
