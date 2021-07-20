import torch

from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes

from bindsnet.network.topology import *

from bindsnet.learning import Hebbian, PostPre

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
        time = 100
        weights = [None, torch.Tensor(*shape_w)]
        wmins = [-np.inf, 0, torch.zeros(*shape_w)]
        wmaxes = [np.inf, 0, torch.zeros(*shape_w)]
        for w in weights:
            for wmin in wmins:
                for wmax in wmaxes:
                    print(f"Testing {conn_type} with: w={w}, wmin={wmin}, wmax={wmax}")
                    l_a = LIFNodes(shape= shape_a, traces = True)
                    l_b = LIFNodes(shape= shape_b, traces = True)
                    conn = conn_type(l_a, l_b, w = w, wmin =wmin, wmax =wmax,
                                     *args, **kwargs)

                    network = Network(dt=1.0)
                    network.add_layer(l_a, name="a")
                    network.add_layer(l_b, name="b")
                    network.add_connection(conn, source= 'a', target= 'b')
                    network.run(
                        inputs={}, 
                        time=time
                    )

if __name__ == "__main__":
    tester = TestConnection()

    # tester.test_transfer()
    tester.test_weights(Connection, [100], [50], (100,50),
                        nu = 1E-2, update_rule=Hebbian)
    tester.test_weights(Conv2dConnection, [1,28,28], [1,26,26], (1,1,3,3), 3, norm = 2,
                        nu = 1E-2, update_rule=Hebbian)
    tester.test_weights(LocalConnection, [1,28,28], [1,26,26], (784, 676), 3, 1, 1,
                        nu = 1E-2, update_rule=Hebbian)
