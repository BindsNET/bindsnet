import torch

from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes, SRM0Nodes

from bindsnet.network.topology import *

from bindsnet.learning import (
    Hebbian,
    PostPre,
    WeightDependentPostPre,
    MSTDP,
    MSTDPET,
    Rmax,
    NoOp,
)
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
        wmins = [-np.inf, 0, torch.zeros(*shape_w)]
        wmaxes = [np.inf, 0, torch.zeros(*shape_w)]
        for w in weights:
            for wmin in wmins:
                for wmax in wmaxes:
                    print(
                        f"- w: {type(w).__name__}, "
                        f"wmin: {type(wmax).__name__}, wmax: {type(wmax).__name__}"
                    )
                    if kwargs.get('update_rule') == Rmax:
                        l_a = SRM0Nodes(shape= shape_a, traces = True, traces_additive = True)
                        l_b = SRM0Nodes(shape= shape_b, traces = True, traces_additive = True)
                    else:
                        l_a = LIFNodes(shape= shape_a, traces = True, traces_additive = True)
                        l_b = LIFNodes(shape= shape_b, traces = True, traces_additive = True)


                    network = Network(dt=1.0)
                    network.add_layer(
                        Input(n=100, traces = True, traces_additive = True),
                        name="input"
                    )
                    network.add_layer(l_a, name="a")
                    network.add_layer(l_b, name="b")

                    network.add_connection(
                        conn_type(l_a, l_b, w = w, wmin =wmin, wmax =wmax,
                                 *args, **kwargs), source= 'a', target= 'b')
                    network.add_connection(
                        Connection(
                            source=network.layers["input"],
                            target=network.layers["a"],
                            **kwargs,
                        ),
                        source="input",
                        target="a",
                    )
                    network.run(
                        inputs={"input": torch.bernoulli(torch.rand(time, 100)).byte()}, 
                        time=time,
                        reward = 1,
                    )

if __name__ == "__main__":
    tester = TestConnection()

    tester.test_transfer()

    # Connections with learning ability
    conn_types = [Connection, Conv2dConnection, LocalConnection]
    args =[
        [[100], [50], (100,50)], 
        [[1,28,28], [1,26,26], (1,1,3,3), 3,],
        [[1,28,28], [1,26,26], (784, 676), 3, 1, 1,]
    ]
    for update_rule in (
        Hebbian,
        PostPre,
        # WeightDependentPostPre,
        MSTDP,
        MSTDPET,
        # Rmax,
    ):
        print("Learning Rule:", update_rule)
        for conn_type, arg in zip(conn_types, args):
            tester.test_weights(conn_type, nu = 1E-2, update_rule=update_rule, *arg)
    
    # Other connections
    conn_types = [MeanFieldConnection, MaxPool2dConnection]
    args =[
        [[1,28,28], [1,26,26], (1,1,3,3), 3, 1], 
        [[100], [50], [1]],
    ]
    for conn_type, arg in zip(conn_types, args):
        tester.test_weights(conn_type, decay = 1, update_rule = NoOp, *arg)