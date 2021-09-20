import torch

from bindsnet.network import Network
from bindsnet.network.nodes import (
    AdaptiveLIFNodes,
    IFNodes,
    Input,
    LIFNodes,
    McCullochPitts,
    Nodes,
    SRM0Nodes,
)


class TestNodes:
    """
    Tests all stable groups of neurons / nodes.
    """

    def test_init(self):
        network = Network()
        for i, nodes in enumerate(
            [Input, McCullochPitts, IFNodes, LIFNodes, AdaptiveLIFNodes, SRM0Nodes]
        ):
            for n in [1, 100, 10000]:
                layer = nodes(n)
                network.add_layer(layer=layer, name=f"{i}_{n}")

                assert layer.n == n
                assert (layer.s.float() == torch.zeros(n)).all()

                if nodes in [LIFNodes, AdaptiveLIFNodes]:
                    assert (layer.v == layer.rest * torch.ones(n)).all()

                layer = nodes(n, traces=True, tc_trace=1e5)
                network.add_layer(layer=layer, name=f"{i}_traces_{n}")

                assert layer.n == n
                assert layer.tc_trace == 1e5
                assert (layer.s.float() == torch.zeros(n)).all()
                assert (layer.x == torch.zeros(n)).all()
                assert (layer.x == torch.zeros(n)).all()

                if nodes in [LIFNodes, AdaptiveLIFNodes, SRM0Nodes]:
                    assert (layer.v == layer.rest * torch.ones(n)).all()

        for nodes in [LIFNodes, AdaptiveLIFNodes]:
            for n in [1, 100, 10000]:
                layer = nodes(
                    n, rest=0.0, reset=-10.0, thresh=10.0, refrac=3, tc_decay=1.5e3
                )
                network.add_layer(layer=layer, name=f"{i}_params_{n}")

                assert layer.rest == 0.0
                assert layer.reset == -10.0
                assert layer.thresh == 10.0
                assert layer.refrac == 3
                assert layer.tc_decay == 1.5e3
                assert (layer.s.float() == torch.zeros(n)).all()
                assert (layer.v == layer.rest * torch.ones(n)).all()

    def test_transfer(self):
        if not torch.cuda.is_available():
            return

        for nodes in Nodes.__subclasses__():
            layer = nodes(10)

            layer.to(torch.device("cuda:0"))

            layer_tensors = [
                k for k, v in layer.state_dict().items() if isinstance(v, torch.Tensor)
            ]

            tensor_devs = [getattr(layer, k).device for k in layer_tensors]

            print("State dict in {} : {}".format(nodes, layer.state_dict().keys()))
            print("__dict__ in {} : {}".format(nodes, layer.__dict__.keys()))
            print("Tensors in {} : {}".format(nodes, layer_tensors))
            print("Tensor devices {}".format(list(zip(layer_tensors, tensor_devs))))

            for d in tensor_devs:
                print(d, d == torch.device("cuda:0"))
                assert d == torch.device("cuda:0")

            print("Reset layer")
            layer.reset_state_variables()
            layer_tensors = [
                k for k, v in layer.state_dict().items() if isinstance(v, torch.Tensor)
            ]

            tensor_devs = [getattr(layer, k).device for k in layer_tensors]

            for d in tensor_devs:
                print(d, d == torch.device("cuda:0"))
                assert d == torch.device("cuda:0")


if __name__ == "__main__":
    tester = TestNodes()

    tester.test_init()
    tester.test_transfer()
