import torch

from bindsnet.network.nodes import (
    Input,
    McCullochPitts,
    IFNodes,
    LIFNodes,
    AdaptiveLIFNodes,
    SRM0Nodes,
)


class TestNodes:
    """
    Tests all stable groups of neurons / nodes.
    """

    def test_init(self):
        for nodes in [Input, McCullochPitts, IFNodes, LIFNodes, AdaptiveLIFNodes, SRM0Nodes]:
            for n in [1, 100, 10000]:
                layer = nodes(n)

                assert layer.n == n
                assert (layer.s.float() == torch.zeros(n)).all()

                if nodes in [LIFNodes, AdaptiveLIFNodes]:
                    assert (layer.v == layer.rest * torch.ones(n)).all()

                layer = nodes(n, traces=True, tc_trace=1e5)

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

                assert layer.rest == 0.0
                assert layer.reset == -10.0
                assert layer.thresh == 10.0
                assert layer.refrac == 3
                assert layer.tc_decay == 1.5e3
                assert (layer.s.float() == torch.zeros(n)).all()
                assert (layer.v == layer.rest * torch.ones(n)).all()
