import torch

from bindsnet.network.nodes import Input, McCullochPitts, IFNodes, LIFNodes, AdaptiveLIFNodes


class TestNodes:
    """
    Tests all stable groups of neurons / nodes.
    """

    def test_init(self):
        for nodes in [Input, McCullochPitts, IFNodes, LIFNodes, AdaptiveLIFNodes]:
            for n in [1, 100, 10000]:
                layer = nodes(n)

                assert layer.n == n
                assert (layer.s.float() == torch.zeros(n)).all()

                if nodes in [LIFNodes,
                             AdaptiveLIFNodes]:
                    assert (layer.v == layer.rest * torch.ones(n)).all()

                layer = nodes(n, traces=True, trace_tc=1e-5)

                assert layer.n == n;
                assert layer.trace_tc == 1e-5
                assert (layer.s.float() == torch.zeros(n)).all()
                assert (layer.x == torch.zeros(n)).all()
                assert (layer.x == torch.zeros(n)).all()

                if nodes in [LIFNodes,
                             AdaptiveLIFNodes]:
                    assert (layer.v == layer.rest * torch.ones(n)).all()

        for nodes in [LIFNodes, AdaptiveLIFNodes]:
            for n in [1, 100, 10000]:
                layer = nodes(n, rest=0.0, reset=-10.0, thresh=10.0, refrac=3, decay=7e-4)

                assert layer.rest == 0.0;
                assert layer.reset == -10.0;
                assert layer.thresh == 10.0
                assert layer.refrac == 3;
                assert layer.decay == 7e-4
                assert (layer.s.float() == torch.zeros(n)).all()
                assert (layer.v == layer.rest * torch.ones(n)).all()
