import torch

from bindsnet.network.nodes import *


class TestNodes:
    """
    Tests groups of neurons (nodes).
    """

    def test_nodes(self):
        for node_type in [Input, McCullochPitts, IFNodes, LIFNodes, AdaptiveLIFNodes, CurrentLIFNodes,
                          AdaptiveCurrentLIFNodes, DiehlAndCookNodes, IzhikevichNodes]:
            for n in [1, 100, 10000]:
                layer = node_type(n)

                assert layer.n == n
                assert (layer.s.float() == torch.zeros(n)).all()

                if node_type in [LIFNodes, AdaptiveLIFNodes]:
                    assert (layer.v == layer.rest * torch.ones(n)).all()

                layer = node_type(n, traces=True, trace_tc=1e-5)

                assert layer.n == n;
                assert layer.trace_tc == 1e-5
                assert (layer.s.float() == torch.zeros(n)).all()
                assert (layer.x == torch.zeros(n)).all()
                assert (layer.x == torch.zeros(n)).all()

                inpts = torch.bernoulli(torch.rand(n))
                layer.step(inpts=inpts, dt=1.0)
                layer.reset_()

                if node_type in [LIFNodes, AdaptiveLIFNodes, CurrentLIFNodes, AdaptiveCurrentLIFNodes,
                                 DiehlAndCookNodes, IzhikevichNodes]:
                    assert (layer.v == layer.rest * torch.ones(n)).all()

                    layer = node_type(n, rest=0.0, reset=-10.0, thresh=10.0, refrac=3, decay=7e-4)

                    assert layer.rest == 0.0;
                    assert layer.reset == -10.0;
                    assert layer.thresh == 10.0
                    assert layer.refrac == 3;
                    assert layer.decay == 7e-4
                    assert (layer.s.float() == torch.zeros(n)).all()
                    assert (layer.v == layer.rest * torch.ones(n)).all()
