import torch

from bindsnet.encoding import *


class TestEncodings:
    """
    Tests all stable encoding functions and generators.
    """

    def test_bernoulli(self):
        for n in [1, 100]:  # number of nodes in layer
            for t in [1, 100]:  # number of timesteps
                for m in [0.1, 1.0]:  # maximum spiking probability
                    datum = torch.empty(n).uniform_(0, m)
                    spikes = bernoulli(datum, time=t, max_prob=m)

                    assert spikes.size() == torch.Size((t, n))

    def test_multidim_bernoulli(self):
        for shape in [[5, 5], [10, 10], [25, 25]]:  # shape of nodes in layer
            for t in [1, 100]:  # number of timesteps
                for m in [0.1, 1.0]:  # maximum spiking probability
                    datum = torch.empty(shape).uniform_(0, m)
                    spikes = bernoulli(datum, time=t, max_prob=m)

                    assert spikes.size() == torch.Size((t, *shape))

    def test_bernoulli_loader(self):
        for s in [1, 100]:  # number of data samples
            for n in [1, 100]:  # number of nodes in layer
                for m in [0.1, 1.0]:  # maximum spiking probability
                    for t in [1, 100]:  # number of timesteps
                        data = torch.empty(s, n).uniform_(0, 1)
                        spike_loader = bernoulli_loader(data, time=t, max_prob=m)

                        for i, spikes in enumerate(spike_loader):
                            assert spikes.size() == torch.Size((t, n))

    def test_bernoulli_loader_max_prob(self):
        # Regression test (PR #743): bernoulli_loader must honor the ``max_prob``
        # keyword argument. Previously it read ``kwargs.get("dt")``, which never
        # exists in kwargs (``dt`` is a named parameter), so max_prob was silently
        # ignored and the spike rate stayed at 1.0 regardless of the argument.
        torch.manual_seed(0)

        # All-ones input over many trials: the empirical spike rate should track
        # max_prob, not the dt default of 1.0.
        data = torch.ones(1, 20000)
        for m in [0.0, 0.25, 0.5, 0.75]:
            spikes = next(bernoulli_loader(data, time=1, max_prob=m))
            assert abs(spikes.float().mean().item() - m) < 0.02

        # dt must not leak into max_prob: with dt set but max_prob explicit,
        # the rate follows max_prob.
        spikes = next(bernoulli_loader(data, time=1, dt=0.5, max_prob=0.3))
        assert abs(spikes.float().mean().item() - 0.3) < 0.02

        # Back-compat: omitting max_prob defaults to 1.0.
        spikes = next(bernoulli_loader(data, time=1))
        assert spikes.float().mean().item() == 1.0

    def test_poisson(self):
        for n in [1, 100]:  # number of nodes in layer
            for t in [1000]:  # number of timesteps
                datum = torch.empty(n).uniform_(20, 100)  # Generate firing rates.
                spikes = poisson(datum, time=t)  # Encode as spikes.

                assert spikes.size() == torch.Size((t, n))

    def test_poisson_loader(self):
        for s in [1, 10]:  # number of data samples
            for n in [1, 100]:  # number of nodes in layer
                for t in [1000]:  # number of timesteps
                    data = torch.empty(s, n).uniform_(20, 100)  # Generate firing rates.
                    spike_loader = poisson_loader(data, time=t)  # Encode as spikes.

                    for i, spikes in enumerate(spike_loader):
                        assert spikes.size() == torch.Size((t, n))
