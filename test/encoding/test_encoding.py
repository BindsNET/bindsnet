import os
import torch

from bindsnet.encoding import *


class TestEncodings:
    """
    Tests all stable encoding functions and generators.
    """

    def test_bernoulli(self):
        for n in [1, 10, 100, 1000]:  # number of nodes in layer
            for t in [1, 10, 100, 1000]:  # number of timesteps
                for m in [0.01, 0.1, 1.0]:  # maximum spiking probability
                    datum = torch.empty(n).uniform_(0, m)
                    spikes = bernoulli(datum, time=t, max_prob=m)

                    assert spikes.size() == torch.Size((t, n))

    def test_multidim_bernoulli(self):
        for shape in [[5, 5], [10, 10], [25, 25]]:  # shape of nodes in layer
            for t in [1, 10, 100]:  # number of timesteps
                for m in [0.01, 0.1, 1.0]:  # maximum spiking probability
                    datum = torch.empty(shape).uniform_(0, m)
                    spikes = bernoulli(datum, time=t, max_prob=m)

                    assert spikes.size() == torch.Size((t, *shape))

    def test_bernoulli_loader(self):
        for s in [1, 10, 100]:  # number of data samples
            for n in [1, 10, 100]:  # number of nodes in layer
                for m in [0.01, 0.1, 1.0]:  # maximum spiking probability
                    for t in [1, 10, 100]:  # number of timesteps
                        data = torch.empty(s, n).uniform_(0, 1)
                        spike_loader = bernoulli_loader(data, time=t, max_prob=m)

                        for i, spikes in enumerate(spike_loader):
                            assert spikes.size() == torch.Size((t, n))

    def test_poisson(self):
        for n in [1, 10, 100, 1000]:  # number of nodes in layer
            for t in [1000]:  # number of timesteps
                datum = torch.empty(n).uniform_(20, 100)  # Generate firing rates.
                spikes = poisson(datum, time=t)  # Encode as spikes.

                assert spikes.size() == torch.Size((t, n))

    def test_poisson_loader(self):
        for s in [1, 10]:  # number of data samples
            for n in [1, 10, 100]:  # number of nodes in layer
                for t in [1000]:  # number of timesteps
                    data = torch.empty(s, n).uniform_(20, 100)  # Generate firing rates.
                    spike_loader = poisson_loader(data, time=t)  # Encode as spikes.

                    for i, spikes in enumerate(spike_loader):
                        assert spikes.size() == torch.Size((t, n))

    def test_numenta_encoder(self):
        csvfile = './test/encoding/test.csv'
        encodingfile = './test/encoding/encoding.p'

        def test1():
            enc = NumentaEncoder(csvfile, save=True, encodingfile=encodingfile)
            val = enc.get_encoding()
            assert os.path.exists(encodingfile)
            os.remove(encodingfile)
            assert val is not None

        def test2():
            enc = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile)
            val = enc.get_encoding()
            assert val is not None
            assert not os.path.exists(encodingfile)

        def test3():
            enc1 = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile)
            enc2 = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile)
            v1, v2 = enc1.get_encoding(), enc2.get_encoding()
            assert torch.all(torch.eq(v1, v2))
            assert len(v1) == 30

        def test4():
            enc = NumentaEncoder(csvfile, save=False, encodingfile=encodingfile, n=5000, w=35)
            val = enc.get_encoding()
            assert val.shape[1] == 5000
            assert sum(sum(val)) == 35 * 30

        test1(), test2(), test3(), test4()
