from bindsnet.network.topology import Connection
from bindsnet.models import TwoLayerNetwork, DiehlAndCook2015
from bindsnet.network.nodes import Input, LIFNodes, DiehlAndCookNodes


class TestTwoLayerNetwork:

    def test_init(self):
        for n_inpt in [50, 100, 200]:
            for n_neurons in [50, 100, 200]:
                for dt in [1.0, 2.0]:
                    network = TwoLayerNetwork(n_inpt, n_neurons=n_neurons, dt=dt)

                    assert network.n_inpt == n_inpt
                    assert network.n_neurons == n_neurons
                    assert network.dt == dt

                    assert isinstance(network.layers['X'], Input) and network.layers['X'].n == n_inpt
                    assert isinstance(network.layers['Y'], LIFNodes) and network.layers['Y'].n == n_neurons
                    assert isinstance(network.connections[('X', 'Y')], Connection)
                    assert network.connections[('X', 'Y')].source.n == n_inpt and network.connections[
                        ('X', 'Y')].target.n == n_neurons


class TestDiehlAndCook2015:

    def test_init(self):
        for n_inpt in [50, 100, 200]:
            for n_neurons in [50, 100, 200]:
                for dt in [1.0, 2.0]:
                    for exc in [13.3, 14.53]:
                        for inh in [10.5, 12.2]:
                            network = DiehlAndCook2015(n_inpt=n_inpt, n_neurons=n_neurons, exc=exc,
                                                       inh=inh, dt=dt)

                            assert network.n_inpt == n_inpt
                            assert network.n_neurons == n_neurons
                            assert network.dt == dt
                            assert network.exc == exc
                            assert network.inh == inh

                            assert isinstance(network.layers['X'], Input) and network.layers['X'].n == n_inpt
                            assert isinstance(network.layers['Ae'], DiehlAndCookNodes) and network.layers[
                                'Ae'].n == n_neurons
                            assert isinstance(network.layers['Ai'], LIFNodes) and network.layers['Ae'].n == n_neurons

                            for conn in [('X', 'Ae'), ('Ae', 'Ai'), ('Ai', 'Ae')]:
                                assert conn in network.connections
