import os

from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network import Network, load_network


class TestNetwork:
    """
    Tests basic network functionality.
    """

    def test_empty(self):
        for dt in [0.1, 1.0, 5.0]:
            network = Network(dt=dt)
            assert network.dt == dt

            network.run(inpts={}, time=1000)

            network.save('net.p')
            _network = load_network('net.p')
            assert _network.dt == dt

            os.remove('net.p')

    def test_add_objects(self):
        network = Network(dt=1.0)

        inpt = Input(100)
        network.add_layer(inpt, name='X')
        lif = LIFNodes(50)
        network.add_layer(lif, name='Y')

        assert inpt == network.layers['X']
        assert lif == network.layers['Y']

        conn = Connection(inpt, lif)
        network.add_connection(conn, source='X', target='Y')

        assert conn == network.connections[('X', 'Y')]

        monitor = Monitor(lif, state_vars=['s', 'v'])
        network.add_monitor(monitor, 'Y')

        assert monitor == network.monitors['Y']
