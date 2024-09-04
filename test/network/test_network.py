import os

from bindsnet.network import Network, load
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection


class TestNetwork:
    """
    Tests basic network functionality.
    """

    def test_empty(self):
        for dt in [0.1, 1.0, 5.0]:
            network = Network(dt=dt)
            assert network.dt == dt

            network.run(inputs={}, time=1000)

            network.save("net.pt")
            _network = load("net.pt")
            assert _network.dt == dt
            assert _network.learning
            del _network

            _network = load("net.pt", learning=True)
            assert _network.dt == dt
            assert _network.learning
            del _network

            _network = load("net.pt", learning=False)
            assert _network.dt == dt
            assert not _network.learning
            del _network

            os.remove("net.pt")

    def test_add_objects(self):
        network = Network(dt=1.0, learning=False)

        inpt = Input(100)
        network.add_layer(inpt, name="X")
        lif = LIFNodes(50)
        network.add_layer(lif, name="Y")

        assert inpt == network.layers["X"]
        assert lif == network.layers["Y"]

        conn = Connection(inpt, lif)
        network.add_connection(conn, source="X", target="Y")

        assert conn == network.connections[("X", "Y")]

        monitor = Monitor(lif, state_vars=["s", "v"])
        network.add_monitor(monitor, "Y")

        assert monitor == network.monitors["Y"]

        network.save("net.pt")
        _network = load("net.pt", learning=True)
        assert _network.learning
        assert "X" in _network.layers
        assert "Y" in _network.layers
        assert ("X", "Y") in _network.connections
        assert "Y" in _network.monitors
        del _network

        os.remove("net.pt")
