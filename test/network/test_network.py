import os
import pytest

from bindsnet.network import Network, load
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection


class TestNetwork:
    """
    Tests basic network functionality.
    """

    def test_empty(self, tmp_path):
        for dt in [0.1, 1.0, 5.0]:
            network = Network(dt=dt)
            assert network.dt == dt

            network.run(inputs={}, time=1000)

            file_path = str(tmp_path / "net.pt")
            network.save(file_path)
            _network = load(file_path)
            assert _network.dt == dt
            assert _network.learning
            del _network

            _network = load(file_path, learning=True)
            assert _network.dt == dt
            assert _network.learning
            del _network

            _network = load(file_path, learning=False)
            assert _network.dt == dt
            assert not _network.learning
            del _network

    def test_add_objects(self, tmp_path):
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

        file_path = str(tmp_path / "net.pt")
        network.save(file_path)
        _network = load(file_path, learning=True)
        assert _network.learning
        assert "X" in _network.layers
        assert "Y" in _network.layers
        assert ("X", "Y") in _network.connections
        assert "Y" in _network.monitors
        del _network

    def test_clone(self):
        """
        ``clone()`` round-trips a network in memory.

        Regression test: ``clone()`` called ``torch.load`` without
        ``weights_only=False``, so it broke outright when PyTorch 2.6 changed
        that default to ``True``. It had no test and no caller, so the breakage
        went unnoticed.
        """
        import torch

        network = Network(dt=1.0, learning=False)
        inpt = Input(10)
        network.add_layer(inpt, name="X")
        lif = LIFNodes(5)
        network.add_layer(lif, name="Y")
        w = torch.rand(10, 5)
        network.add_connection(
            Connection(inpt, lif, w=w.clone()), source="X", target="Y"
        )

        clone = network.clone()

        assert isinstance(clone, Network)
        assert clone is not network
        assert clone.dt == network.dt
        assert clone.learning == network.learning
        assert "X" in clone.layers and "Y" in clone.layers
        assert ("X", "Y") in clone.connections
        assert torch.equal(clone.connections[("X", "Y")].w, w)

    def test_clone_after_save(self, tmp_path):
        """
        ``clone()`` still works after ``save()`` has run in the same process.

        Regression test: ``save()`` called
        ``torch.serialization.add_safe_globals([self])`` with an instance
        instead of a class, which corrupted PyTorch's safe-globals registry and
        made every later load in the process fail with
        ``'Network' object has no attribute '__qualname__'``.
        """
        network = Network(dt=1.0)
        network.add_layer(Input(4), name="X")

        network.save(str(tmp_path / "net.pt"))

        clone = network.clone()
        assert isinstance(clone, Network)
        assert clone.dt == 1.0

    def test_load_weights_only_parameter(self, tmp_path):
        """
        ``load()`` exposes ``weights_only`` and still defaults to ``False``.

        The default has to stay ``False`` because ``save()`` writes the whole
        network object, which the safe loader cannot read. ``weights_only=True``
        is offered for files holding plain tensors only, so on a real network
        file it is expected to fail rather than silently return something wrong.
        """
        import inspect

        signature = inspect.signature(load)
        assert "weights_only" in signature.parameters
        assert signature.parameters["weights_only"].default is False

        file_path = str(tmp_path / "net.pt")
        network = Network(dt=1.0)
        network.add_layer(Input(4), name="X")
        network.save(file_path)

        # Default path loads a saved network.
        assert isinstance(load(file_path), Network)

        # Safe path refuses this file rather than mis-loading it.
        with pytest.raises(Exception):
            load(file_path, weights_only=True)
