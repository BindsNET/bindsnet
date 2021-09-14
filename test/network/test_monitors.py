import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor, NetworkMonitor
from bindsnet.network.nodes import IFNodes, Input
from bindsnet.network.topology import Connection


class TestMonitor:
    """
    Testing Monitor object.
    """

    network = Network()

    inpt = Input(75)
    network.add_layer(inpt, name="X")
    _if = IFNodes(25)
    network.add_layer(_if, name="Y")
    conn = Connection(inpt, _if, w=torch.rand(inpt.n, _if.n))
    network.add_connection(conn, source="X", target="Y")

    inpt_mon = Monitor(inpt, state_vars=["s"])
    network.add_monitor(inpt_mon, name="X")
    _if_mon = Monitor(_if, state_vars=["s", "v"])
    network.add_monitor(_if_mon, name="Y")

    network.run(inputs={"X": torch.bernoulli(torch.rand(100, inpt.n))}, time=100)

    assert inpt_mon.get("s").size() == torch.Size([100, 1, inpt.n])
    assert _if_mon.get("s").size() == torch.Size([100, 1, _if.n])
    assert _if_mon.get("v").size() == torch.Size([100, 1, _if.n])

    del network.monitors["X"], network.monitors["Y"]

    inpt_mon = Monitor(inpt, state_vars=["s"], time=500)
    network.add_monitor(inpt_mon, name="X")
    _if_mon = Monitor(_if, state_vars=["s", "v"], time=500)
    network.add_monitor(_if_mon, name="Y")

    network.run(inputs={"X": torch.bernoulli(torch.rand(500, inpt.n))}, time=500)

    assert inpt_mon.get("s").size() == torch.Size([500, 1, inpt.n])
    assert _if_mon.get("s").size() == torch.Size([500, 1, _if.n])
    assert _if_mon.get("v").size() == torch.Size([500, 1, _if.n])


class TestNetworkMonitor:
    """
    Testing NetworkMonitor object.
    """

    network = Network()

    inpt = Input(25)
    network.add_layer(inpt, name="X")
    _if = IFNodes(75)
    network.add_layer(_if, name="Y")
    conn = Connection(inpt, _if, w=torch.rand(inpt.n, _if.n))
    network.add_connection(conn, source="X", target="Y")

    mon = NetworkMonitor(network, state_vars=["s", "v", "w"])
    network.add_monitor(mon, name="monitor")

    network.run(inputs={"X": torch.bernoulli(torch.rand(50, inpt.n))}, time=50)

    recording = mon.get()

    assert recording["X"]["s"].size() == torch.Size([50, 1, inpt.n])
    assert recording["Y"]["s"].size() == torch.Size([50, 1, _if.n])
    assert recording["Y"]["s"].size() == torch.Size([50, 1, _if.n])

    del network.monitors["monitor"]

    mon = NetworkMonitor(network, state_vars=["s", "v", "w"], time=50)
    network.add_monitor(mon, name="monitor")

    network.run(inputs={"X": torch.bernoulli(torch.rand(50, inpt.n))}, time=50)

    recording = mon.get()

    assert recording["X"]["s"].size() == torch.Size([50, 1, inpt.n])
    assert recording["Y"]["s"].size() == torch.Size([50, 1, _if.n])
    assert recording["Y"]["s"].size() == torch.Size([50, 1, _if.n])


if __name__ == "__main__":
    tm = TestMonitor()
    tnm = TestNetworkMonitor()
