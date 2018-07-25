import torch

from bindsnet.network import Network
from bindsnet.network.monitors import Monitor, NetworkMonitor
from bindsnet.network.nodes import Input, IFNodes
from bindsnet.network.topology import Connection


class TestMonitor:
    """
    Testing Monitor object.
    """
    network = Network()

    inpt = Input(75)
    network.add_layer(inpt, name='X')
    _if = IFNodes(25)
    network.add_layer(_if, name='Y')
    conn = Connection(inpt, _if, w=torch.rand(inpt.n, _if.n))
    network.add_connection(conn, source='X', target='Y')

    inpt_mon = Monitor(inpt, state_vars=['s'])
    network.add_monitor(inpt_mon, name='X')
    _if_mon = Monitor(_if, state_vars=['s', 'v'])
    network.add_monitor(_if_mon, name='Y')

    network.run(inpts={'X': torch.bernoulli(torch.rand(100, inpt.n))}, time=100)

    assert inpt_mon.get('s').size() == torch.Size([inpt.n, 100])
    assert _if_mon.get('s').size() == torch.Size([_if.n, 100])
    assert _if_mon.get('v').size() == torch.Size([_if.n, 100])

    del network.monitors['X'], network.monitors['Y']

    inpt_mon = Monitor(inpt, state_vars=['s'], time=500)
    network.add_monitor(inpt_mon, name='X')
    _if_mon = Monitor(_if, state_vars=['s', 'v'], time=500)
    network.add_monitor(_if_mon, name='Y')

    network.run(inpts={'X': torch.bernoulli(torch.rand(500, inpt.n))}, time=500)

    assert inpt_mon.get('s').size() == torch.Size([inpt.n, 500])
    assert _if_mon.get('s').size() == torch.Size([_if.n, 500])
    assert _if_mon.get('v').size() == torch.Size([_if.n, 500])


class TestNetworkMonitor:
    """
    Testing NetworkMonitor object.
    """
    network = Network()

    inpt = Input(25)
    network.add_layer(inpt, name='X')
    _if = IFNodes(75)
    network.add_layer(_if, name='Y')
    conn = Connection(inpt, _if, w=torch.rand(inpt.n, _if.n))
    network.add_connection(conn, source='X', target='Y')

    mon = NetworkMonitor(network, state_vars=['s', 'v', 'w'])
    network.add_monitor(mon, name='monitor')

    network.run(inpts={'X': torch.bernoulli(torch.rand(50, inpt.n))}, time=50)

    recording = mon.get()

    assert recording['X']['s'].size() == torch.Size([inpt.n, 50])
    assert recording['Y']['s'].size() == torch.Size([_if.n, 50])
    assert recording['Y']['s'].size() == torch.Size([_if.n, 50])

    del network.monitors['monitor']

    mon = NetworkMonitor(network, state_vars=['s', 'v', 'w'], time=50)
    network.add_monitor(mon, name='monitor')

    network.run(inpts={'X': torch.bernoulli(torch.rand(50, inpt.n))}, time=50)

    recording = mon.get()

    assert recording['X']['s'].size() == torch.Size([inpt.n, 50])
    assert recording['Y']['s'].size() == torch.Size([_if.n, 50])
    assert recording['Y']['s'].size() == torch.Size([_if.n, 50])
