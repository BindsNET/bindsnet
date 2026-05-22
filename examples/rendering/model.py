from pandas.core.internals.construction import nested_data_to_arrays

from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight, Mask
from bindsnet.network.network import GUINetwork
import torch

def create_model(
  in_size = 100,
  exc_size=10_000,
  inh_size=1_500,
  i_to_exc_connectivity=0.05,
  i_to_inh_connectivity=0.05,
  inh_to_exc_connectivity=0.05,
  exc_to_inh_connectivity=0.05,
) -> GUINetwork:

  device = torch.device('cuda:0')
  network = GUINetwork()
  network.add_layer(layer=Input(in_size), name='I')
  network.add_layer(layer=LIFNodes(exc_size), name='EXC_LIF')
  network.add_layer(layer=LIFNodes(inh_size), name='INH_LIF')
  network.add_connection(
    connection=MulticompartmentConnection(
      source=network.layers['I'],
      target=network.layers['EXC_LIF'],
      device=device,
      pipeline=[
        Weight(
          name='I_to_EXC_weight',
          value=torch.rand(in_size, exc_size, device=device),
        ),
        Mask(
          name='I_to_EXC_mask',
          value=torch.rand(in_size, exc_size, device=device) > (1-i_to_exc_connectivity),
        )
      ]),
    source='I',
    target='EXC_LIF')
  network.add_connection(
    connection=MulticompartmentConnection(
      source=network.layers['I'],
      target=network.layers['INH_LIF'],
      device=device,
      pipeline=[
        Weight(
          name='I_to_INH_weight',
          value=torch.rand(in_size, inh_size, device=device),
        ),
        Mask(
          name='I_to_INH_mask',
          value=torch.rand(in_size, inh_size, device=device) > (1-i_to_inh_connectivity),
        )
      ]),
    source='I',
    target='INH_LIF')
  network.add_connection(
    connection=MulticompartmentConnection(
      source=network.layers['INH_LIF'],
      target=network.layers['EXC_LIF'],
      device=device,
      pipeline=[
        Weight(
          name='INH_to_EXC_weight',
          value=-torch.rand(inh_size, exc_size, device=device),
        ),
        Mask(
          name='INH_to_EXC_mask',
          value=torch.rand(inh_size, exc_size, device=device) > (1-inh_to_exc_connectivity),
        )
      ]),
    source='INH_LIF',
    target='EXC_LIF')
  network.add_connection(
    connection=MulticompartmentConnection(
      source=network.layers['EXC_LIF'],
      target=network.layers['INH_LIF'],
      device=device,
      pipeline=[
        Weight(
          name='EXC_to_INH_weight',
          value=torch.rand(exc_size, inh_size, device=device),
        ),
        Mask(
          name='EXC_to_INH_mask',
          value=torch.rand(exc_size, inh_size, device=device) > (1-exc_to_inh_connectivity),
        )
      ]),
    source='EXC_LIF',
    target='INH_LIF')
  network.to(device)
  return network
