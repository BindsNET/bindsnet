import pickle as pkl
import torch
import numpy as np
import matplotlib.pyplot as plt
from Grid_Cells import activity_to_spike

from bindsnet.learning.MCC_learning import PostPre, MSTDP
from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight


class Memory_SNN(Network):
  def __init__(self,
               key_size, val_size, in_size,
               w_in_key, w_in_val, w_assoc,
               hyper_params, device='cpu'):
    super().__init__()

    ## Layers ##
    key_input = Input(n=in_size)
    val_input = Input(n=in_size)
    key = AdaptiveLIFNodes(
      n=key_size,
      thresh=hyper_params['thresh'],
      theta_plus=hyper_params['theta_plus'],
      refrac=hyper_params['refrac'],
      reset=hyper_params['reset'],
      tc_theta_decay=hyper_params['tc_theta_decay'],
      tc_decay=hyper_params['tc_decay'],
      traces=True,
    )
    value = AdaptiveLIFNodes(
      n=val_size,
      thresh=hyper_params['thresh'],
      theta_plus=hyper_params['theta_plus'],
      refrac=hyper_params['refrac'],
      reset=hyper_params['reset'],
      tc_theta_decay=hyper_params['tc_theta_decay'],
      tc_decay=hyper_params['tc_decay'],
      traces = True,
    )
    val_monitor = Monitor(value, ["s"], device=device)
    self.add_monitor(val_monitor, name='val_monitor')
    self.val_monitor = val_monitor
    self.add_layer(key_input, name='key_input')
    self.add_layer(val_input, name='val_input')
    self.add_layer(key, name='key')
    self.add_layer(value, name='value')

    ## Connections ##
    # Key
    in_key_wfeat = Weight(name='in_key_weight_feature', value=w_in_key)
    in_key_conn = MulticompartmentConnection(
      source=key_input, target=key,
      device=device, pipeline=[in_key_wfeat],
    )
    # Value
    in_val_wfeat = Weight(name='in_val_weight_feature', value=w_in_val)
    in_val_conn = MulticompartmentConnection(
      source=val_input, target=value,
      device=device, pipeline=[in_val_wfeat],
    )
    # Association
    assoc_wfeat = Weight(name='assoc_weight_feature', value=w_assoc,
                         learning_rule=MSTDP, nu=hyper_params['nu'], range=[0, 1], decay=hyper_params['decay'])
    assoc_conn = MulticompartmentConnection(
      source=key, target=value,
      device=device, pipeline=[assoc_wfeat], traces=True,
    )
    assoc_monitor = Monitor(assoc_wfeat, ["value"], device=device)
    self.add_connection(in_key_conn, source='key_input', target='key')
    self.add_connection(in_val_conn, source='val_input', target='value')
    self.add_connection(assoc_conn, source='key', target='value')
    self.add_monitor(assoc_monitor, name='assoc_monitor')
    self.assoc_monitor = assoc_monitor

    ## Migrate device ##
    self.to(device)

  # Store memory
  # input: torch.Tensor of shape (time, in_size)
  # output: Association output (time, key_size, val_size), Value output (time, val_size)
  def store(self, key_train, sim_time=100, lr_params={}):
    self.learning = True
    self.run(inputs={'key_input':key_train, 'val_input':key_train}, time=sim_time, reward=1, **lr_params)
    assoc_out = self.assoc_monitor.get('value')
    val_spikes = self.val_monitor.get('s')
    return assoc_out, val_spikes

  # Recall memory given a key
  # input: torch.Tensor of shape (in_size) (key)
  # output: torch.Tensor of shape (val_size) (value)
  def recall(self, key_train, sim_time=100):
    self.learning = False
    self.run(inputs={'val_input':key_train}, time=sim_time)
    val_spikes = self.val_monitor.get('s')
    return val_spikes


def assign_inhibition(weights, percent, inhib_scale):
  layer_shape = weights.shape
  layer_size = np.prod(layer_shape)
  indices_to_flip = np.random.choice(layer_size, int(layer_size * percent), replace=False)
  indices_to_flip = np.unravel_index(indices_to_flip, layer_shape)
  weights[indices_to_flip] = -weights[indices_to_flip]*inhib_scale
  return weights

# Note: percent = number of weights to zero out
def sparsify(weights, percent):
  layer_shape = weights.shape
  layer_size = np.prod(layer_shape)
  indices_to_zero = np.random.choice(layer_size, int(layer_size * percent), replace=False)
  indices_to_zero = np.unravel_index(indices_to_zero, layer_shape)
  weights[indices_to_zero] = 0
  return weights
