from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDP


class Reservoir(Network):
  def __init__(self, in_size, res_size, hyper_params,
               w_in_res, w_res_res, device='cpu'):
    super().__init__()

    ## Layers ##
    input = Input(n=in_size)
    res = AdaptiveLIFNodes(
      n=res_size,
      thresh=hyper_params['thresh'],
      theta_plus=hyper_params['theta_plus'],
      refrac=hyper_params['refrac'],
      reset=hyper_params['reset'],
      tc_theta_decay=hyper_params['tc_theta_decay'],
      tc_decay=hyper_params['tc_decay'],
      traces=True,
    )
    res_monitor = Monitor(res, ["s"], device=device)
    self.add_monitor(res_monitor, name='res_monitor')
    self.res_monitor = res_monitor
    self.add_layer(input, name='input')
    self.add_layer(res, name='res')

    ## Connections ##
    in_res_wfeat = Weight(name='in_res_weight_feature', value=w_in_res,)
    in_res_conn = MulticompartmentConnection(
      source=input, target=res,
      device=device, pipeline=[in_res_wfeat],
    )
    res_res_wfeat = Weight(name='res_res_weight_feature', value=w_res_res,
                           # learning_rule=MSTDP,
                           nu=hyper_params['nu'], range=hyper_params['range'], decay=hyper_params['decay'])
    res_res_conn = MulticompartmentConnection(
      source=res, target=res,
      device=device, pipeline=[res_res_wfeat],
    )
    self.add_connection(in_res_conn, source='input', target='res')
    self.add_connection(res_res_conn, source='res', target='res')
    self.res_res_conn = res_res_conn

    ## Migrate ##
    self.to(device)

  def store(self, spike_train, sim_time):
    self.learning = True
    self.run(inputs={'input': spike_train}, time=sim_time, reward=1)
    res_spikes = self.res_monitor.get('s')
    self.learning = False
    return res_spikes

  def recall(self, spike_train, sim_time):
    self.learning = False
    self.run(inputs={'input': spike_train}, time=sim_time,)
    res_spikes = self.res_monitor.get('s')
    return res_spikes
