from bindsnet.network import Network
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, AdaptiveLIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight
from bindsnet.learning.MCC_learning import MSTDP


class Reservoir(Network):
  def __init__(self, in_size, exc_size, inh_size, hyper_params,
               w_in_exc, w_in_inh, w_exc_exc, w_exc_inh, w_inh_exc, w_inh_inh,
               device='cpu'):
    super().__init__()

    ## Layers ##
    input = Input(n=in_size)
    res_exc = AdaptiveLIFNodes(
      n=exc_size,
      thresh=hyper_params['thresh_exc'],
      theta_plus=hyper_params['theta_plus_exc'],
      refrac=hyper_params['refrac_exc'],
      reset=hyper_params['reset_exc'],
      tc_theta_decay=hyper_params['tc_theta_decay_exc'],
      tc_decay=hyper_params['tc_decay_exc'],
      traces=True,
    )
    exc_monitor = Monitor(res_exc, ["s"], device=device)
    self.add_monitor(exc_monitor, name='res_monitor_exc')
    self.exc_monitor = exc_monitor
    res_inh = AdaptiveLIFNodes(
      n=inh_size,
      thresh=hyper_params['thresh_inh'],
      theta_plus=hyper_params['theta_plus_inh'],
      refrac=hyper_params['refrac_inh'],
      reset=hyper_params['reset_inh'],
      tc_theta_decay=hyper_params['tc_theta_decay_inh'],
      tc_decay=hyper_params['tc_decay_inh'],
      traces=True,
    )
    inh_monitor = Monitor(res_inh, ["s"], device=device)
    self.add_monitor(inh_monitor, name='res_monitor_inh')
    self.inh_monitor = inh_monitor
    self.add_layer(input, name='input')
    self.add_layer(res_exc, name='res_exc')
    self.add_layer(res_inh, name='res_inh')

    ## Connections ##
    in_exc_wfeat = Weight(name='in_exc_weight_feature', value=w_in_exc,)
    in_exc_conn = MulticompartmentConnection(
      source=input, target=res_exc,
      device=device, pipeline=[in_exc_wfeat],
    )
    in_inh_wfeat = Weight(name='in_inh_weight_feature', value=w_in_inh,)
    in_inh_conn = MulticompartmentConnection(
      source=input, target=res_inh,
      device=device, pipeline=[in_inh_wfeat],
    )

    exc_exc_wfeat = Weight(name='exc_exc_weight_feature', value=w_exc_exc,)
                           # learning_rule=MSTDP,
                           # nu=hyper_params['nu_exc_exc'], range=hyper_params['range_exc_exc'], decay=hyper_params['decay_exc_exc'])
    exc_exc_conn = MulticompartmentConnection(
      source=res_exc, target=res_exc,
      device=device, pipeline=[exc_exc_wfeat],
    )
    exc_inh_wfeat = Weight(name='exc_inh_weight_feature', value=w_exc_inh,)
                           # learning_rule=MSTDP,
                           # nu=hyper_params['nu_exc_inh'], range=hyper_params['range_exc_inh'], decay=hyper_params['decay_exc_inh'])
    exc_inh_conn = MulticompartmentConnection(
      source=res_exc, target=res_inh,
      device=device, pipeline=[exc_inh_wfeat],
    )
    inh_exc_wfeat = Weight(name='inh_exc_weight_feature', value=w_inh_exc,)
                           # learning_rule=MSTDP,
                           # nu=hyper_params['nu_inh_exc'], range=hyper_params['range_inh_exc'], decay=hyper_params['decay_inh_exc'])
    inh_exc_conn = MulticompartmentConnection(
      source=res_inh, target=res_exc,
      device=device, pipeline=[inh_exc_wfeat],
    )
    inh_inh_wfeat = Weight(name='inh_inh_weight_feature', value=w_inh_inh,)
                           # learning_rule=MSTDP,
                           # nu=hyper_params['nu_inh_inh'], range=hyper_params['range_inh_inh'], decay=hyper_params['decay_inh_inh'])
    inh_inh_conn = MulticompartmentConnection(
      source=res_inh, target=res_inh,
      device=device, pipeline=[inh_inh_wfeat],
    )
    self.add_connection(in_exc_conn, source='input', target='res_exc')
    self.add_connection(in_inh_conn, source='input', target='res_inh')
    self.add_connection(exc_exc_conn, source='res_exc', target='res_exc')
    self.add_connection(exc_inh_conn, source='res_exc', target='res_inh')
    self.add_connection(inh_exc_conn, source='res_inh', target='res_exc')
    self.add_connection(inh_inh_conn, source='res_inh', target='res_inh')

    ## Migrate ##
    self.to(device)

  def store(self, spike_train, sim_time):
    self.learning = True
    self.run(inputs={'input': spike_train}, time=sim_time, reward=1)
    exc_spikes = self.exc_monitor.get('s')
    inh_spikes = self.inh_monitor.get('s')
    self.learning = False
    return exc_spikes, inh_spikes

  def recall(self, spike_train, sim_time):
    self.learning = False
    self.run(inputs={'input': spike_train}, time=sim_time,)
    exc_spikes = self.exc_monitor.get('s')
    inh_spikes = self.inh_monitor.get('s')
    return exc_spikes, inh_spikes
