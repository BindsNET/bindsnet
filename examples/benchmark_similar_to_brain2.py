import os
import sys
import torch
import numpy             as np
import matplotlib.pyplot as plt
from timeit import default_timer

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'datasets')))

from network           import Network
from encoding          import get_poisson
from connections       import Connection, post_pre
from nodes             import LIFNodes, Input
from analysis.plotting import plot_spikes, plot_weights

## param
gpu = True
n = 4000
n_e = int(n * 0.8)
n_i = n - n_e
dt = 1.0
run_time = 1000 # 1000 = 1s

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')

# Build network.
network = Network(dt=dt)

# Excitatory / Inhibitory layer.
exc = LIFNodes(n=n_e, rest=-49.0, reset=-60.0, threshold=-50.0, refractory=0, voltage_decay=1 / 20)
inh = LIFNodes(n=n_i, rest=-49.0, reset=-60.0, threshold=-50.0, refractory=0, voltage_decay=1 / 20)

exc.v = -60.0 * torch.ones(n_e)
inh.v = -60.0 * torch.ones(n_i)

# Connectivity between neurons.
# Excitatory -> inhibitory.
w = torch.zeros_like(torch.Tensor(n_e, n_i))
rand = torch.rand(n_e, n_i)

w[rand <= 0.02] = 1.62 * torch.rand(n_e, n_i)

exc_inh_conn = Connection(source=exc, target=inh, w=w, update_rule=None)

# Excitatory -> excitatory.
w = torch.zeros_like(torch.Tensor(n_e, n_e))
rand = torch.rand(n_e, n_e)
w[rand <= 0.02] = 1.62 * torch.rand(n_e, n_e)

exc_exc_conn = Connection(source=exc, target=exc, w=w, update_rule=None)

# Inhibitory -> excitatory.
w = torch.zeros_like(torch.Tensor(n_i, n_e))
rand = torch.rand(n_i, n_e)
w[rand <= 0.02] = -9 * torch.rand(n_i, n_e)

inh_exc_conn = Connection(source=inh, target=exc, w=w, update_rule=None)

# Inhibitory -> inhibitory.
w = torch.zeros_like(torch.Tensor(n_i, n_i))
rand = torch.rand(n_i, n_i)
w[rand <= 0.02] = -9 * torch.rand(n_i, n_i)

inh_inh_conn = Connection(source=inh, target=inh, w=w, update_rule=None)

# Add all layers and connections to the network.
network.add_layer(exc, name='Ae')
network.add_layer(inh, name='Ai')
network.add_connection(exc_inh_conn, source='Ae', target='Ai')
network.add_connection(inh_exc_conn, source='Ai', target='Ae')
network.add_connection(exc_exc_conn, source='Ae', target='Ae')
network.add_connection(inh_inh_conn, source='Ai', target='Ai')

# Run the network on the input for time `t`.
spikes = network.run(inpts={}, time=run_time)

plot_spikes(spikes)

import matplotlib.pyplot as plt
plt.show()
