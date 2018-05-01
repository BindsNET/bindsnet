import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from timeit                  import default_timer
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bindsnet.pipeline            import *
from bindsnet.evaluation          import *
from bindsnet.analysis.plotting   import *
from bindsnet.datasets.preprocess import *


from bindsnet.network             import Network
from bindsnet.encoding            import get_bernoulli
from bindsnet.environment         import SpaceInvaders

from bindsnet.learning            import hebbian
from bindsnet.network.monitors    import Monitor
from bindsnet.network.connections import Connection 
from bindsnet.network.nodes       import LIFNodes, Input


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--plot_interval', type=int, default=200)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--env_plot', dest='env_plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, env_plot=False, gpu=False)

locals().update(vars(parser.parse_args()))

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	torch.cuda.manual_seed_all(seed)
else:
	torch.manual_seed(seed)

# Build network.
network = Network(dt=dt)

# Layers of neurons.
inpt = Input(n=6552, traces=True)  # Input layer
exc = LIFNodes(n=n_neurons, refractory=0, traces=True)  # Excitatory layer
readout = LIFNodes(n=5, refractory=0, traces=True)  # Readout layer
layers = {'X' : inpt, 'E' : exc, 'R' : readout}

# Connections between layers.
# Input -> excitatory.
w = 0.01 * torch.rand(layers['X'].n, layers['E'].n)
input_exc_conn = Connection(source=layers['X'], target=layers['E'], w=w, wmax=0.02)
input_exc_norm = 0.01 * layers['X'].n

# Excitatory -> readout.
w = 0.01 * torch.rand(layers['E'].n, layers['R'].n)
exc_readout_conn = Connection(source=layers['E'], target=layers['R'], w=w,
							  update_rule=hebbian, nu_pre=1e-2, nu_post=1e-2)
exc_readout_norm = 0.5 * layers['E'].n

# Readout -> readout.
w = -10 * torch.ones(layers['R'].n, layers['R'].n) + 10 * torch.diag(torch.ones(layers['R'].n))
readout_readout_conn = Connection(source=layers['R'], target=layers['R'], w=w, wmin=-10.0)

# Spike recordings for all layers.
spikes = {}
for layer in layers:
	spikes[layer] = Monitor(layers[layer], ['s'], time=plot_interval)

# Voltage recordings for excitatory and readout layers.
voltages = {}
for layer in set(layers.keys()) - {'X'}:
	voltages[layer] = Monitor(layers[layer], ['v'], time=plot_interval)

# Add all layers and connections to the network.
for layer in layers:
	network.add_layer(layers[layer], name=layer)

network.add_connection(input_exc_conn, source='X', target='E')
network.add_connection(exc_readout_conn, source='E', target='R')
network.add_connection(readout_readout_conn, source='R', target='R')

# Add all monitors to the network.
for layer in layers:
	network.add_monitor(spikes[layer], name='%s_spikes' % layer)
	
	if layer in voltages:
		network.add_monitor(voltages[layer], name='%s_voltages' % layer)

# Normalize adaptable weights.
network.connections[('E', 'R')].normalize(exc_readout_norm)
	
# Load SpaceInvaders environment.
env = SpaceInvaders()
env.reset()

p = Pipeline(network, env, encoding=get_bernoulli, plot=False, time=1, render=True, history=0)
print()

while True:
	p.step()
	p.normalize('E', 'R', exc_readout_norm)
	
	if p.done == True:
		env.reset()
