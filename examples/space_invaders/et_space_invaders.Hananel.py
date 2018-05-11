import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from bindsnet                import *
from time                    import sleep
from timeit                  import default_timer
from mpl_toolkits.axes_grid1 import make_axes_locatable


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--plot_interval', type=int, default=100)
parser.add_argument('--print_interval', type=int, default=1000)
parser.add_argument('--a_plus', type=int, default=1)
parser.add_argument('--a_minus', type=int, default=-0.5)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--render', dest='render', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, render=False, gpu=False)

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
exc = LIFNodes(n=n_neurons, refrac=0, traces=True, thresh=-52.0 + torch.randn(n_neurons))  # Excitatory layer
readout = LIFNodes(n=60, refrac=0, traces=True, thresh=-40.0)  # Readout layer
layers = {'X' : inpt, 'E' : exc, 'R' : readout}

# Connections between layers.
# Input -> excitatory.
w = 1e-3 * torch.rand(layers['X'].n, layers['E'].n)
input_exc_conn = Connection(source=layers['X'], target=layers['E'], w=w, wmin=-1e-2, wmax=1e-2)

# Excitatory -> readout.
w = 0.01 * torch.rand(layers['E'].n, layers['R'].n)
exc_readout_conn = Connection(source=layers['E'], target=layers['R'], w=w, wmin=-0.5, wmax=0.5, update_rule=m_stdp_et, nu=2e-2)
exc_readout_norm = 0.15 * layers['E'].n

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

# Add all monitors to the network.
for layer in layers:
	network.add_monitor(spikes[layer], name='%s_spikes' % layer)
	
	if layer in voltages:
		network.add_monitor(voltages[layer], name='%s_voltages' % layer)

# Normalize adaptable weights.
network.connections[('E', 'R')].normalize(exc_readout_norm)

# Load SpaceInvaders environment.
environment = GymEnvironment('SpaceInvaders-v0')
environment.reset()

pipeline = Pipeline(network,
			 environment,
			 encoding=bernoulli,
			 plot=plot,
			 time=1,
			 render=render,
			 history=5,
			 delta=10,
			 plot_interval=plot_interval,
			 print_interval=print_interval,
			 feedback=select_multinomial,
			 output='R')

try:
	iteration = 0
	while True:
		pipeline.step()
		pipeline.normalize('E', 'R', exc_readout_norm)
		iteration = pipeline.iteration
		if pipeline.done == True:
			print(iteration)
			pipeline._reset()
except KeyboardInterrupt:
	environment.close()
