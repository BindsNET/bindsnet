import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from bindsnet                import *
from timeit                  import default_timer
from mpl_toolkits.axes_grid1 import make_axes_locatable

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--plot_interval', type=int, default=100)
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

def network_1():
	# Build network.
	network = Network(dt=dt)
	
	# Layers of neurons.
	inpt = Input(n=6552, traces=True)  # Input layer
	exc = LIFNodes(n=n_neurons, refrac=0, traces=True)  # Excitatory layer
	readout = LIFNodes(n=6, refrac=0, traces=True)  # Readout layer
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

	return network, exc_readout_norm

def network_2():
	# Build network.
	network = Network(dt=dt)
	
	# Layers of neurons.
	inpt = Input(n=6552, traces=True)  # Input layer
	exc = LIFNodes(n=n_neurons, refrac=0, traces=True)  # Excitatory layer
	inh = LIFNodes(n=n_neurons, traces=True, rest=-60.0, reset=-45.0, thresh=-40.0,
                                 decay=1e-1, refrac=2, trace_tc=1 / 20)
	readout = LIFNodes(n=6, refrac=0, traces=True)  # Readout layer
	layers = {'X' : inpt, 'E' : exc, 'I': inh, 'R' : readout}
	
	# Connections between layers.
	# Input -> excitatory.
	w = 0.01 * torch.rand(layers['X'].n, layers['E'].n)
	input_exc_conn = Connection(source=layers['X'], target=layers['E'], w=w, wmax=0.02)
	input_exc_norm = 0.01 * layers['X'].n
	
	# Excitatory -> inhibitory.
	w = 10 * torch.randn(layers['E'].n, layers['I'].n) # Mu = 0,  Var = 10 
	exc_inh_conn = Connection(source=layers['E'], target=layers['I'], w=w, wmax=0.05, update_rule=hebbian) # learning rule?
	exc_inh_norm = 0.01 * layers['E'].n
	
	# Inhibitory -> excitatory.
	inh_exc_w = np.concatenate( [np.ones(int(0.7*exc.n)), np.zeros(int(0.3*exc.n))] )
	inh_exc_w = np.random.permutation(inh_exc_w)
	inh_exc_w = -17.5 * (inh_exc_w * (torch.ones(inh_exc_w.shape) - torch.diag(torch.ones(inh.n))) ).float() 
	inh_exc_conn = Connection(source=layers['I'], target=layers['E'], w=inh_exc_w, update_rule=None)
	
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
	network.add_connection(exc_inh_conn, source='E', target='I')
	network.add_connection(inh_exc_conn, source='I', target='E')
	
	# Add all monitors to the network.
	for layer in layers:
		network.add_monitor(spikes[layer], name='%s_spikes' % layer)
		
		if layer in voltages:
			network.add_monitor(voltages[layer], name='%s_voltages' % layer)
	
	# Normalize adaptable weights.
	network.connections[('E', 'R')].normalize(exc_readout_norm)
	
	return network, exc_readout_norm

network, exc_readout_norm = network_2()

# Load SpaceInvaders environment.
env = SpaceInvaders(max_prob=1)
env.reset()

p = Pipeline(network,
			 env,
			 encoding=bernoulli,
			 plot=False,
			 time=1,
			 history=5,
			 delta=3,
			 render=False,
			 plot_interval=plot_interval,
			 feedback=select_softmax,
			 output='R')

print()

try:
	while True:
		p.step()
		p.normalize('E', 'R', exc_readout_norm)

		if p.done == True:
			env.reset()
except KeyboardInterrupt:
	plt.close('all')
	env.close()
