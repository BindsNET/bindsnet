import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from timeit                  import default_timer
from mpl_toolkits.axes_grid1 import make_axes_locatable

from bindsnet.evaluation          import *
from bindsnet.analysis.plotting   import *
from bindsnet.network             import Network
from bindsnet.encoding            import get_bernoulli
from bindsnet.environment         import SpaceInvaders

from bindsnet.network.monitors    import Monitor
from bindsnet.network.nodes       import LIFNodes, Input
from bindsnet.network.connections import Connection, hebbian

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--plot_interval', type=int, default=500)
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
inpt = Input(n=3780, traces=True)  # Input layer
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

i = 0
done = False
s, inpts = {}, {}

print()

count = 0
lengths = []
rewards = []
mean_rewards = []

while True:
	count += 1
	
	if i % plot_interval == 0:
		spike_record = {layer : spikes[layer].get('s') for layer in spikes}
		voltage_record = {layer : voltages[layer].get('v') for layer in voltages}
		
		for m in network.monitors:
			network.monitors[m]._reset()
	
	if plot or env_plot:
		env.render()
	
	if i % 100 == 0:
		print('Iteration %d' % i)

	# Choose action based on readout neuron spiking.
	if s == {} or s['R'].sum() == 0:
		action = np.random.choice(range(6))
	else:
		action = torch.multinomial((s['R'] / s['R'].sum()).view(-1), 1)[0] + 1
	
	# Get observations, reward, done flag, and other information.
	obs, reward, done, info = env.step(action)
	
	# Update mean reward.
	rewards.append(reward)
	
	# Get next input sample.
	inpts.update({'X' : obs})
	
	# Run the network on the input.
	network.run(inpts=inpts, time=1)
	
	# Normalize adaptable weights.
	network.connections[('E', 'R')].normalize(exc_readout_norm)
	
	# Get spikes from previous iteration.
	for layer in layers:
		s[layer] = spikes[layer].get('s')[:, i % plot_interval]
	
	# Optionally plot various simulation info.
	if plot:
		if i == 0:
			spike_ims, spike_axes = plot_spikes(spike_record)
			voltage_ims, voltage_axes = plot_voltages(voltage_record)
			
			w_fig, w_axes = plt.subplots(2, 1)
			
			i_e_w_im = w_axes[0].matshow(input_exc_conn.w.t(), cmap='hot_r', vmax=input_exc_conn.wmax)
			e_r_w_im = w_axes[1].matshow(exc_readout_conn.w.t(), cmap='hot_r', vmax=exc_readout_conn.wmax)
			
			d1 = make_axes_locatable(w_axes[0])
			d2 = make_axes_locatable(w_axes[1])
			c1 = d1.append_axes("right", size="5%", pad=0.05)
			c2 = d2.append_axes("right", size="5%", pad=0.05)

			plt.colorbar(i_e_w_im, cax=c1)
			plt.colorbar(e_r_w_im, cax=c2)
			
			for ax in w_axes:
				ax.set_xticks(()); ax.set_yticks(())
				ax.set_aspect('auto')
			
			plt.pause(1e-8)
			
			r_fig, r_axes = plt.subplots(2, 1, sharex=True)
			r_line, = r_axes[0].plot(mean_rewards, marker='o')
			l_line, = r_axes[1].plot(lengths, marker='o')
			
			r_axes[1].set_xlabel('Episode')
			r_axes[0].set_ylabel('Mean reward')
			r_axes[1].set_ylabel('Length of episode')
			r_axes[0].set_title('Mean reward per episode')
			r_axes[1].set_title('Iteration length of episodes')
			
			for ax in r_axes:
				ax.grid()
			
		else:
			if i % plot_interval == 0:
				spike_ims, spike_axes = plot_spikes(spike_record, ims=spike_ims, axes=spike_axes)
				voltage_ims, voltage_axes = plot_voltages(voltage_record, ims=voltage_ims, axes=voltage_axes)
				
				i_e_w_im.set_data(input_exc_conn.w.t())
				e_r_w_im.set_data(exc_readout_conn.w.t())
				
				for ax in w_axes:
					ax.set_aspect('auto')
					
				r_line.set_xdata(range(len(mean_rewards)))
				r_line.set_ydata(mean_rewards)
				
				l_line.set_xdata(range(len(lengths)))
				l_line.set_ydata(lengths)
				
				for ax in r_axes:
					ax.relim() 
					ax.autoscale_view(True, True, True) 

				r_fig.canvas.draw()
				
				plt.pause(1e-8)
	
	i += 1
	
	if done == True:
		env.reset()
		
		mean_rewards.append(np.mean(rewards))
		rewards = []
		
		lengths.append(count)
		count = 0

print()
	
env.render()
