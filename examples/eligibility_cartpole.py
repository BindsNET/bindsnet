import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from timeit                       import default_timer
from mpl_toolkits.axes_grid1      import make_axes_locatable

from bindsnet.evaluation          import *
from bindsnet.analysis.plotting   import *
from bindsnet.environment         import CartPole
from bindsnet.encoding            import get_bernoulli
from bindsnet.network.nodes       import LIFNodes, Input
from bindsnet.network             import Network, Monitor
from bindsnet.network.connections import Connection, m_stdp_et


parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--plot_interval', type=int, default=250)
parser.add_argument('--a_plus', type=int, default=1)
parser.add_argument('--a_minus', type=int, default=-0.5)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.add_argument('--no-gpu', dest='gpu', action='store_false')
parser.set_defaults(plot=False, gpu=False)

locals().update(vars(parser.parse_args()))

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	torch.cuda.manual_seed_all(seed)
else:
	torch.manual_seed(seed)

# Build network.
network = Network(dt=dt)

# Layers of neurons.
# Input layer.
input_layer = Input(n=6, traces=True)

# Excitatory layer.
exc_layer = LIFNodes(n=n_neurons, refractory=0, traces=True)

# Readout layer.
readout_layer = LIFNodes(n=2, refractory=0, traces=True)

# Connections between layers.
# Input -> excitatory.
input_exc_w = torch.rand(input_layer.n, exc_layer.n)
input_exc_conn = Connection(source=input_layer, target=exc_layer, w=input_exc_w,
										wmax=5.0, update_rule=m_stdp_et, nu=5e-4)
input_exc_norm = 2.5 * input_layer.n

# Excitatory -> readout.
exc_readout_w = 0.01 * torch.rand(exc_layer.n, readout_layer.n)
exc_readout_conn = Connection(source=exc_layer, target=readout_layer, w=exc_readout_w,
							  update_rule=m_stdp_et, nu=1e-2)
exc_readout_norm = 0.5 * exc_layer.n

# Readout -> readout.
readout_readout_w = -10 * torch.ones(readout_layer.n, readout_layer.n) + 10 * torch.diag(torch.ones(readout_layer.n))
readout_readout_conn = Connection(source=readout_layer, target=readout_layer, w=readout_readout_w, wmin=-10.0)

# Voltage recording for excitatory and readout layers.
exc_voltage_monitor = Monitor(exc_layer, ['v'])
readout_voltage_monitor = Monitor(readout_layer, ['v'])

# Add all layers and connections to the network.
network.add_layer(input_layer, name='X')
network.add_layer(exc_layer, name='E')
network.add_layer(readout_layer, name='R')
network.add_connection(input_exc_conn, source='X', target='E')
network.add_connection(exc_readout_conn, source='E', target='R')
network.add_connection(readout_readout_conn, source='R', target='R')
network.add_monitor(exc_voltage_monitor, name='exc_voltage')
network.add_monitor(readout_voltage_monitor, name='readout_voltage')

# Normalize adaptable weights.
network.connections[('X', 'E')].normalize(input_exc_norm)
network.connections[('E', 'R')].normalize(exc_readout_norm)

# Load CartPole environment.
env = CartPole()
env.reset()

i = 0
done = False
spikes, inpts = {}, {}

spike_record = {'X' : torch.zeros(input_layer.n, plot_interval),
				'E' : torch.zeros(exc_layer.n, plot_interval),
			    'R' : torch.zeros(readout_layer.n, plot_interval)}
voltage_record = {'E' : torch.zeros(exc_layer.n, plot_interval),
				  'R' : torch.zeros(readout_layer.n, plot_interval)}

print()

count = 0
lengths = []
rewards = []
sum_rewards = []

while True:
	count += 1
	env.render()
	
	if i % 100 == 0:
		print('Iteration %d' % i)
		
	if spikes == {} or spikes['R'].sum() == 0:
		# No spikes -> no action.
		action = 0
	else:
		# If there are spikes, randomly choose from among them for possible actions.
		action = torch.multinomial((spikes['R'] / spikes['R'].sum()).view(-1), 1)[0]
	
	obs, reward, done, info = env.step(action)
	
	# Update mean reward.
	rewards.append(reward)
	
	# Get next input sample.
	inpts.update({'X' : obs})
	
	# Run the network on the input.
	kwargs = {'reward' : reward, 'a_plus' : a_plus, 'a_minus' : a_minus}
	network.run(inpts=inpts, time=1, **kwargs)
	
	# Normalize adaptable weights.
	network.connections[('X', 'E')].normalize(input_exc_norm)
	network.connections[('E', 'R')].normalize(exc_readout_norm)
	
	# for key in spike_record:
	# 	spike_record[key][:, i % plot_interval] = spikes[key]
	
	# Get voltage recordings.
	exc_voltages = exc_voltage_monitor.get('v')
	voltage_record['E'][:, i % plot_interval] = exc_voltages
	exc_voltage_monitor._reset()
	
	readout_voltages = readout_voltage_monitor.get('v')
	voltage_record['R'][:, i % plot_interval] = readout_voltages
	readout_voltage_monitor._reset()
	
	# Optionally plot various simulation information.
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
			
			r_fig, r_ax = plt.subplots()
			r_line, = r_ax.plot(sum_rewards, marker='o')
			r_ax.set_xlabel('Episode'); r_ax.set_ylabel('Reward')
			r_ax.set_title('Reward per episode'); r_ax.grid()
			
		else:
			if i % plot_interval == 0:
				spike_ims, spike_axes = plot_spikes(spike_record, ims=spike_ims, axes=spike_axes)
				voltage_ims, voltage_axes = plot_voltages(voltage_record, ims=voltage_ims, axes=voltage_axes)
				
				i_e_w_im.set_data(input_exc_conn.w.t())
				e_r_w_im.set_data(exc_readout_conn.w.t())
				
				for ax in w_axes:
					ax.set_aspect('auto')
					
				r_line.set_xdata(range(len(sum_rewards)))
				r_line.set_ydata(sum_rewards)
				r_ax.relim(); r_ax.autoscale_view(True, True, True) 
				r_fig.canvas.draw()
				
				plt.pause(1e-8)
	
	i += 1
	
	if done == True:
		env.reset()
		network._reset()
		
		sum_rewards.append(np.sum(rewards))
		rewards = []
		
		lengths.append(count)
		count = 0

print()
	
env.render()