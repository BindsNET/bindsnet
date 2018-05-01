import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from time                         import time

from bindsnet.analysis.plotting import *
from bindsnet.learning          import *
from bindsnet.network           import Network
from bindsnet.encoding          import get_bernoulli

from bindsnet.network.monitors  import Monitor
from bindsnet.network.nodes     import LIFNodes, Input
from bindsnet.network.topology  import Connection

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=100)
parser.add_argument('-i', type=int, default=100000)
parser.add_argument('--plot_interval', type=int, default=500)
parser.add_argument('--print_interval', type=int, default=25)
parser.add_argument('--change_interval', type=int, default=10000)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)
locals().update(vars(parser.parse_args()))

sqrt = int(np.sqrt(n))

network = Network(dt=1.0)

inpt = Input(100, traces=True)
output = LIFNodes(n, traces=True)

w = 1 * torch.rand(100, n)
conn = Connection(inpt, output, w=w, update_rule=m_stdp, nu=1, wmin=0, wmax=1)

network.add_layer(inpt, 'X')
network.add_layer(output, 'Y')
network.add_connection(conn, source='X', target='Y')

spike_monitors = {layer : Monitor(network.layers[layer], ['s']) for layer in network.layers}
for layer in spike_monitors:
	network.add_monitor(spike_monitors[layer], '%s' % layer)

data = torch.rand(i, 100)
loader = get_bernoulli(data, time=1, max_prob=0.05)

reward = 1e-4
a_plus = 1
a_minus = 0

avg_rates = torch.zeros(n)
target_rates = 0.03 + torch.rand(n) / 20

distances = [torch.sum(torch.sqrt((target_rates - avg_rates) ** 2))]
rewards = [reward]

spike_record = {'X' : torch.zeros(plot_interval, 100),
			    'Y' : torch.zeros(plot_interval, n)}

print()
for i in range(i):
	if i > 0 and i % change_interval == 0:
		avg_rates = torch.zeros(n)
		target_rates = 0.03 + torch.rand(n) / 20
	
	inpts = {'X' : next(loader)}
	kwargs = {str(('X', 'Y')) : {'reward' : reward, 'a_plus' : a_plus, 'a_minus' : a_minus}}
	network.run(inpts, 1, **kwargs)
	
	spikes = {layer : spike_monitors[layer].get('s').view(-1) for layer in spike_monitors}
	for layer in spike_record:
		spike_record[layer][i % plot_interval] = spikes[layer]
	
	a = i % change_interval
	if a == 0:
		avg_rates = spikes['Y']
	else:
		avg_rates = ((a - 1) / a) * avg_rates + (1 / a) * spikes['Y']
	
	reward = target_rates - avg_rates
	rewards.append(reward.sum())
	
	distance = torch.sum(torch.sqrt((target_rates - avg_rates) ** 2))
	distances.append(distance)
	
	if i % print_interval == 0:
		print('Current distance (iteration %d): %.4f' % (i, distances[-1].item()))
		
	for m in spike_monitors:
		spike_monitors[m]._reset()
	
	if plot:
		if i == 0:
			spike_ims, spike_axes = plot_spikes(spike_record)
			weights_im = plot_weights(conn.w.view(100, n))
			
			fig, ax = plt.subplots()
			im = ax.matshow(torch.stack([avg_rates, target_rates, torch.abs(avg_rates - target_rates)]), cmap='hot_r')
			ax.set_xticks(()); ax.set_yticks([0, 1, 2])
			ax.set_yticklabels(['Actual rates', 'Target rates', 'Differences'])
			ax.set_aspect('auto')
			ax.set_title('Difference between target and actual firing rates.')
			cbar = plt.colorbar(im)
			plt.tight_layout()
			
			fig2, ax2 = plt.subplots()
			line2, = ax2.semilogy(distances, label='Distance')
			ax2.axhline(0, ls='--', c='r')
			ax2.set_title('Sum of squared differences over time')
			ax2.set_xlabel('Timesteps')
			ax2.set_ylabel('Sum of squared differences')
			plt.legend()
			
			plt.pause(1e-8)

		elif i % plot_interval == 0:
			spike_ims, spike_axes = plot_spikes(spike_record, spike_ims, spike_axes)
			weights_im = plot_weights(conn.w.view(100, n), im=weights_im)

			im.set_data(torch.stack([avg_rates, target_rates, torch.abs(avg_rates - target_rates)]))
			cbar.set_clim(vmin=0, vmax=max(avg_rates))
			cbar_ticks = np.linspace(0., max(avg_rates), num=11, endpoint=True)
			cbar.set_ticks(cbar_ticks) 
			cbar.draw_all()
			
			if not len(distances) > change_interval:
				line2.set_xdata(range(len(distances)))
				line2.set_ydata(distances)
			else:
				line2.set_xdata(range(len(distances) - change_interval, len(distances)))
				line2.set_ydata(distances[-change_interval:])
			
			ax2.relim() 
			ax2.autoscale_view(True, True, True) 

			plt.pause(1e-8)
