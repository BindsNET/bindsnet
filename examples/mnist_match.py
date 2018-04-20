import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from time                         import time

from bindsnet.learning          import *
from bindsnet.encoding          import *
from bindsnet.analysis.plotting import *
from bindsnet.datasets          import MNIST
from bindsnet.network           import Network

from bindsnet.network.nodes     import *
from bindsnet.network.monitors  import Monitor
from bindsnet.network.topology  import Connection

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=100)
parser.add_argument('-i', type=int, default=250 * 1000)
parser.add_argument('--intensity', type=float, default=1.0)
parser.add_argument('--target_rate', type=float, default=0.08)
parser.add_argument('--low_rate', type=float, default=0.0)
parser.add_argument('--plot_interval', type=int, default=500)
parser.add_argument('--print_interval', type=int, default=25)
parser.add_argument('--change_interval', type=int, default=500)
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)
locals().update(vars(parser.parse_args()))

def get_square_weights(weights, n_sqrt):
	square_weights = torch.zeros_like(torch.Tensor(28 * n_sqrt, 28 * n_sqrt))
	for i in range(n_sqrt):
		for j in range(n_sqrt):
			if not i * n_sqrt + j < weights.size(1):
				break
			
			fltr = weights[:, i * n_sqrt + j].contiguous().view(28, 28)
			square_weights[i * 28 : (i + 1) * 28, (j % n_sqrt) * 28 : ((j % n_sqrt) + 1) * 28] = fltr
	
	return square_weights

sqrt = int(np.ceil(np.sqrt(n)))

network = Network(dt=1.0)

inpt = Input(n=784, traces=True)
exc = AdaptiveLIFNodes(n=n, traces=True)
output = AdaptiveLIFNodes(n=10, traces=True)

ew = 0.3 * torch.rand(784, n)
econn = Connection(inpt, exc, w=ew, update_rule=m_stdp_et, nu=0.1, wmin=0, wmax=1)
ow = 10 * torch.rand(n, 10)
oconn = Connection(exc, output, w=ow, update_rule=m_stdp_et, nu=0.1, wmin=0, wmax=10)

network.add_layer(inpt, 'X')
network.add_layer(exc, 'Y')
network.add_layer(output, 'Z')
network.add_connection(econn, source='X', target='Y')
network.add_connection(oconn, source='Y', target='Z')

spike_monitors = {layer : Monitor(network.layers[layer], ['s']) for layer in network.layers}
for layer in spike_monitors:
	network.add_monitor(spike_monitors[layer], '%s' % layer)

# Load MNIST data.
images, labels = MNIST(path=os.path.join('..', 'data')).get_train()
images *= intensity
images /= 4

# Lazily encode data as Poisson spike trains.
ims = []
for j in range(0, i, change_interval):
	ims.extend([images[j % 60000]] * change_interval)

lbls = []
for j in range(0, i, change_interval):
	lbls.extend([int(labels[j % 60000])] * change_interval)

loader = get_bernoulli(data=torch.stack(ims), time=1, max_prob=0.05)

reward = 0
out_reward = 0
a_plus = 1
a_minus = 0

avg_rates = torch.zeros(n)
target_rates = low_rate * torch.ones(n)
for j in np.random.choice(range(n)[lbls[0]::10], 5, replace=False):
	target_rates[j] = target_rate

distances = [torch.sum(torch.sqrt((target_rates - avg_rates) ** 2))]
rewards = [reward]

out_avg_rates = torch.zeros(10)
out_target_rates = low_rate * torch.ones(10)
out_target_rates[lbls[0]] = target_rate

out_distances = [torch.sum(torch.sqrt((out_target_rates - out_avg_rates) ** 2))]
out_rewards = [out_reward]

spike_record = {layer : torch.zeros(network.layers[layer].n, plot_interval) for layer in network.layers}

print()
for i in range(i):
	if i > 0 and i % change_interval == 0:
		avg_rates = torch.zeros(n)
		target_rates = low_rate * torch.ones(n)
		for j in np.random.choice(range(n)[lbls[i]::10], 5, replace=False):
			target_rates[j] = target_rate
		
		out_avg_rates = torch.zeros(10)
		out_target_rates = low_rate * torch.ones(10)
		out_target_rates[lbls[i]] = target_rate
	
	inpts = {'X' : next(loader)}
	kwargs = {str(('X', 'Y')) : {'reward' : reward, 'a_plus' : a_plus, 'a_minus' : a_minus},
			  str(('Y', 'Z')) : {'reward' : out_reward, 'a_plus' : a_plus, 'a_minus' : a_minus}}
	
	network.run(inpts, 1, **kwargs)
	econn.normalize()
	
	spikes = {layer : spike_monitors[layer].get('s').view(-1) for layer in spike_monitors}
	for layer in spike_record:
		spike_record[layer][:, i % plot_interval] = spikes[layer]
	
	a = i % change_interval
	if a == 0:
		avg_rates = spikes['Y']
		out_avg_rates = spikes['Z']
	else:
		avg_rates = ((a - 1) / a) * avg_rates + (1 / a) * spikes['Y']
		out_avg_rates = ((a - 1) / a) * out_avg_rates + (1 / a) * spikes['Z']
	
	reward = target_rates - avg_rates
	reward[reward < 0] = 0
	rewards.append(reward.sum())
	distance = torch.sum(torch.sqrt((target_rates - avg_rates) ** 2))
	distances.append(distance)
	
	out_reward = out_target_rates - out_avg_rates
	out_rewards.append(out_reward.sum())
	out_distance = torch.sum(torch.sqrt((out_target_rates - out_avg_rates) ** 2))
	out_distances.append(out_distance)
	
	if i > 0 and i % change_interval == 0:
		sums = []
		for j in range(10):
			if gpu:
				sums.append(np.sum(spike_record['Z'].cpu().numpy()))
			else:
				sums.append(np.sum(spike_record['Z'].numpy()))
		
		a = int(i / change_interval)
		if a == 1:
			performance = np.argmax(sums) == lbls[i - 1]
		else:
			performance = ((a - 1) / a) * performance + (1 / a) * int(np.argmax(sums) == lbls[i])
		
		print('Performance on iteration %d: %.2f' % (i, performance * 100))
		
	for m in spike_monitors:
		spike_monitors[m]._reset()
	
	if plot:
		if i == 0:
			spike_ims, spike_axes = plot_spikes(spike_record)
			weights_im = plot_weights(get_square_weights(econn.w, sqrt))
			out_weights_im = plot_weights(oconn.w, wmax=10)

			fig, axes = plt.subplots(2, 1); ims = []
			ims.append(axes[0].matshow(torch.stack([avg_rates, target_rates]), cmap='hot_r'))
			axes[0].set_xticks(()); axes[0].set_yticks([0, 1])
			axes[0].set_yticklabels(['Actual', 'Targets'])
			axes[0].set_aspect('auto')
			axes[0].set_title('Difference between target and actual firing rates.')
			
			ims.append(axes[1].matshow(torch.stack([out_avg_rates, out_target_rates]), cmap='hot_r'))
			axes[1].set_xticks(()); axes[1].set_yticks([0, 1])
			axes[1].set_yticklabels(['Actual', 'Targets'])
			axes[1].set_aspect('auto')
			axes[1].set_title('Difference between target and actual firing rates.')
			
			plt.tight_layout()
			
			fig2, ax2 = plt.subplots()
			line2, = ax2.semilogy(distances, label='Exc. distance')
			line3, = ax2.semilogy(out_distances, label='Output distance')
			ax2.axhline(0, ls='--', c='r')
			ax2.set_title('Sum of squared differences over time')
			ax2.set_xlabel('Timesteps')
			ax2.set_ylabel('Sum of squared differences')
			plt.legend()
			
			plt.pause(1e-8)

		elif (i + 1) % plot_interval == 0:
			spike_ims, spike_axes = plot_spikes(spike_record, spike_ims, spike_axes)
			weights_im = plot_weights(get_square_weights(econn.w, sqrt), im=weights_im)
			out_weights_im = plot_weights(oconn.w, im=out_weights_im, wmax=10)
			
			ims[0].set_data(torch.stack([avg_rates, target_rates]))
			ims[1].set_data(torch.stack([out_avg_rates, out_target_rates]))
			
			if not len(distances) > change_interval:
				line2.set_xdata(range(len(distances)))
				line2.set_ydata(distances)
				line3.set_xdata(range(len(out_distances)))
				line3.set_ydata(out_distances)
			else:
				line2.set_xdata(range(len(distances) - change_interval, len(distances)))
				line2.set_ydata(distances[-change_interval:])
				line3.set_xdata(range(len(out_distances) - change_interval, len(out_distances)))
				line3.set_ydata(out_distances[-change_interval:])
			
			ax2.relim() 
			ax2.autoscale_view(True, True, True) 

			plt.pause(1e-8)