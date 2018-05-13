import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from bindsnet import *
from time     import time as t

def get_square_weights(weights, n_sqrt):
	square_weights = torch.zeros_like(torch.Tensor(28 * n_sqrt, 28 * n_sqrt))
	for i in range(n_sqrt):
		for j in range(n_sqrt):
			if not i * n_sqrt + j < weights.size(1):
				break
			
			fltr = weights[:, i * n_sqrt + j].contiguous().view(28, 28)
			square_weights[i * 28 : (i + 1) * 28, (j % n_sqrt) * 28 : ((j % n_sqrt) + 1) * 28] = fltr
	
	return square_weights

def get_square_assignments(assignments, n_sqrt):
	square_assignments = -1 * torch.ones_like(torch.Tensor(n_sqrt, n_sqrt))
	for i in range(n_sqrt):
		for j in range(n_sqrt):
			if not i * n_sqrt + j < assignments.size(0):
				break
			
			assignment = assignments[i * n_sqrt + j]
			square_assignments[i : (i + 1), (j % n_sqrt) : ((j % n_sqrt) + 1)] = assignments[i * n_sqrt + j]
	
	return square_assignments

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--excite', type=float, default=22.5)
parser.add_argument('--inhib', type=float, default=17.5)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--dt', type=int, default=1.0)
parser.add_argument('--intensity', type=float, default=0.25)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)

locals().update(vars(parser.parse_args()))

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	torch.cuda.manual_seed_all(seed)
else:
	torch.manual_seed(seed)

if not train:
	update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity
	
# Build network.
network = Network()
input_layer = Input(n=784,
					shape=(1, 1, 28, 28),
					traces=True)

output_layer = LIFNodes(n=16*24*24,
						shape=(1, 16, 24, 24),
						traces=True)

conv_layer = Conv2dConnection(input_layer,
							  output_layer,
							  kernel_size=(5, 5),
							  stride=(1, 1),
							  update_rule=post_pre)

network.add_layer(input_layer, name='X')
network.add_layer(output_layer, name='Y')
network.add_connection(conv_layer, source='X', target='Y')

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

# Load MNIST data.
images, labels = MNIST(path=os.path.join('..', '..', 'data', 'MNIST')).get_train()

# Lazily encode data as Poisson spike trains.
data_loader = poisson_loader(data=images, time=time)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons))
proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
rates = torch.zeros_like(torch.Tensor(n_neurons, 10))

# Sequence of accuracy estimates.
accuracy = {'all' : [], 'proportion' : []}

spikes = {}
for layer in set(network.layers):
	spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
	network.add_monitor(spikes[layer], name='%s_spikes' % layer)

# Train the network.
print('Begin training.\n')
start = t()

for i in range(n_train):    
	if i % progress_interval == 0:
		print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start))
		start = t()
	
	if i % update_interval == 0 and i > 0:
		# Get network predictions.
		all_activity_pred = all_activity(spike_record, assignments, 10)
		proportion_pred = proportion_weighting(spike_record, assignments, proportions, 10)

		# Compute network accuracy according to available classification strategies.
		accuracy['all'].append(100 * torch.sum(labels[i - update_interval:i].long() \
												== all_activity_pred) / update_interval)
		accuracy['proportion'].append(100 * torch.sum(labels[i - update_interval:i].long() \
														== proportion_pred) / update_interval)

		print('\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)' \
						% (accuracy['all'][-1], np.mean(accuracy['all']), np.max(accuracy['all'])))
		print('Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n' \
						% (accuracy['proportion'][-1], np.mean(accuracy['proportion']),
						  np.max(accuracy['proportion'])))

		# Assign labels to excitatory layer neurons.
		assignments, proportions, rates = assign_labels(spike_record, labels[i - update_interval:i], 10, rates)
	
	# Get next input sample.
	sample = next(data_loader).unsqueeze(1).unsqueeze(1)
	inpts = {'X' : sample}
	
	# Run the network on the input.
	network.run(inpts=inpts, time=time)
	
	# Get voltage recording.
	voltages = voltage_monitor.get('v')
	
	# Add to spikes recording.
	# spike_record[i % update_interval] = spikes['Y'].get('s').t()
	
	# Optionally plot various simulation information.
	if plot:
		inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
		weights = network.connections[('X', 'Y')].w.view(16, 25)
		square_assignments = get_square_assignments(assignments, n_sqrt)
		voltages = {'Y' : voltages}
		_spikes = {'X' : spikes['X'].get('s').view(time, 28 ** 2),
				   'Y' : spikes['Y'].get('s').view(time, 16 * 24 ** 2)}
		
		if i == 0:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt.view(28, 28), label=labels[i])
			spike_ims, spike_axes = plot_spikes(_spikes)
			weights_im = plot_weights(weights)
			assigns_im = plot_assignments(square_assignments)
			perf_ax = plot_performance(accuracy)
			# voltage_ims, voltage_axes = plot_voltages(voltages)
			
		else:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
			spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
			weights_im = plot_weights(weights, im=weights_im)
			assigns_im = plot_assignments(square_assignments, im=assigns_im)
			perf_ax = plot_performance(accuracy, ax=perf_ax)
			# voltage_ims, voltage_axes = plot_voltages(voltages, ims=voltage_ims, axes=voltage_axes)
		
		plt.pause(1e-8)
	
	network._reset()  # Reset state variables.

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, t() - start))
print('Training complete.\n')
