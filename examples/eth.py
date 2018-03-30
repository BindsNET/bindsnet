import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from timeit import default_timer

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'datasets')))

from datasets          import MNIST
from encoding          import get_poisson
from network           import Network, Monitor
from connections       import Connection, post_pre
from nodes             import AdaptiveLIFNodes, LIFNodes, Input

from evaluation        import *
from analysis.plotting import *


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
parser.add_argument('--intensity', type=float, default=1.0)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--no-plot', dest='plot', action='store_false')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.add_argument('--no-gpu', dest='gpu', action='store_false')
parser.set_defaults(plot=False, gpu=False, train=True)

locals().update(vars(parser.parse_args()))

if gpu:
	torch.set_default_tensor_type('torch.cuda.FloatTensor')
	torch.cuda.manual_seed_all(seed)
else:
	torch.manual_seed(seed)

if not train:
	update_interval = n_test

# Build network.
network = Network(dt=dt)

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Layers of neurons.
# Input layer.
input_layer = Input(n=784, traces=True, trace_tc=1 / 20)

# Excitatory layer.
exc_layer = AdaptiveLIFNodes(n=n_neurons, traces=True, rest=-65.0, reset=-65.0, threshold=-52.0, refractory=5,
                                    voltage_decay=1e-2, trace_tc=1 / 20, theta_plus=0.05, theta_decay=1e-7)

# Inhibitory layer.
inh_layer = LIFNodes(n=n_neurons, traces=False, rest=-60.0, reset=-45.0, threshold=-40.0,
                                 voltage_decay=1e-1, refractory=2, trace_tc=1 / 20)

# Connections between layers.
# Input -> excitatory.
input_exc_w = 0.3 * torch.rand(input_layer.n, exc_layer.n)
input_exc_conn = Connection(source=input_layer, target=exc_layer, w=input_exc_w, update_rule=post_pre, wmin=0.0, wmax=1.0)

# Excitatory -> inhibitory.
exc_inh_w = 22.5 * torch.diag(torch.ones(exc_layer.n))
exc_inh_conn = Connection(source=exc_layer, target=inh_layer, w=exc_inh_w, update_rule=None)

# Inhibitory -> excitatory.
inh_exc_w = -17.5 * (torch.ones(inh_layer.n, exc_layer.n) - torch.diag(torch.ones(inh_layer.n)))
inh_exc_conn = Connection(source=inh_layer, target=exc_layer, w=inh_exc_w, update_rule=None)

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(exc_layer, ['v'])
inh_voltage_monitor = Monitor(inh_layer, ['v'])

# Add all layers and connections to the network.
network.add_layer(input_layer, name='X')
network.add_layer(exc_layer, name='Ae')
network.add_layer(inh_layer, name='Ai')
network.add_connection(input_exc_conn, source='X', target='Ae')
network.add_connection(exc_inh_conn, source='Ae', target='Ai')
network.add_connection(inh_exc_conn, source='Ai', target='Ae')
network.add_monitor(exc_voltage_monitor, name='exc_voltage')
network.add_monitor(inh_voltage_monitor, name='inh_voltage')

# Load MNIST data.
images, labels = MNIST(path='../data').get_train()
images *= intensity
images /= 4  # Normalize and enforce minimum expected inter-spike interval.
images = images.view(images.size(0), -1)  # Flatten images to one dimension.

# Lazily encode data as Poisson spike trains.
data_loader = get_poisson(data=images, time=time)

# Record spikes during the simulation.
spike_record = torch.zeros_like(torch.Tensor(update_interval, time, n_neurons))
spike_record_full = torch.zeros_like(torch.Tensor(n_train, time, n_neurons))

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons))
proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
rates = torch.zeros_like(torch.Tensor(n_neurons, 10))

# Sequence of accuracy estimates.
accuracy = {'all' : [], 'proportion' : []}

# Train the network.
print('Begin training.\n')
start = default_timer()

for i in range(n_train):    
	if i % progress_interval == 0:
		print('Progress: %d / %d (%.4f seconds)' % (i, n_train, default_timer() - start))
		start = default_timer()
	
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
	sample = next(data_loader)
	inpts = {'X' : sample}
	
	# Run the network on the input.
	n_spikes = 0; n_retries = 0; intensity = start_intensity
	while n_spikes < 5 and n_retries < 3:
		spikes = network.run(inpts=inpts, time=time)
		n_spikes = spikes['Ae'].sum(); intensity += 1
		image = images[i] * intensity
		sample = next(get_poisson(image.view(1, -1), time=time))
	
	# Get voltage recording.
	exc_voltages = exc_voltage_monitor.get('v')
	inh_voltages = inh_voltage_monitor.get('v')
	
	network._reset()  # Reset state variables.
	network.connections[('X', 'Ae')].normalize()  # Normalize input -> excitatory weights
	
	# Optionally plot various simulation information.
	if plot:
		inpt = inpts['X'].t()
		exc_spikes = spikes['Ae']; inh_spikes = spikes['Ai']
		input_exc_weights = network.connections[('X', 'Ae')].w
		square_weights = get_square_weights(input_exc_weights, n_sqrt)
		square_assignments = get_square_assignments(assignments, n_sqrt)
		voltages = {'Ae' : exc_voltages.numpy().T, 'Ai' : inh_voltages.numpy().T}
		
		if i == 0:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i])
			spike_ims, spike_axes = plot_spikes({'Ae' : exc_spikes, 'Ai' : inh_spikes})
			weights_im = plot_weights(square_weights)
			assigns_im = plot_assignments(square_assignments)
			perf_ax = plot_performance(accuracy)
			voltage_ims, voltage_axes = plot_voltages(voltages)
			
		else:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
			spike_ims, spike_axes = plot_spikes({'Ae' : exc_spikes, 'Ai' : inh_spikes}, ims=spike_ims, axes=spike_axes)
			weights_im = plot_weights(square_weights, im=weights_im)
			assigns_im = plot_assignments(square_assignments, im=assigns_im)
			perf_ax = plot_performance(accuracy, ax=perf_ax)
			voltage_ims, voltage_axes = plot_voltages(voltages, ims=voltage_ims, axes=voltage_axes)
		
		plt.pause(1e-8)

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, default_timer() - start))
print('Training complete.\n')
