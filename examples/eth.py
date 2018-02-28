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
from network           import Network
from encoding          import get_poisson
from connections       import Connection, post_pre
from nodes             import AdaptiveLIFNodes, LIFNodes, Input
from analysis.plotting import plot_input, plot_spikes, plot_weights


def get_square_weights(weights, n_sqrt):
	square_weights = torch.zeros_like(torch.Tensor(28 * n_sqrt, 28 * n_sqrt))
	for i in range(n_sqrt):
		for j in range(n_sqrt):
			filter_ = weights[:, i * n_sqrt + j].contiguous().view(28, 28)
			square_weights[i * 28 : (i + 1) * 28, (j % n_sqrt) * 28 : ((j % n_sqrt) + 1) * 28] = filter_
	
	return square_weights

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
parser.add_argument('--min_isi', type=float, default=25.0)
parser.add_argument('--progress_interval', type=int, default=10)
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

# Build network.
network = Network(dt=dt)

n_sqrt = int(np.sqrt(n_neurons))

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

# Add all layers and connections to the network.
network.add_layer(input_layer, name='X')
network.add_layer(exc_layer, name='Ae')
network.add_layer(inh_layer, name='Ai')
network.add_connection(input_exc_conn, source='X', target='Ae')
network.add_connection(exc_inh_conn, source='Ae', target='Ai')
network.add_connection(inh_exc_conn, source='Ai', target='Ae')

# Load MNIST data.
images, labels = MNIST(path='../data').get_train()
images /= (255 * min_isi)  # Normalize and enforce minimum expected inter-spike interval.
images = images.view(images.size(0), -1)  # Flatten images to one dimension.

# Lazily encode data as Poisson spike trains.
data_loader = get_poisson(data=images, time=time)

# Train the network.
print('Begin training.\n')
start = default_timer()

for i in range(n_train):    
	if i % progress_interval == 0:
		print('Progress: %d / %d (%.4f seconds)' % (i, n_train, default_timer() - start))
		start = default_timer()
	
	# Get next input datum.
	sample = next(data_loader)
	inpts = {'X' : sample}
	
	# Run the network on the input for time `t`.
	spikes = network.run(inpts=inpts, time=time)
	network._reset()  # Reset state variables.
	network.connections[('X', 'Ae')].normalize()  # Normalize input -> excitatory weights
	
	# Optionally plot the excitatory, inhibitory spiking.
	if plot:
		inpt = inpts['X'].t()
		exc_spikes = spikes['Ae']; inh_spikes = spikes['Ai']
		input_exc_weights = network.connections[('X', 'Ae')].w
		square_weights = get_square_weights(input_exc_weights, n_sqrt)
		
		if i == 0:
			inpt_ims = plot_input(images[i].view(28, 28), inpt)
			spike_ims, spike_axes = plot_spikes({'Ae' : exc_spikes, 'Ai' : inh_spikes})
			weights_im = plot_weights(square_weights)
		else:
			inpt_ims = plot_input(images[i].view(28, 28), inpt, ims=inpt_ims)
			spike_ims, spike_axes = plot_spikes({'Ae' : exc_spikes, 'Ai' : inh_spikes}, ims=spike_ims, axes=spike_axes)
			weights_im = plot_weights(square_weights, im=weights_im)
		
		plt.pause(1e-8)

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, default_timer() - start))
print('Training complete.\n')