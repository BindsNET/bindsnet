import os
import sys
import torch
import numpy             as np
import pickle            as p
import argparse
import matplotlib.pyplot as plt

from tqdm                 import tqdm
from timeit               import default_timer
from sklearn.linear_model import LogisticRegression

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

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=100)
parser.add_argument('--n_test', type=int, default=100)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--dt', type=int, default=1.0)
parser.add_argument('--intensity', type=float, default=1.0)
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

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

if train:
	# Build network.
	network = Network(dt=dt)

	# Groups of neurons.
	# Input.
	inputs = Input(n=784)

	# Reservoir.
	rest = -65.0 + torch.randn(n_neurons)
	reset = -65.0 + torch.randn(n_neurons)
	threshold = -52.0 + torch.randn(n_neurons)
	reservoir = LIFNodes(n=n_neurons, rest=rest, reset=reset, threshold=threshold)

	# Connections between layers.
	# Input -> reservoir.
	input_w = 0.1 * torch.rand(inputs.n, reservoir.n)
	input_conn = Connection(source=inputs, target=reservoir, w=input_w)

	# Reservoir -> reservoir.
	reservoir_w = (100 / (n_neurons ** 2)) * torch.rand(reservoir.n, reservoir.n)
	reservoir_conn = Connection(source=reservoir, target=reservoir, w=reservoir_w)

	# Add all layers and connections to the network.
	network.add_layer(inputs, name='X')
	network.add_layer(reservoir, name='R')
	network.add_connection(input_conn, source='X', target='R')
	network.add_connection(reservoir_conn, source='R', target='R')

	# Serialize network object to disk.
	fname = '_'.join(map(str, [n_neurons, n_train, 'lsm.p']))
	p.dump(network, open(os.path.join('lsm', fname), 'wb'))
else:
	fname = '_'.join(map(str, [n_neurons, n_train, 'lsm.p']))
	network = p.load(open(os.path.join('lsm', fname), 'rb'))

# Load MNIST data.
if train:
	images, labels = MNIST(path=os.path.join('..', 'data')).get_train()
else:
	images, labels = MNIST(path=os.path.join('..', 'data')).get_test()

images *= intensity
images /= 4  # Normalize and enforce minimum expected inter-spike interval.
images = images.view(images.size(0), -1)  # Flatten images to one dimension.

# Lazily encode data as Poisson spike trains.
data_loader = get_poisson(data=images, time=time)

# Record reservoir spikes per neuron, per example.
if train:
	spike_counts = torch.zeros(n_train, n_neurons)
else:
	spike_counts = torch.zeros(n_test, n_neurons)

if train:
	samples = n_train
else:
	samples = n_test

for i in tqdm(range(samples)):
	# Get next training image.
	sample = next(data_loader)
	inpts = {'X' : sample}
	
	# Run the network on the input.
	spikes = network.run(inpts=inpts, time=time)
	
	# Calculate sum of spikes per reservoir neuron.
	spike_counts[i] = torch.sum(spikes['R'], 1)
	
	network._reset()  # Reset state variables.

	# Optionally plot various simulation information.
	if plot:
		inpt = inpts['X'].t()
		x_spikes = spikes['X']; r_spikes = spikes['R']
		
		if i == 0:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i])
			spike_ims, spike_axes = plot_spikes({'X' : x_spikes, 'R' : r_spikes})
			
		else:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
			spike_ims, spike_axes = plot_spikes({'X' : x_spikes, 'R' : r_spikes}, ims=spike_ims, axes=spike_axes)
		
		plt.pause(1e-8)

if train:
	if gpu:
		labels = labels.cpu().numpy()[:n_train]
	else:
		labels = labels.numpy()[:n_train]
	
	model = LogisticRegression()
	model.fit(spike_counts, labels)
	predictions = model.predict(spike_counts)

	print()
	print('Training accuracy:', np.mean(predictions == labels))
	print()

	fname = '_'.join(map(str, [n_neurons, n_train, 'lr.p']))
	p.dump(model, open(os.path.join('lsm', fname), 'wb'))
else:
	if gpu:
		labels = labels.cpu().numpy()[:n_test]
	else:
		labels = labels.numpy()[:n_test]
	
	fname = '_'.join(map(str, [n_neurons, n_train, 'lr.p']))
	model = p.load(open(os.path.join('lsm', fname), 'rb'))
	predictions = model.predict(spike_counts)
	
	print(predictions)
	print(labels)

	print()
	print('Test accuracy:', np.mean(predictions == labels))
	print()
