import os
import sys
import torch
import numpy             as np
import argparse
import matplotlib.pyplot as plt

from bindsnet import *
from time     import time as t

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--kernel_size', type=int, default=5)
parser.add_argument('--stride', type=int, default=1)
parser.add_argument('--n_filters', type=int, default=16)
parser.add_argument('--time', type=int, default=100)
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

conv_size = int((28 - kernel_size + 1) / stride)

# Build network.
network = Network()
input_layer = Input(n=784,
					shape=(1, 1, 28, 28),
					traces=True)

conv_layer = LIFNodes(n=n_filters * conv_size * conv_size,
						shape=(1, n_filters, conv_size, conv_size),
						traces=True)

output_layer = LIFNodes(n=10,
						traces=True)

conv_weights = Conv2dConnection(input_layer,
							    conv_layer,
							    kernel_size=kernel_size,
							    stride=stride,
							    update_rule=post_pre,
							    norm=0.2 * conv_layer.shape[1],
							    nu_pre=0,
							    nu_post=1e-3,
							    wmax=0.5)

output_weights = Connection(conv_layer,
						    output_layer,
						    update_rule=post_pre,
						    wmax=1.0)

network.add_layer(input_layer, name='X')
network.add_layer(conv_layer, name='Y')
network.add_layer(output_layer, name='Z')
network.add_connection(conv_weights, source='X', target='Y')
network.add_connection(output_weights, source='Y', target='Z')

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

# Load MNIST data.
images, labels = MNIST(path=os.path.join('..', '..', 'data', 'MNIST')).get_train()
images *= intensity

# Lazily encode data as Poisson spike trains.
data_loader = poisson_loader(data=images, time=time)

spikes = {}
for layer in set(network.layers):
	spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
	network.add_monitor(spikes[layer], name='%s_spikes' % layer)

# Train the network.
print('Begin training.\n'); start = t()

for i in range(n_train):    
	if i % progress_interval == 0:
		print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start)); start = t()
	
	# Get next input sample.
	sample = next(data_loader).unsqueeze(1).unsqueeze(1)
	inpts = {'X' : sample}
	
	# Run the network on the input.
	clamp = {'Z' : labels[i].long()}
	network.run(inpts=inpts, time=time, clamp=clamp)
	
	# Get voltage recording.
	voltages = voltage_monitor.get('v')
	
	# Optionally plot various simulation information.
	if plot:
		inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
		weights1 = conv_weights.w.view(conv_weights.w.size(0), conv_weights.w.size(2) * conv_weights.w.size(3))
		weights2 = output_weights.w.view(output_weights.w.size(1) * output_weights.w.size(2) * output_weights.w.size(3),
									   output_weights.w.size(4))
		voltages = {'Y' : voltages}
		_spikes = {'X' : spikes['X'].get('s').view(time, 28 ** 2),
				   'Y' : spikes['Y'].get('s').view(time, n_filters * conv_size ** 2),
				   'Z' : spikes['Z'].get('s').view(time, 10)}
		
		if i == 0:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i])
			spike_ims, spike_axes = plot_spikes(_spikes)
			weights1_im = plot_weights(weights1, wmax=conv_weights.wmax)
			weights2_im = plot_weights(weights2, wmax=output_weights.wmax)
			
		else:
			inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
			spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
			weights1_im = plot_weights(weights1, im=weights1_im)
			weights2_im = plot_weights(weights2, im=weights2_im)
		
		plt.pause(1e-8)
	
	network._reset()  # Reset state variables.

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, t() - start))
print('Training complete.\n')
