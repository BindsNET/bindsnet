import os
import torch
import argparse
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.encoding import poisson_loader
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Conv2dConnection, Connection
from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_conv2d_weights, plot_voltages

print()

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_train', type=int, default=60000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--kernel_size', type=int, default=16)
parser.add_argument('--stride', type=int, default=4)
parser.add_argument('--n_filters', type=int, default=25)
parser.add_argument('--padding', type=int, default=0)
parser.add_argument('--time', type=int, default=50)
parser.add_argument('--dt', type=int, default=1.0)
parser.add_argument('--intensity', type=float, default=1)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_train = args.n_train
n_test = args.n_test
kernel_size = args.kernel_size
stride = args.stride
n_filters = args.n_filters
padding = args.padding
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

conv_size = int((28 - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build network.
network = Network()
input_layer = Input(n=784, shape=(1, 1, 28, 28), traces=True)

conv_layer = DiehlAndCookNodes(n=n_filters * conv_size * conv_size,
                               shape=(1, n_filters, conv_size, conv_size),
                               traces=True)

conv_conn = Conv2dConnection(input_layer, conv_layer, kernel_size=kernel_size, stride=stride, update_rule=PostPre,
                             norm=0.4 * kernel_size ** 2, nu_pre=1e-4, nu_post=1e-2, wmax=1.0)

# indices, values = [], []
# # w = torch.zeros(1, n_filters, conv_size, conv_size, 1, n_filters, conv_size, conv_size)
# for fltr1 in range(n_filters):
#   for fltr2 in range(n_filters):
#       if fltr1 != fltr2:
#           for i in range(conv_size):
#               for j in range(conv_size):
#                   # indices.append([0, fltr1, i, j, 0, fltr2, i, j])
#                   indices.append([fltr1 * (i * conv_size) + j, fltr2 * (i + conv_size) + j])
#                   values.append(-17.5)
#                   # w[0, fltr1, i, j, 0, fltr2, i, j] = -17.5

# w = torch.sparse.FloatTensor(torch.Tensor(indices).long().t(),
#                            torch.Tensor(values).float(),
#                            torch.Size([n_filters * conv_size ** 2, n_filters * conv_size ** 2]))

# recurrent_conn = Connection(conv_layer,
#                                 conv_layer,
#                                 w=w)

w = torch.zeros(1, n_filters, conv_size, conv_size, 1, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[0, fltr1, i, j, 0, fltr2, i, j] = -100.0

recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name='X')
network.add_layer(conv_layer, name='Y')
network.add_connection(conv_conn, source='X', target='Y')
network.add_connection(recurrent_conn, source='Y', target='Y')

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers['Y'], ['v'], time=time)
network.add_monitor(voltage_monitor, name='output_voltage')

# Load MNIST data.
images, labels = MNIST(path=os.path.join('..', '..', 'data', 'MNIST'), download=True).get_train()
images *= intensity

# Lazily encode data as Poisson spike trains.
data_loader = poisson_loader(data=images, time=time)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)

voltages = {}
for layer in set(network.layers) - {'X'}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=['v'], time=time)
    network.add_monitor(voltages[layer], name='%s_voltages' % layer)

# Train the network.
print('Begin training.\n');
start = t()

for i in range(n_train):
    if i % progress_interval == 0:
        print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start));
        start = t()

    # Get next input sample.
    sample = next(data_loader).unsqueeze(1).unsqueeze(1)
    inpts = {'X': sample}

    # Run the network on the input.
    # choice = torch.bernoulli(0.01 * torch.rand(1, 16, 24, 24))
    # clamp = {'Y' : choice.byte()}
    network.run(inpts=inpts, time=time)  # , clamp=clamp)

    # Optionally plot various simulation information.
    if plot:
        inpt = inpts['X'].view(time, 784).sum(0).view(28, 28)
        weights1 = conv_conn.w
        _spikes = {'X': spikes['X'].get('s').view(28 ** 2, time),
                   'Y': spikes['Y'].get('s').view(n_filters * conv_size ** 2, time)}
        _voltages = {'Y': voltages['Y'].get('v').view(n_filters * conv_size ** 2, time)}

        if i == 0:
            inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i])
            spike_ims, spike_axes = plot_spikes(_spikes)
            weights1_im = plot_conv2d_weights(weights1, wmax=conv_conn.wmax)
            voltage_ims, voltage_axes = plot_voltages(_voltages)

        else:
            inpt_axes, inpt_ims = plot_input(images[i].view(28, 28), inpt, label=labels[i], axes=inpt_axes,
                                             ims=inpt_ims)
            spike_ims, spike_axes = plot_spikes(_spikes, ims=spike_ims, axes=spike_axes)
            weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
            voltage_ims, voltage_axes = plot_voltages(_voltages, ims=voltage_ims, axes=voltage_axes)

        plt.pause(1e-8)

    network.reset_()  # Reset state variables.

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, t() - start))
print('Training complete.\n')
