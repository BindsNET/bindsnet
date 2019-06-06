import os
import torch
from torchvision import transforms
import argparse
import matplotlib.pyplot as plt

from time import time as t
from tqdm import tqdm

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.learning import PostPre
from bindsnet.encoding import poisson_loader
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import DiehlAndCookNodes, Input
from bindsnet.network.topology import Conv2dConnection, Connection
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_conv2d_weights,
    plot_voltages,
)

print()

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--kernel_size", type=int, default=16)
parser.add_argument("--stride", type=int, default=4)
parser.add_argument("--n_filters", type=int, default=25)
parser.add_argument("--padding", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128.0)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_epochs = args.n_epochs
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

conv_layer = DiehlAndCookNodes(
    n=n_filters * conv_size * conv_size,
    shape=(1, n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=PostPre,
    norm=0.4 * kernel_size ** 2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
)

w = torch.zeros(1, n_filters, conv_size, conv_size, 1, n_filters, conv_size, conv_size)
for fltr1 in range(n_filters):
    for fltr2 in range(n_filters):
        if fltr1 != fltr2:
            for i in range(conv_size):
                for j in range(conv_size):
                    w[0, fltr1, i, j, 0, fltr2, i, j] = -100.0

w = w.view(n_filters * conv_size * conv_size, n_filters * conv_size * conv_size)
recurrent_conn = Connection(conv_layer, conv_layer, w=w)

network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_connection(conv_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

# Voltage recording for excitatory and inhibitory layers.
voltage_monitor = Monitor(network.layers["Y"], ["v"], time=time)
network.add_monitor(voltage_monitor, name="output_voltage")

if gpu:
    network.to('cuda')

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=4,
        pin_memory=gpu
    )

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Get next input sample.

        inpts = {"X": batch["encoded_image"]}
        if gpu:
            inpts = {k:v.cuda() for k, v in inpts.items()}
        label = batch["label"]
        
        # Run the network on the input.
        network.run(inpts=inpts, time=time, input_time_dim=1)

        # Optionally plot various simulation information.
        if plot:
            image = batch["image"].view(28, 28)

            inpt = inpts["X"].view(time, 784).sum(0).view(28, 28)
            weights1 = conv_conn.w
            _spikes = {
                "X": spikes["X"].get("s").view(28 ** 2, time),
                "Y": spikes["Y"].get("s").view(n_filters * conv_size ** 2, time),
            }
            _voltages = {
                "Y": voltages["Y"].get("v").view(n_filters * conv_size ** 2, time)
            }

            if step == 0 and epoch == 0:
                inpt_axes, inpt_ims = plot_input(image, inpt, label=label)
                spike_ims, spike_axes = plot_spikes(_spikes)
                weights1_im = plot_conv2d_weights(weights1, wmax=conv_conn.wmax)
                voltage_ims, voltage_axes = plot_voltages(_voltages)

            else:
                inpt_axes, inpt_ims = plot_input(
                    image, inpt, label=label, axes=inpt_axes, ims=inpt_ims
                )
                spike_ims, spike_axes = plot_spikes(
                    _spikes, ims=spike_ims, axes=spike_axes
                )
                weights1_im = plot_conv2d_weights(weights1, im=weights1_im)
                voltage_ims, voltage_axes = plot_voltages(
                    _voltages, ims=voltage_ims, axes=voltage_axes
                )

            plt.pause(1e-8)

        network.reset_()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")
