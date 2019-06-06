import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from time import time as t

from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.encoding import poisson_loader
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_weights, get_square_assignments
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_weights,
    plot_assignments,
    plot_performance,
    plot_voltages,
)

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=50.0)
parser.add_argument("--time", type=int, default=350)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=0.5)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_train = args.n_train
n_test = args.n_test
exc = args.exc
inh = args.inh
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

device = torch.device("cpu")

if gpu:
    # try:
    #    if not torch.cuda.is_available():
    #        raise Exception("Cuda Unavailable")
    # torch.set_default_tensor_type("torch.cuda.FloatTensor")
    torch.cuda.manual_seed_all(seed)
    # torch.cuda.set_device("cuda")
    # except Exception:
    #    print("GPU not available for use, check CUDA or run without --gpu")
    #    exit
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Build network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    dt=dt,
    norm=78.4,
    theta_plus=1,
    inpt_shape=(1, 1, 28, 28),
)

# Directs network to GPU

if gpu:
    network.to("cuda")

# Load MNIST data.

dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * intensity),
            # transforms.Lambda(lambda x: x.view(784)),
        ]
    ),
)

# Create a dataloader to iterate and batch data

dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=4, pin_memory=gpu
)

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons))
proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
rates = torch.zeros_like(torch.Tensor(n_neurons, 10))

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(network.layers["Ae"], ["v"], time=time)
inh_voltage_monitor = Monitor(network.layers["Ai"], ["v"], time=time)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

# Train the network.
print("\nBegin training.\n")
start = t()

# This is needed for my implementation, but I want to reformat this away
labels = []

for step, batch in enumerate(tqdm(dataloader)):

    # sample = dataset[step]
    if step > n_train:
        break

    # Get next input sample.
    inpts = {"X": batch["encoded_image"]}
    if gpu:
        inpts = {k: v.cuda() for k, v in inpts.items()}
    labels.append(batch["label"])

    #    if step % progress_interval == 0:
    #        print("Progress: %d / %d (%.4f seconds)" % (step, n_train, t() - start))
    #        start = t()

    if step % update_interval == 0 and step > 0:

        # Convert the array of labels into a tensor
        label_tensor = torch.tensor(labels)

        # Get network predictions.
        all_activity_pred = all_activity(spike_record, assignments, 10)
        proportion_pred = proportion_weighting(
            spike_record, assignments, proportions, 10
        )

        # Compute network accuracy according to available classification strategies.
        accuracy["all"].append(
            100
            * torch.sum(
                label_tensor[step - update_interval : step].long() == all_activity_pred
            ).item()
            / update_interval
        )
        accuracy["proportion"].append(
            100
            * torch.sum(
                label_tensor[step - update_interval : step].long() == proportion_pred
            ).item()
            / update_interval
        )

        print(
            "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
            % (accuracy["all"][-1], np.mean(accuracy["all"]), np.max(accuracy["all"]))
        )
        print(
            "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n"
            % (
                accuracy["proportion"][-1],
                np.mean(accuracy["proportion"]),
                np.max(accuracy["proportion"]),
            )
        )

        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(
            spike_record, label_tensor[step - update_interval : step], 10, rates
        )

    # Run the network on the input.
    network.run(inpts=inpts, time=time, input_time_dim=1)

    # Get voltage recording.
    exc_voltages = exc_voltage_monitor.get("v")
    inh_voltages = inh_voltage_monitor.get("v")

    # Add to spikes recording.
    spike_record[step % update_interval] = spikes["Ae"].get("s").t()

    # Optionally plot various simulation information.
    if plot:
        inpt = inpts["X"].view(time, 784).sum(0).view(28, 28)
        input_exc_weights = network.connections[("X", "Ae")].w
        square_weights = get_square_weights(
            input_exc_weights.view(784, n_neurons), n_sqrt, 28
        )
        square_assignments = get_square_assignments(assignments, n_sqrt)
        spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
        voltages = {"Ae": exc_voltages, "Ai": inh_voltages}

        inpt_axes, inpt_ims = plot_input(
            images[step].view(28, 28),
            inpt,
            label=labels[step],
            axes=inpt_axes,
            ims=inpt_ims,
        )
        spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
        weights_im = plot_weights(square_weights, im=weights_im)
        assigns_im = plot_assignments(square_assignments, im=assigns_im)
        perf_ax = plot_performance(accuracy, ax=perf_ax)
        voltage_ims, voltage_axes = plot_voltages(
            voltages, ims=voltage_ims, axes=voltage_axes, plot_type="line"
        )

        plt.pause(1e-8)

    network.reset_()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_train, n_train, t() - start))
print("Training complete.\n")
