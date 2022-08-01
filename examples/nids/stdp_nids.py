import argparse
import os
import random
from time import time as t
from os.path import exists

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm
from torch.utils.data import Dataset
from bindsnet import ROOT_DIR
from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
import seaborn as sns
sns.set_theme(style="darkgrid")
import pandas as pd

import matplotlib
from bindsnet.datasets import MNIST, DataLoader
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from torch.utils.data import DataLoader as dl
parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=400)
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=20000)
parser.add_argument("--n_train", type=int, default=300000/4)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--update_steps", type=int, default=10)  #Modify along with batch size
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=256)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
batch_size = args.batch_size
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
n_workers = args.n_workers
update_steps = args.update_steps
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
train = args.train
plot = args.plot
gpu = args.gpu
target_batch_accuracy = 60
target_count = 50
update_interval = update_steps * batch_size

device = "cpu"
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 1 * torch.cuda.device_count()

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

class NidsDatasetTrain(Dataset):
    """Nids dataset."""

    def __init__(self):
        x = np.load('training_data1.npy', allow_pickle=True)
        xtrain = x.item(0)['trainx']
        ytrain = x.item(0)['trainy']
        self.x_data = torch.clamp(torch.from_numpy(xtrain),0.0, 1.0)*256
        self.y_data = torch.from_numpy(ytrain)
        self.len = torch.from_numpy(xtrain).shape[0]

    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index].view(1,25), self.y_data[index]

class NidsDatasetTest(Dataset):
    """Nids dataset."""

    def __init__(self):
        x = np.load('training_data1.npy',allow_pickle=True)
        xtest = x.item(0)['testx']
        ytest = x.item(0)['testy']
        self.x_data = torch.clamp(torch.from_numpy(xtest),0.0, 1.0)*256
        self.y_data = torch.from_numpy(ytest)
        self.len = torch.from_numpy(xtest).shape[0]
    def __len__(self):
        return self.len

    def __getitem__(self, index):
        return self.x_data[index].view(1,25), self.y_data[index]

# Build network.
network = DiehlAndCook2015(
    n_inpt=25,
    n_neurons=n_neurons,
    exc=exc,
    inh=inh,
    norm=62.5,
    wmin=0.0,
    wmax=25,
    dt=dt,
    nu=(1e-4, 1e-2),
    theta_plus=theta_plus,
    inpt_shape=(1,25),
)

# Directs network to GPU
if gpu:
    network.to("cuda")

# Load NIDS
datasetTrain = NidsDatasetTrain()

# Neuron assignments and spike proportions.
n_classes = 9
assignments = -torch.ones(n_neurons, device=device)
proportions = torch.zeros((n_neurons, n_classes), device=device)
rates = torch.zeros((n_neurons, n_classes), device=device)

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["Ae"], ["v"], time=int(time / dt), device=device
)
inh_voltage_monitor = Monitor(
    network.layers["Ai"], ["v"], time=int(time / dt), device=device
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

# Set up monitors for spikes and voltages
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(
        network.layers[layer], state_vars=["v"], time=int(time / dt), device=device
    )
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None

spike_record = torch.zeros((update_interval, int(time / dt), n_neurons), device=device)

# Train the network.
print("\nBegin training.\n")
start = t()
flag = 0
for epoch in range(n_epochs):
    labels = []
    if flag:
        break
    if epoch % progress_interval == 0:
        print("\n Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    train_loader = dl(dataset=datasetTrain, batch_size=args.batch_size, shuffle=True, num_workers=0)

    pbar_training = tqdm(total=n_train*args.batch_size)
    fifty_count = 0
    for step, batch in enumerate(train_loader):
        if step > n_train:
            break
        # Get next input sample.
        inputs = {"X": PoissonEncoder(time=time, dt=dt)(batch[0])}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}

        if step % update_steps == 0 and step > 0:
            # Convert the array of labels into a tensor
            label_tensor = torch.tensor(labels, device=device)

            # Get network predictions.
            all_activity_pred = all_activity(
                spikes=spike_record, assignments=assignments, n_labels=n_classes
            )
            proportion_pred = proportion_weighting(
                spikes=spike_record,
                assignments=assignments,
                proportions=proportions,
                n_labels=n_classes,
            )

            # Compute network accuracy according to available classification strategies.
            val = 100 * torch.sum(label_tensor.long() == all_activity_pred).item() / len(label_tensor)
            accuracy["all"].append(
                val
            )
            if val >= target_batch_accuracy:
                fifty_count+=1
                print('Times above target accuracy: ', fifty_count)
                if fifty_count == target_count:
                    flag = 1
                    break

            print("\nAll activity accuracy: %.2f" % (torch.sum(label_tensor.long() == all_activity_pred).item()))
            accuracy["proportion"].append(
                100
                * torch.sum(label_tensor.long() == proportion_pred).item()
                / len(label_tensor)
            )

            print(
                "\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)"
                % (
                    accuracy["all"][-1],
                    np.mean(accuracy["all"]),
                    np.max(accuracy["all"]),
                )
            )
            matplotlib.rcParams['font.family'] = "Arial"
            print(
                "Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f"
                " (best)\n"
                % (
                    accuracy["proportion"][-1],
                    np.mean(accuracy["proportion"]),
                    np.max(accuracy["proportion"]),
                )
            )
            n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
            square_weights = get_square_weights(
                network.connections["X", "Ae"].w.view(25, n_neurons),
                n_sqrt,
                5,
            )
            # Assign labels to excitatory layer neurons.
            assignments, proportions, rates = assign_labels(
                spikes=spike_record,
                labels=label_tensor,
                n_labels=n_classes,
                rates=rates,
            )

            labels = []

        labels.extend((batch[1].argmax(dim=1)).tolist())

        # Run the network on the input.
        network.run(inputs=inputs, time=time, input_time_dim=1)

        # Add to spikes recording.
        s = spikes["Ae"].get("s").permute((1, 0, 2))
        spike_record[
            (step * batch_size)
            % update_interval : (step * batch_size % update_interval)
            + s.size(0)
        ] = s

        # Get voltage recording.
        exc_voltages = exc_voltage_monitor.get("v")
        inh_voltages = inh_voltage_monitor.get("v")

        network.reset_state_variables()  # Reset state variables.
        pbar_training.update(batch_size)

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Training complete.\n")

datasetTest = NidsDatasetTest()

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Train the network.
print("\nBegin testing\n")
network.train(mode=False)
start = t()

pbar = tqdm(total=n_test)
test_batch_size = 8
test_loader = dl(dataset=datasetTest, batch_size=test_batch_size, shuffle=True, num_workers=0)

label_list = []
prediction_list = []
for step, batch in enumerate(test_loader):
    if step > n_test:
        break
    # Get next input sample.
    inputs = {"X": PoissonEncoder(time=time, dt=dt)(batch[0])}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time, input_time_dim=1)

    # Add to spikes recording.
    spike_record = spikes["Ae"].get("s").permute((1, 0, 2))

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor((batch[1].argmax(dim=1)), device=device)

    # Get network predictions.
    all_activity_pred = all_activity(
        spikes=spike_record, assignments=assignments, n_labels=n_classes
    )
    proportion_pred = proportion_weighting(
        spikes=spike_record,
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes,
    )

    # Compute network accuracy according to available classification strategies.
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    label_list = label_list + (label_tensor.long()).tolist()
    prediction_list = prediction_list + all_activity_pred.tolist()
    accuracy["proportion"] += float(
        torch.sum(label_tensor.long() == proportion_pred).item()
    )
    print("All activity accuracy: %.2f" % (accuracy["all"]/(test_batch_size*(step+1))))
    print("Proportion accuracy: %.2f" % (accuracy["proportion"] / (test_batch_size * (step + 1))))
    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

# x = np.asarray(prediction_list)
# np.save('/home/opc/workspace/project1/SNN-Conversion/prediction_list.npy',x)
#
# y = np.asarray(label_list)
# np.save('/home/opc/workspace/project1/SNN-Conversion/label_list.npy',y)

print("\nAll activity accuracy: %.2f" % (accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (accuracy["proportion"] / n_test))

print("Progress: %d / %d (%.4f seconds)" % (epoch + 1, n_epochs, t() - start))
print("Testing complete.\n")
