
import argparse
import os
from time import time as t

import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_assignments,
    plot_input,
    plot_performance,
    plot_spikes,
    plot_voltages,
    plot_weights,
    plot_traces,  # added
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import Salah_model
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network import Network, load

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=1000)
parser.add_argument("--n_epochs", type=int, default=1)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_train", type=int, default=60000)
parser.add_argument("--exc", type=float, default=22.5)
parser.add_argument("--inh", type=float, default=120)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--train", dest="train", action="store_true")
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.add_argument("--saved_as")
parser.set_defaults(plot=True, gpu=False, train="False", n_test=10000 ) 
args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_test = args.n_test
n_train = args.n_train
exc = args.exc
inh = args.inh
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu
saved_as = args.saved_as # "exp_26"

#================================================
# Sets up Gpu use (not used)
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

#================================================

n_classes = 10
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Load pre-trained network
network = load(f"net_{saved_as}.pt")
network.train(mode=False)

if gpu:
    network.to("cuda")

# Load assignments, obtained while training
train_details = torch.load(f"./details_{saved_as}.pt", map_location=torch.device(device))

# Assign the variables from the loaded dictionary
assignments = train_details["assignments"]
proportions = train_details["proportions"]
rates = train_details["rates"]

#================================================
spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(
        network.layers[layer], state_vars=["s"], time=int(time / dt), device=device 
    )
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

inpt_ims, inpt_axes = None, None
spike_ims, spike_axes = None, None
weights_im = None
assigns_im = None
perf_ax = None
voltage_axes, voltage_ims = None, None
trace_axes, trace_ims = None, None

#================================================
# Load MNIST data.
test_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    train=False,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),)

# Sequence of accuracy estimates.
accuracy = {"all": 0, "proportion": 0}

# Record spikes during the simulation.
spike_record = torch.zeros((1, int(time / dt), n_neurons), device=device)


#------- start part one -------

#==================================================================================
#*************************  Testing the network ***********************************

# Initialize dictionary to store spikes for analysis
spike_history = {}

'''
# Start testing
print("\nBegin testing\n")
start = t()
pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break

    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if device == 'cuda':
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input
    network.run(inputs=inputs, time=time)

    # Record spikes for all layers at each step
    spike_history[step] = {layer: spikes[layer].get("s").clone().detach() for layer in spikes}

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

    # Get network predictions using spikes from the 'Ae' layer for simplicity
    all_activity_pred = all_activity(
        spikes=spike_history[step]['Ae'],  # Updated to use spike_history
        assignments=assignments,
        n_labels=n_classes
    )

    proportion_pred = proportion_weighting(
        spikes=spike_history[step]['Ae'],  # Updated to use spike_history
        assignments=assignments,
        proportions=proportions,
        n_labels=n_classes
    )

    # Compute network accuracy according to available classification strategies
    accuracy["all"] += float(torch.sum(label_tensor.long() == all_activity_pred).item())
    accuracy["proportion"] += float(torch.sum(label_tensor.long() == proportion_pred).item())

# plot the spikes of all layers while testing  
    if step % update_interval == 0 and step > 0:
        if plot:
            spikes_ = {layer: spikes[layer].get("s") for layer in spikes}
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            plt.pause(1e-8)
#----------
    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

# Save the spike history to a .pt file for later analysis
torch.save(spike_history, "spike_history.pt")

pbar.close()
print("\nAll activity accuracy: %.2f \n" % (100 * accuracy["all"] / n_test))
print("Proportion weighting accuracy: %.2f \n" % (100 * accuracy["proportion"] / n_test))
print("Testing complete after %.4f seconds \n" % (t() - start))

#-----------------------------
'''
#------- end part one -------


#------- start part two -------

# Uncomment below lines to load and plot spikes after testing

spike_history = torch.load("spike_history.pt")
# Function to plot spikes from 'Ae' layer
def plot_spikes_for_interval(start_step, end_step, layer_name="Ae"):
    spikes_to_plot = {layer_name: torch.cat([spike_history[step][layer_name] for step in range(start_step, end_step)], dim=0)}
    _, axes = plot_spikes(spikes=spikes_to_plot, figsize=(12, 6))
    plt.show()

# Example usage: plot spikes for Ae from steps 0 to update_interval
# Here, each step is an image, each image will last for time/dt points horizontally
#plot_spikes_for_interval(0, update_interval, "Ae")
plot_spikes_for_interval(0, 10000, "X")

#step(image) - layer - 
#spike_history[1]["Ae"][1][:])
plt.pause(300)
#------- end part two -------
'''
Notes : 

The spike_history file is a dictionary structured to store detailed spiking activity data from a neural network simulation, covering 10,000 steps, likely corresponding to 10,000 distinct image presentations. Each entry in the dictionary is keyed by a step number (from 0 to 9999) and contains another dictionary mapping layer names to tensors representing the spike data.

For each layer at each step, the spike data is stored in a tensor with specific dimensions and types:

Layers: The data includes layers labeled as "Ai", "X", and "Ae".
Shape of the Tensors: For "Ai" and "Ae" layers, the tensors have a shape of [250, 1, 100], indicating 250 time steps, with spikes recorded from 100 neurons (the middle dimension is likely a singleton dimension added for batch processing or other architectural reasons). For the "X" layer, the shape is [250, 1, 1, 28, 28], suggesting a different data structure possibly representing input or processed images over time.
Data Types: The tensors in the "Ai" and "Ae" layers are of type torch.bool, indicating binary spiking data (true for a spike, false for no spike). The tensors in the "X" layer are torch.uint8, which might be representing grayscale pixel values of images.
Storage: All data is stored on the CPU as indicated by the tensor device information.

print('value  ', spike_history[1]["Ae"][10,0,50])
'''
