
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
parser.add_argument("--n_neurons", type=int, default=100)
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
saved_as = "exp_10"

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

#==================================================================================
#*************************  Testing the network ***********************************

print("\nBegin testing\n")
start = t()
'''
pbar = tqdm(total=n_test)
for step, batch in enumerate(test_dataset):
    if step >= n_test:
        break
    # Get next input sample.
    inputs = {"X": batch["encoded_image"].view(int(time / dt), 1, 1, 28, 28)}
    if gpu:
        inputs = {k: v.cuda() for k, v in inputs.items()}

    # Run the network on the input.
    network.run(inputs=inputs, time=time)

    # Add to spikes recording.
    spike_record[0] = spikes["Ae"].get("s").squeeze()
    #spike_record[0] = spikes

    # Convert the array of labels into a tensor
    label_tensor = torch.tensor(batch["label"], device=device)

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
    accuracy["proportion"] += float(torch.sum(label_tensor.long() == proportion_pred).item())

    network.reset_state_variables()  # Reset state variables.
    pbar.set_description_str("Test progress: ")
    pbar.update()

print("\nAll activity accuracy:         %.2f \n" % (100*accuracy["all"] / n_test))
print("Proportion weighting accuracy:   %.2f \n" % (100*accuracy["proportion"] / n_test))
print("Testing complete after:          %.4f seconds \n" % (t() - start))
'''
#==================================================================================
#**********************************  Ploting **************************************

print("training time", train_details['train_time'])
print(train_details["train_accur"])

#extract weights
input_exc_weights = network.connections[("X", "Ae")].w
square_weights = get_square_weights( input_exc_weights.view(784, n_neurons), n_sqrt, 28 )
square_assignments = get_square_assignments(assignments, n_sqrt)
train_accur = train_details["train_accur"]

#plot
weights_im = plot_weights(square_weights, im=None)
assigns_im = plot_assignments(square_assignments, im=None)
perf_ax = plot_performance(train_accur, x_scale=update_interval, ax=None)
plt.pause(300)




