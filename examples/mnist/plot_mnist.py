
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
from bindsnet.evaluation import all_activity, assign_labels, proportion_weighting
from bindsnet.models import Salah_model
from bindsnet.network.monitors import Monitor
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.network import Network, load

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_test", type=int, default=10000)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--theta_plus", type=float, default=0.05)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--test", dest="train", action="store_false")
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=True, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_test = args.n_test
n_workers = args.n_workers
theta_plus = args.theta_plus
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
plot = args.plot
gpu = args.gpu
'''
# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
'''
device = "cpu"
# Determines number of workers to use
if n_workers == -1:
    n_workers = 0  # gpu * 4 * torch.cuda.device_count()

update_interval = n_test
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
start_intensity = intensity

# Sequence of accuracy estimates.
accuracy = {"all": [], "proportion": []}

my_network = load("net_24_5.pt", map_location = device ,learning =None)


input_exc_weights = my_network.connections[("X", "Ae")].w
square_weights = get_square_weights( input_exc_weights.view(784, n_neurons), n_sqrt, 28 )

#print(dir(network.connections[("X", "Ae")])) 

#weights_im = plot_weights(square_weights, im=None)
local_weights = square_weights.detach().clone().cpu().numpy()

fig, ax = plt.subplots(figsize=(15,10))
im = ax.imshow(local_weights)#, cmap=cmap, vmin=wmin, vmax=wmax)
plt.show()
#plot_weights(square_weights)
