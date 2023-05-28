
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
n_sqrt = int(np.ceil(np.sqrt(n_neurons)))

#load pre-trained network 
my_network = load("net_24_5.pt") 

#extract weights
input_exc_weights = my_network.connections[("X", "Ae")].w
square_weights = get_square_weights( input_exc_weights.view(784, n_neurons), n_sqrt, 28 )

spikes = {}
for layer in set(my_network.layers):
    spikes[layer] = Monitor(
        my_network.layers[layer], state_vars=["s"], time=int(time / dt), device="cpu" 
    )
    my_network.add_monitor(spikes[layer], name="%s_spikes" % layer)



print(dir(my_network))
#weights_im = plot_weights(square_weights, im=None)
#plt.pause(400)



