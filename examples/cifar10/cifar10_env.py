import os
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt

from time import time as t

from bindsnet.datasets import CIFAR10
from bindsnet.encoding import poisson
from bindsnet.pipeline import Pipeline
from bindsnet.models import DiehlAndCook2015
from bindsnet.network.monitors import Monitor
from bindsnet.environment import DatasetEnvironment
from bindsnet.utils import get_square_assignments, get_square_weights
from bindsnet.evaluation import all_activity, proportion_weighting, assign_labels
from bindsnet.analysis.plotting import plot_input, plot_assignments, plot_performance, plot_weights

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--n_train', type=int, default=50000)
parser.add_argument('--n_test', type=int, default=10000)
parser.add_argument('--exc', type=float, default=22.5)
parser.add_argument('--inh', type=float, default=17.5)
parser.add_argument('--time', type=int, default=350)
parser.add_argument('--dt', type=int, default=1.0)
parser.add_argument('--intensity', type=float, default=0.25)
parser.add_argument('--progress_interval', type=int, default=10)
parser.add_argument('--update_interval', type=int, default=250)
parser.add_argument('--train', dest='train', action='store_true')
parser.add_argument('--test', dest='train', action='store_false')
parser.add_argument('--plot', dest='plot', action='store_true')
parser.add_argument('--gpu', dest='gpu', action='store_true')
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

if gpu and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

if not train:
    update_interval = n_test

n_sqrt = int(np.ceil(np.sqrt(n_neurons)))
path = os.path.join('..', '..', 'data', 'CIFAR10')
    
# Build network.
network = DiehlAndCook2015(n_inpt=32*32*3, n_neurons=n_neurons, exc=exc, inh=inh, dt=dt, nu_pre=2e-5, nu_post=2e-3,
                           norm=10.0)

# Initialize data "environment".
environment = DatasetEnvironment(dataset=CIFAR10(path=path, download=True), train=train, time=time, intensity=intensity)

# Specify data encoding.
encoding = poisson

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=['s'], time=time)
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)

voltages = {}
for layer in set(network.layers) - {'X'}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=['v'], time=time)
    network.add_monitor(voltages[layer], name='%s_voltages' % layer)

# Build pipeline from above-specified components.
pipeline = Pipeline(network=network,
                    environment=environment,
                    encoding=encoding,
                    time=time,
                    plot_interval=1)

# Neuron assignments and spike proportions.
assignments = -torch.ones_like(torch.Tensor(n_neurons))
proportions = torch.zeros_like(torch.Tensor(n_neurons, 10))
rates = torch.zeros_like(torch.Tensor(n_neurons, 10))

# Record spikes during the simulation.
spike_record = torch.zeros(update_interval, time, n_neurons)

# Get data labels.
labels = pipeline.env.labels

# Sequence of accuracy estimates.
accuracy = {'all' : [], 'proportion' : []}

# Image categories.
classes = ['none', 'plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

# Train the network.
print('Begin training.\n')
start = t()

for i in range(n_train):
    if i % progress_interval == 0:
        print('Progress: %d / %d (%.4f seconds)' % (i, n_train, t() - start))
        start = t()
    
    if i % update_interval == 0 and i > 0:
        # Get network predictions.
        all_activity_pred = all_activity(spike_record, assignments, 10)
        proportion_pred = proportion_weighting(spike_record, assignments, proportions, 10)

        # Compute network accuracy according to available classification strategies.
        accuracy['all'].append(100 * torch.sum(labels[i - update_interval:i].long() == all_activity_pred) / update_interval)
        accuracy['proportion'].append(100 * torch.sum(labels[i - update_interval:i].long() == proportion_pred) / update_interval)

        print('\nAll activity accuracy: %.2f (last), %.2f (average), %.2f (best)' \
                        % (accuracy['all'][-1], np.mean(accuracy['all']), np.max(accuracy['all'])))
        print('Proportion weighting accuracy: %.2f (last), %.2f (average), %.2f (best)\n' \
                        % (accuracy['proportion'][-1], np.mean(accuracy['proportion']),
                          np.max(accuracy['proportion'])))

        # Assign labels to excitatory layer neurons.
        assignments, proportions, rates = assign_labels(spike_record, labels[i - update_interval:i], 10, rates)
    
    pipeline.step()
    
    # Add to spikes recording.
    spike_record[i % update_interval] = spikes['Ae'].get('s').t()
    
    # Optionally plot various simulation info.
    if plot:
        if gpu:
            image = pipeline.obs.view(3, 32, 32).cpu().numpy().transpose(1, 2, 0) / intensity
            image /= image.max()
            inpt = 255 - pipeline.encoded['X'].view(time, 3*32*32).sum(0).view(3, 32, 32).sum(0).cpu()
            weights = network.connections[('X', 'Ae')].w.view(3, 32, 32, n_neurons).cpu().numpy()
        else:
            image = pipeline.obs.view(3, 32, 32).numpy().transpose(1, 2, 0) / intensity
            image /= image.max()
            inpt = 255 - pipeline.encoded['X'].view(time, 3*32*32).sum(0).view(3, 32, 32).sum(0)
            weights = network.connections[('X', 'Ae')].w.view(3, 32, 32, n_neurons).numpy()
                
        weights = weights.transpose(1, 2, 0, 3).sum(2).reshape(32*32, n_neurons)
        weights = torch.from_numpy(weights)
            
        square_assignments = get_square_assignments(assignments, n_sqrt)
        square_weights = get_square_weights(weights, n_sqrt, 32)
        
        if i == 0:
            inpt_axes, inpt_ims = plot_input(image, inpt, label=labels[i])
            assigns_im = plot_assignments(square_assignments, classes=classes)
            perf_ax = plot_performance(accuracy)
            weights_ax = plot_weights(square_weights, wmin=0.0, wmax=0.025)
        else:
            inpt_axes, inpt_ims = plot_input(image, inpt, label=labels[i], axes=inpt_axes, ims=inpt_ims)
            assigns_im = plot_assignments(square_assignments, im=assigns_im)
            perf_ax = plot_performance(accuracy, ax=perf_ax)
            weights_im = plot_weights(square_weights, im=weights_ax)
        
        plt.pause(1e-8)
    
    network.reset_()  # Reset state variables.

print('Progress: %d / %d (%.4f seconds)\n' % (n_train, n_train, t() - start))
print('Training complete.\n')
