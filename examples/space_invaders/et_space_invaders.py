import torch
import argparse
import matplotlib.pyplot as plt

from bindsnet.encoding import bernoulli
from bindsnet.environment import GymEnvironment
from bindsnet.learning import MSTDPET
from bindsnet.network import Network, Input
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import Pipeline
from bindsnet.pipeline.action import select_multinomial

parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--a_plus', type=int, default=1)
parser.add_argument('--a_minus', type=int, default=-0.5)
parser.add_argument('--render_interval', type=int, default=None)
parser.add_argument('--plot_interval', type=int, default=None)
parser.add_argument('--print_interval', type=int, default=None)
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, render=False, gpu=False)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
dt = args.dt
a_plus = args.a_plus
a_minus = args.a_minus
render_interval = args.render_interval
plot_interval = args.plot_interval
print_interval = args.print_interval
gpu = args.gpu

if gpu:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)

# Build network.
network = Network(dt=dt)

# Layers of neurons.
inpt = Input(n=6552, shape=[78, 84], traces=True)  # Input layer
exc = LIFNodes(n=n_neurons, refrac=0, traces=True, thresh=-52.0 + torch.randn(n_neurons))  # Excitatory layer
readout = LIFNodes(n=60, refrac=0, traces=True, thresh=-40.0)  # Readout layer
layers = {'X': inpt, 'E': exc, 'R': readout}

# Connections between layers.
# Input -> excitatory.
input_exc_conn = Connection(source=layers['X'], target=layers['E'],
                            w=torch.rand(layers['X'].n, layers['E'].n), wmax=1e-2)

# Excitatory -> readout.
exc_readout_conn = Connection(source=layers['E'], target=layers['R'], w=torch.rand(layers['E'].n, layers['R'].n),
                              wmax=0.5, update_rule=MSTDPET, nu=2e-2, norm=0.15 * layers['E'].n)

# Spike recordings for all layers.
spikes = {}
for layer in layers:
    spikes[layer] = Monitor(layers[layer], ['s'], time=plot_interval)

# Voltage recordings for excitatory and readout layers.
voltages = {}
for layer in set(layers.keys()) - {'X'}:
    voltages[layer] = Monitor(layers[layer], ['v'], time=plot_interval)

# Add all layers and connections to the network.
for layer in layers:
    network.add_layer(layers[layer], name=layer)

network.add_connection(input_exc_conn, source='X', target='E')
network.add_connection(exc_readout_conn, source='E', target='R')

# Add all monitors to the network.
for layer in layers:
    network.add_monitor(spikes[layer], name='%s_spikes' % layer)

    if layer in voltages:
        network.add_monitor(voltages[layer], name='%s_voltages' % layer)

# Load SpaceInvaders environment.
environment = GymEnvironment('SpaceInvaders-v0')
environment.reset()

pipeline = Pipeline(network, environment, encoding=bernoulli, time=1, history_length=5, delta=10,
                    plot_interval=plot_interval, print_interval=print_interval, render_interval=render_interval,
                    feedback=select_multinomial, output='R')

try:
    while True:
        pipeline.step()

        if pipeline.done:
            pipeline.reset_()
except KeyboardInterrupt:
    plt.close("all")
    environment.close()
