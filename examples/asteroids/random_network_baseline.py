import torch
import argparse

from bindsnet.network import Network
from bindsnet.learning import Hebbian
from bindsnet.pipeline import Pipeline
from bindsnet.encoding import bernoulli
from bindsnet.network.monitors import Monitor
from bindsnet.environment import GymEnvironment
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_multinomial

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=1000000)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--n_neurons', type=int, default=100)
parser.add_argument('--dt', type=float, default=1.0)
parser.add_argument('--plot_interval', type=int, default=10)
parser.add_argument('--render_interval', type=int, default=10)
parser.add_argument('--print_interval', type=int, default=100)
parser.add_argument('--gpu', dest='gpu', action='store_true')
parser.set_defaults(plot=False, render=False, gpu=False)

args = parser.parse_args()

n = args.n
seed = args.seed
n_neurons = args.n_neurons
dt = args.dt
plot_interval = args.plot_interval
render_interval = args.render_interval
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
inpt = Input(shape=(110, 84), traces=True)  # Input layer
exc = LIFNodes(n=n_neurons, refrac=0, traces=True)  # Excitatory layer
readout = LIFNodes(n=14, refrac=0, traces=True)  # Readout layer
layers = {'X' : inpt, 'E' : exc, 'R' : readout}

# Connections between layers.
# Input -> excitatory.
w = 0.01 * torch.rand(layers['X'].n, layers['E'].n)
input_exc_conn = Connection(source=layers['X'], target=layers['E'], w=0.01 * torch.rand(layers['X'].n, layers['E'].n),
                            wmax=0.02, norm=0.01 * layers['X'].n)

# Excitatory -> readout.
exc_readout_conn = Connection(source=layers['E'], target=layers['R'], w=0.01 * torch.rand(layers['E'].n, layers['R'].n),
                              update_rule=Hebbian, nu_pre=1e-2, nu_post=1e-2, norm=0.5 * layers['E'].n)

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
environment = GymEnvironment('Asteroids-v0')
environment.reset()

pipeline = Pipeline(network, environment, encoding=bernoulli, time=1, history=5, delta=10, plot_interval=plot_interval,
                    print_interval=print_interval, render_interval=render_interval, action_function=select_multinomial,
                    output='R')

total = 0
rewards = []
avg_rewards = []
lengths = []
avg_lengths = []

i = 0
try:
    while i < n:
        pipeline.step()
        
        if pipeline.done:
            pipeline.reset_()

        i += 1
        
except KeyboardInterrupt:
    environment.close()
