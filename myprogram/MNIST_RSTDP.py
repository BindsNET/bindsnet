import bindsnet

import pdb

from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.pipeline import Pipeline
from bindsnet.learning import MSTDPET
from bindsnet.encoding import bernoulli
from bindsnet.network.topology import Connection
from bindsnet.environment import DatasetEnvironment
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_multinomial

# Build network.
network = Network(dt=1.0)

# Layers of neurons.
input = Input(n=78 * 84, shape=[78, 84], traces=True)
hidden = LIFNodes(n=225, traces=True, thresh=1.0)
output = LIFNodes(n=50, refrac=0, traces=True, thresh=-40.0)

# Connections between layers.
# TODO Validate RSTDP
in_hidden = Connection(source=input, target=hidden, wmax=1e-2)
hidden_out = Connection(source=hidden, target=out, wmax=0.5)

# Add all layers and connections to the network.
network.add_layer(input, name='IN')
network.add_layer(hidden, name='HIDDEN')
network.add_layer(output, name='OUT')
network.add_connection(in_hidden, source='IN', target='HIDDEN')
network.add_connection(hidden_out, source='HIDDEN', target='OUT')

environment = DatasetEnvironment(dataset=MNIST(path='../../data/MNIST',
                                 download=True), train=True)
environment = MNISTEnvironment(duration_mean=100.0, duration_uc=10.0,
                               delay_mean=100.0, delay_uc=10.0)
# Build pipeline from specified components.
# TODO First spike encoding.
# TODO Delayed reward with uncertain timing.
pipeline = Pipeline(network, environment, encoding=bernoulli,
                    action_function=select_multinomial, output='OUT',
                    time=1, history_length=2, delta=4,
                    plot_interval=100, render_interval=5)

# Run environment simulation and network training.
n_episode = 0
n_step = 0
while True:
    pipeline.step()
    n_step += 1
    print('{}th episode\'s {}th step'.format(n_episode,n_step))
    if pipeline.done:
        n_episode += 1
        n_step = 0
        pipeline.reset_()
