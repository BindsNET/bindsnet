import torch
import pdb

from bindsnet.network import Network
from bindsnet.pipeline import Pipeline
from bindsnet.learning import MSTDPET
from bindsnet.encoding import bernoulli
from bindsnet.network.topology import Connection
from bindsnet.environment import GymEnvironment
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_multinomial

# TODO : Build critic network.


# Build network.
network = Network(dt=1.0)

# Layers of neurons.
# TODO : Check why input neurons fire only at the first time.
inpt = Input(n=1250, shape=[1250], traces=True)
middle = LIFNodes(n=225, traces=True, thresh=1.0, rest=0.0, reset=0.0, refrac=0,
                  decay=0.05)
out = LIFNodes(n=60, refrac=0, traces=True, thresh=1.0, rest=0.0, reset=0.0)

# Connections between layers.
# TODO : Understand current initializing process.
inpt_middle = Connection(source=inpt, target=middle, wmin=-0.1,wmax=0.5)
middle_out = Connection(source=middle, target=out, wmax=0.5,
                        update_rule=MSTDPET, nu=2e-2,
                        norm=0.15 * middle.n)

# Add all layers and connections to the network.
network.add_layer(inpt, name='X')
network.add_layer(middle, name='Y')
network.add_layer(out, name='Z')
network.add_connection(inpt_middle, source='X', target='Y')
network.add_connection(middle_out, source='Y', target='Z')

# Load SpaceInvaders environment.
environment = GymEnvironment('CartPole-v0')
environment.reset()

# Build pipeline from specified components.
pipeline = Pipeline(network, environment, encoding=bernoulli,
                    action_function=select_multinomial,output='Z',
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