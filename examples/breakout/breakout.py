import torch

from bindsnet.network import Network
from bindsnet.pipeline import Pipeline
from bindsnet.encoding import bernoulli
from bindsnet.network.topology import Connection
from bindsnet.environment import GymEnvironment
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_softmax

# Build network.
network = Network(dt=1.0)

# Layers of neurons.
inpt = Input(n=80 * 80, shape=[80, 80], traces=True)
middle = LIFNodes(n=100, traces=True)
out = LIFNodes(n=4, refrac=0, traces=True)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmax=1e-2)
middle_out = Connection(source=middle, target=out, wmax=1e-1, nu=2e-2)

# Add all layers and connections to the network.
network.add_layer(inpt, name='X')
network.add_layer(middle, name='Y')
network.add_layer(out, name='Z')
network.add_connection(inpt_middle, source='X', target='Y')
network.add_connection(middle_out, source='Y', target='Z')

# Load SpaceInvaders environment.
environment = GymEnvironment('BreakoutDeterministic-v4')
environment.reset()

# Build pipeline from specified components.
pipeline = Pipeline(network, environment, encoding=bernoulli,
                    action_function=select_softmax, output='Z',
                    time=100, history_length=1, delta=1,
                    plot_interval=1, render_interval=1)

# Run environment simulation for 100 episodes.
for i in range(100):
    # initialize episode reward
    reward = 0
    pipeline.reset_()
    while True:
        pipeline.step()
        reward += pipeline.reward
        if pipeline.done:
            break
    print("Episode " + str(i) + " reward:", reward)
