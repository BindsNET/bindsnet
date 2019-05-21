import torch

from bindsnet.network import Network
from bindsnet.pipeline import Pipeline
from bindsnet.learning import MSTDP
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
inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1e-1)
middle_out = Connection(
    source=middle,
    target=out,
    wmin=0,
    wmax=1,
    update_rule=MSTDP,
    nu=1e-1,
    norm=0.5 * middle.n,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(out, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load SpaceInvaders environment.
environment = GymEnvironment("BreakoutDeterministic-v4")
environment.reset()

# Build pipeline from specified components.
pipeline = Pipeline(
    network,
    environment,
    encoding=bernoulli,
    action_function=select_softmax,
    output="Output Layer",
    time=100,
    history_length=1,
    delta=1,
    plot_interval=1,
    render_interval=1,
)


# Train agent for 100 episodes.
print("Training: ")
for i in range(100):
    pipeline.reset_()
    # initialize episode reward
    reward = 0
    while True:
        pipeline.step()
        reward += pipeline.reward
        if pipeline.done:
            break
    print("Episode " + str(i) + " reward:", reward)

# stop MSTDP
pipeline.network.learning = False

print("Testing: ")
for i in range(100):
    pipeline.reset_()
    # initialize episode reward
    reward = 0
    while True:
        pipeline.step()
        reward += pipeline.reward
        if pipeline.done:
            break
    print("Episode " + str(i) + " reward:", reward)
