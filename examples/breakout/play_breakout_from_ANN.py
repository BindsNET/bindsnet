import argparse
from typing import Iterable, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

from bindsnet.encoding import bernoulli, poisson
from bindsnet.environment import GymEnvironment
from bindsnet.network import Network
from bindsnet.network.nodes import (
    AbstractInput,
    IFNodes,
    Input,
    IzhikevichNodes,
    LIFNodes,
    Nodes,
)
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.pipeline.action import *

parser = argparse.ArgumentParser(prefix_chars="@")
parser.add_argument("@@seed", type=int, default=42)
parser.add_argument("@@dt", type=float, default=1.0)
parser.add_argument("@@gpu", dest="gpu", action="store_true")
parser.add_argument("@@layer1scale", dest="layer1scale", type=float, default=57.68)
parser.add_argument("@@layer2scale", dest="layer2scale", type=float, default=77.48)
parser.add_argument("@@num_episodes", type=int, default=10)
parser.add_argument("@@plot_interval", type=int, default=1)
parser.add_argument("@@rander_interval", type=int, default=1)
parser.set_defaults(plot=False, render=False, gpu=True, probabilistic=False)
locals().update(vars(parser.parse_args()))

# Setup PyTorch computing device
device = torch.device("cuda" if torch.cuda.is_available() and gpu else "cpu")
torch.random.manual_seed(seed)


# Build ANN
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(6400, 1000)
        self.fc2 = nn.Linear(1000, 4)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# load ANN
dqn_network = torch.load("trained_shallow_ANN.pt", map_location=device)

# Build Spiking network.
network = Network(dt=dt).to(device)

# Layers of neurons.
inpt = Input(n=6400, traces=False)  # Input layer
middle = LIFNodes(
    n=1000, refrac=0, traces=True, thresh=-52.0, rest=-65.0
)  # Hidden layer
readout = LIFNodes(
    n=4, refrac=0, traces=True, thresh=-52.0, rest=-65.0
)  # Readout layer
layers = {"X": inpt, "M": middle, "R": readout}

# Set the connections between layers with the values set by the ANN
# Input -> hidden.
inpt_middle = Connection(
    source=layers["X"],
    target=layers["M"],
    w=torch.transpose(dqn_network.fc1.weight, 0, 1) * layer1scale,
)
# hidden -> readout.
middle_out = Connection(
    source=layers["M"],
    target=layers["R"],
    w=torch.transpose(dqn_network.fc2.weight, 0, 1) * layer2scale,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="Input Layer")
network.add_layer(middle, name="Hidden Layer")
network.add_layer(readout, name="Output Layer")
network.add_connection(inpt_middle, source="Input Layer", target="Hidden Layer")
network.add_connection(middle_out, source="Hidden Layer", target="Output Layer")

# Load the Breakout environment.
environment = GymEnvironment("BreakoutDeterministic-v4")
environment.reset()

# Build pipeline from specified components.
pipeline = EnvironmentPipeline(
    network,
    environment,
    encoding=poisson,
    encode_factor=50,
    action_function=select_highest,
    percent_of_random_action=0.05,
    random_action_after=5,
    output="Output Layer",
    reset_output_spikes=True,
    time=500,
    overlay_input=4,
    history_length=1,
    plot_interval=plot_interval if plot else None,
    render_interval=render_interval if render else None,
    device=device,
)

# Run environment simulation for number of episodes.
for i in tqdm(range(num_episodes)):
    total_reward = 0
    pipeline.reset_state_variables()
    is_done = False
    pipeline.env.step(1)  # start with fire the ball
    pipeline.env.step(1)  # start with fire the ball
    while not is_done:
        result = pipeline.env_step()
        pipeline.step(result)

        reward = result[1]
        total_reward += reward

        is_done = result[2]
    tqdm.write(f"Episode {i} total reward:{total_reward}")
    with open("play-breakout_results.csv", "a") as myfile:
        myfile.write(f"{i},{layer1scale},{layer2scale},{total_reward}\n")
