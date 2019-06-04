import argparse
import torch
import numpy as np
from tqdm import tqdm

from bindsnet.datasets import SpokenMNIST

from bindsnet.network import Network
from bindsnet.encoding import PoissonEncoder
from bindsnet.network.nodes import DiehlAndCookNodes, RealInput
from bindsnet.network.topology import Connection

from bindsnet.models import DiehlAndCook2015

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dt", type=int, default=1.0)
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)

# Encoding parameters
dt = args.dt

dataset_path = "../../data/SpokenMNIST"
train_dataset = SpokenMNIST(dataset_path, download=True, train=True)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0
)

sample_shape = train_dataset[0][0].shape
print("SpokenMNIST has shape ", sample_shape)

network = Network()
# Make sure to include the batch dimension but not time
input_layer = RealInput(shape=(1, *sample_shape[1:]), traces=True)

out_layer = DiehlAndCookNodes(shape=(1, *sample_shape[1:]), traces=True)

out_conn = Connection(input_layer, out_layer, wmin=0, wmax=1)

network.add_layer(input_layer, name="X")
network.add_layer(out_layer, name="Y")
network.add_connection(out_conn, source="X", target="Y")

for step, batch in enumerate(tqdm(train_dataloader)):
    inpts = {"X": batch["audio"]}

    # the audio has potentially a variable amount of time
    time = spike_audio.shape[1]

    network.run(inpts=inpts, time=time, input_time_dim=1)

    network.reset_()
