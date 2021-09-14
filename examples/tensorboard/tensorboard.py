import argparse
import os
from time import time as t

import torch
from torchvision import transforms
from tqdm import tqdm

import bindsnet.datasets
from bindsnet.analysis.pipeline_analysis import MatplotlibAnalyzer, TensorboardAnalyzer
from bindsnet.encoding import NullEncoder, PoissonEncoder
from bindsnet.learning import PostPre
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection, Conv2dConnection

parser = argparse.ArgumentParser()
parser.add_argument(
    "--dataset",
    type=str,
    default="MNIST",
    choices=["MNIST", "KMNIST", "FashionMNIST", "CIFAR10", "CIFAR100"],
)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--time", type=int, default=50)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--tensorboard", dest="tensorboard", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
torch.manual_seed(seed)

# Encoding parameters
time = args.time
dt = args.dt

# Convolution parameters
kernel_size = 5
stride = 2
n_filters = 5
padding = 0

# Create the datasets and loaders
# This is dynamic so you can test each dataset easily
dataset_type = getattr(bindsnet.datasets, args.dataset)
dataset_path = os.path.join("..", "..", "data", args.dataset)
train_dataset = dataset_type(
    PoissonEncoder(time=time, dt=dt),
    NullEncoder(),
    dataset_path,
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128.0)]
    ),
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=1, shuffle=True, num_workers=0
)

# Grab the shape of a single sample (not including batch)
# So, TxCxHxW
sample_shape = train_dataset[0]["encoded_image"].shape
print(args.dataset, " has shape ", sample_shape)

conv_size = int((sample_shape[-1] - kernel_size + 2 * padding) / stride) + 1
per_class = int((n_filters * conv_size * conv_size) / 10)

# Build a small convolutional network
network = Network()

# Make sure to include the batch dimension but not time
input_layer = Input(shape=(sample_shape[1:]), traces=True)

conv_layer = LIFNodes(
    n=n_filters * conv_size * conv_size,
    shape=(n_filters, conv_size, conv_size),
    traces=True,
)

conv_conn = Conv2dConnection(
    input_layer,
    conv_layer,
    kernel_size=kernel_size,
    stride=stride,
    update_rule=PostPre,
    norm=0.4 * kernel_size ** 2,
    nu=[1e-4, 1e-2],
    wmax=1.0,
)


network.add_layer(input_layer, name="X")
network.add_layer(conv_layer, name="Y")
network.add_connection(conv_conn, source="X", target="Y")

# Train the network.
print("Begin training.\n")

if args.tensorboard:
    analyzer = TensorboardAnalyzer("logs/conv")
else:
    analyzer = MatplotlibAnalyzer()

for step, batch in enumerate(tqdm(train_dataloader)):
    # batch contains image, label, encoded_image since an image_encoder
    # was provided

    # batch["encoded_image"] is in BxTxCxHxW format
    inputs = {"X": batch["encoded_image"].view(time, 1, 1, 28, 28)}

    # Run the network on the input.
    # Specify the location of the time dimension
    network.run(inputs=inputs, time=time, input_time_dim=1)

    network.reset_state_variables()  # Reset state variables.

    analyzer.plot_conv2d_weights(conv_conn.w, step=step)

    analyzer.finalize_step()
