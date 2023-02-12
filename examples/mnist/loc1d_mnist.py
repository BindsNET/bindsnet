### Toy example to test LocanConnection1D (the dataset used is MNIST but each image is raveled (each sample has shape (784,)).

import torch
from torch.nn.modules.utils import _pair

from tqdm import tqdm
import os
from bindsnet.network.monitors import Monitor

import torch
from torchvision import transforms
from tqdm import tqdm

from time import time as t
from torchvision import transforms
from bindsnet.learning import PostPre

from bindsnet.network.nodes import AdaptiveLIFNodes
from bindsnet.network.nodes import Input
from bindsnet.network.network import Network
from bindsnet.network.topology import Connection, LocalConnection1D
from bindsnet.encoding import PoissonEncoder
from bindsnet.datasets import MNIST

# Hyperparameters
in_channels = 1
n_filters = 25
input_shape = 784
kernel_size = 28 * 2
stride = 28
tc_theta_decay = 1e6
theta_plus = 0.05
norm = 0.2 * kernel_size
wmin = 0.0
wmax = 1.0
nu = (1e-4, 1e-2)
inh = 25.0
dt = 1.0
time = 250
intensity = 128
n_epochs = 1
n_train = 500
progress_interval = 10
batch_size = 1

# Build network
network = Network()

input_layer = Input(shape=[in_channels, input_shape], traces=True, tc_trace=20)

compute_conv_size = lambda inp_size, k, s: int((inp_size - k) / s) + 1
conv_size = compute_conv_size(input_shape, kernel_size, stride)

output_layer = AdaptiveLIFNodes(
    shape=[n_filters, conv_size],
    traces=True,
    rest=-65.0,
    reset=-60.0,
    thresh=-52.0,
    refrac=5,
    tc_decay=100.0,
    tc_trace=20.0,
    theta_plus=theta_plus,
    tc_theta_decay=tc_theta_decay,
)

input_output_conn = LocalConnection1D(
    input_layer,
    output_layer,
    kernel_size=kernel_size,
    stride=stride,
    n_filters=n_filters,
    nu=nu,
    update_rule=PostPre,
    wmin=wmin,
    wmax=wmax,
    norm=norm,
)

w_inh_LC = torch.zeros(n_filters, conv_size, n_filters, conv_size)
for c in range(n_filters):
    for w1 in range(conv_size):
        w_inh_LC[c, w1, :, w1] = -inh
        w_inh_LC[c, w1, c, w1] = 0

w_inh_LC = w_inh_LC.reshape(output_layer.n, output_layer.n)
recurrent_conn = Connection(output_layer, output_layer, w=w_inh_LC)

network.add_layer(input_layer, name="X")
network.add_layer(output_layer, name="Y")
network.add_connection(input_output_conn, source="X", target="Y")
network.add_connection(recurrent_conn, source="Y", target="Y")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
gpu = True
seed = 0

if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False

torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)

if gpu:
    network.to("cuda")

# Load MNIST data.
train_dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

spikes = {}
for layer in set(network.layers):
    spikes[layer] = Monitor(network.layers[layer], state_vars=["s"], time=time)
    network.add_monitor(spikes[layer], name="%s_spikes" % layer)

voltages = {}
for layer in set(network.layers) - {"X"}:
    voltages[layer] = Monitor(network.layers[layer], state_vars=["v"], time=time)
    network.add_monitor(voltages[layer], name="%s_voltages" % layer)

# Train the network.
print("Begin training.\n")
start = t()

for epoch in range(n_epochs):
    if epoch % progress_interval == 0:
        print("Progress: %d / %d (%.4f seconds)" % (epoch, n_epochs, t() - start))
        start = t()

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=gpu,
    )

    for step, batch in enumerate(tqdm(train_dataloader)):
        # Get next input sample.
        if step > n_train:
            break
        inputs = {"X": batch["encoded_image"].view(time, batch_size, 1, 28 * 28)}
        if gpu:
            inputs = {k: v.cuda() for k, v in inputs.items()}
        label = batch["label"]

        # Run the network on the input.
        network.run(inputs=inputs, time=time)

        network.reset_state_variables()  # Reset state variables.

print("Progress: %d / %d (%.4f seconds)\n" % (n_epochs, n_epochs, t() - start))
print("Training complete.\n")
