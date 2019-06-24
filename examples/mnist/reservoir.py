import os
import torch
import torch.nn as nn
import argparse
import matplotlib.pyplot as plt

from torchvision import transforms
from tqdm import tqdm

from bindsnet.analysis.plotting import (
    plot_input,
    plot_spikes,
    plot_voltages,
    plot_weights,
)
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.network import Network
from bindsnet.network.nodes import Input

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=100)
parser.add_argument("--n_epochs", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=128)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=False, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
n_workers = args.n_workers
exc = args.exc
inh = args.inh
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

# Sets up Gpu use
if gpu:
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)


network = Network(dt=dt)
inpt = Input(784, shape=(1, 1, 28, 28))
network.add_layer(inpt, name="I")
output = LIFNodes(n_neurons, thresh=-52 + torch.randn(n_neurons))
network.add_layer(output, name="O")
C1 = Connection(source=inpt, target=output, w=torch.randn(inpt.n, output.n))
C2 = Connection(source=output, target=output, w=0.5 * torch.randn(output.n, output.n))

network.add_connection(C1, source="I", target="O")
network.add_connection(C2, source="O", target="O")

spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ["s"], time=time)
    network.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"O": Monitor(network.layers["O"], ["v"], time=time)}
network.add_monitor(voltages["O"], name="O_voltages")

# Directs network to GPU
if gpu:
    network.to("cuda")

# Get MNIST training images and labels.
# Load MNIST data.
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("..", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=gpu
)

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
n_iters = n_epochs
training_pairs = []
pbar = tqdm(enumerate(dataloader))
i = 0
for (step, dataPoint) in pbar:
    datum = dataPoint["encoded_image"]
    label = dataPoint["label"]
    pbar.set_description("Train progress: (%d / %d)" % (i, n_iters))

    network.run(inpts={"I": datum}, time=time, input_time_dim=1)
    training_pairs.append([spikes["O"].get("s").sum(-1), label])

    inpt_axes, inpt_ims = plot_input(
        dataPoint["image"].view(28, 28),
        datum.view(time, 784).sum(0).view(28, 28),
        label=label,
        axes=inpt_axes,
        ims=inpt_ims,
    )
    spike_ims, spike_axes = plot_spikes(
        {layer: spikes[layer].get("s").view(-1, time) for layer in spikes},
        axes=spike_axes,
        ims=spike_ims,
    )
    voltage_ims, voltage_axes = plot_voltages(
        {layer: voltages[layer].get("v").view(-1, time) for layer in voltages},
        ims=voltage_ims,
        axes=voltage_axes,
    )
    weights_im = plot_weights(
        get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
    )
    weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

    plt.pause(1e-8)
    network.reset_()

    if i > n_iters:
        break

    i += 1


# Define logistic regression model using PyTorch.
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x.float())
        return out


# Create and train logistic regression model on reservoir outputs.
model = LogisticRegression(n_neurons, 10)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Training the Model
print("\n Training the read out")
pbar = tqdm(enumerate(range(10)))
for epoch in pbar:
    for i, (s, label) in enumerate(training_pairs):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(s)
        loss = criterion(outputs.unsqueeze(0), label.long())
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            pbar.set_description(
                "Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f"
                % (epoch + 1, 10, i + 1, len(training_pairs), loss.data[0])
            )

n_iters = n_epochs
test_pairs = []
pbar = tqdm(enumerate(dataloader))
i = 0
for (step, dataPoint) in pbar:
    datum = dataPoint["encoded_image"]
    label = dataPoint["label"]
    pbar.set_description("Train progress: (%d / %d)" % (i, n_iters))

    network.run(inpts={"I": datum}, time=250, input_time_dim=1)
    test_pairs.append([spikes["O"].get("s").sum(-1), label])

    inpt_axes, inpt_ims = plot_input(
        dataPoint["image"].view(28, 28),
        datum.view(time, 784).sum(0).view(28, 28),
        label=label,
        axes=inpt_axes,
        ims=inpt_ims,
    )
    spike_ims, spike_axes = plot_spikes(
        {layer: spikes[layer].get("s").view(-1, 250) for layer in spikes},
        axes=spike_axes,
        ims=spike_ims,
    )
    voltage_ims, voltage_axes = plot_voltages(
        {layer: voltages[layer].get("v").view(-1, 250) for layer in voltages},
        ims=voltage_ims,
        axes=voltage_axes,
    )
    weights_im = plot_weights(
        get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2
    )
    weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)

    plt.pause(1e-8)
    network.reset_()

    if i > n_iters:
        break
    i += 1

# Test the Model
correct, total = 0, 0
for s, label in test_pairs:
    outputs = model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    correct += int(predicted == label.long())

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (n_iters, 100 * correct / total)
)
