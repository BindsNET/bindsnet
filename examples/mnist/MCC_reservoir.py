import os
import numpy as np
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
from bindsnet.network.topology_features import Probability, Weight, Mask

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.utils import get_square_weights


parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--n_neurons", type=int, default=500)
parser.add_argument("--n_epochs", type=int, default=100)
parser.add_argument("--examples", type=int, default=500)
parser.add_argument("--n_workers", type=int, default=-1)
parser.add_argument("--time", type=int, default=250)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--intensity", type=float, default=64)
parser.add_argument("--progress_interval", type=int, default=10)
parser.add_argument("--update_interval", type=int, default=250)
parser.add_argument("--plot", dest="plot", action="store_true")
parser.add_argument("--gpu", dest="gpu", action="store_true")
parser.set_defaults(plot=False, gpu=True, train=True)

args = parser.parse_args()

seed = args.seed
n_neurons = args.n_neurons
n_epochs = args.n_epochs
examples = args.examples
n_workers = args.n_workers
time = args.time
dt = args.dt
intensity = args.intensity
progress_interval = args.progress_interval
update_interval = args.update_interval
train = args.train
plot = args.plot
gpu = args.gpu

np.random.seed(seed)
torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

# Sets up Gpu use
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if gpu and torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)
else:
    torch.manual_seed(seed)
    device = "cpu"
    if gpu:
        gpu = False
torch.set_num_threads(os.cpu_count() - 1)
print("Running on Device = ", device)


### Base model ###
model = Network()
model.to(device)


### Layers ###
input_l = Input(n=784, shape=(1, 28, 28), traces=True)
output_l = LIFNodes(
    n=n_neurons, thresh=-52 + np.random.randn(n_neurons).astype(float), traces=True
)

model.add_layer(input_l, name="X")
model.add_layer(output_l, name="Y")


### Connections ###
p = torch.rand(input_l.n, output_l.n)
d = torch.rand(input_l.n, output_l.n) / 5
w = torch.sign(torch.randint(-1, +2, (input_l.n, output_l.n)))
prob_feature = Probability(name="input_prob_feature", value=p)
weight_feature = Weight(name="input_weight_feature", value=w)
pipeline = [prob_feature, weight_feature]
input_con = MulticompartmentConnection(
    source=input_l,
    target=output_l,
    device=device,
    pipeline=pipeline,
)

p = torch.rand(output_l.n, output_l.n)
d = torch.rand(output_l.n, output_l.n) / 5
w = torch.sign(torch.randint(-1, +2, (output_l.n, output_l.n)))
prob_feature = Probability(name="recc_prob_feature", value=p)
weight_feature = Weight(name="recc_weight_feature", value=w)
pipeline = [prob_feature, weight_feature]
recurrent_con = MulticompartmentConnection(
    source=output_l,
    target=output_l,
    device=device,
    pipeline=pipeline,
)

model.add_connection(input_con, source="X", target="Y")
model.add_connection(recurrent_con, source="Y", target="Y")

# Directs network to GPU
if gpu:
    model.to("cuda")

### MNIST ###
dataset = MNIST(
    PoissonEncoder(time=time, dt=dt),
    None,
    root=os.path.join("../../test", "..", "data", "MNIST"),
    download=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * intensity)]
    ),
)


### Monitor setup ###
inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None
spikes = {}
voltages = {}
for l in model.layers:
    spikes[l] = Monitor(model.layers[l], ["s"], time=time, device=device)
    model.add_monitor(spikes[l], name="%s_spikes" % l)

voltages = {"Y": Monitor(model.layers["Y"], ["v"], time=time, device=device)}
model.add_monitor(voltages["Y"], name="Y_voltages")


### Running model on MNIST ###

# Create a dataloader to iterate and batch data
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True
)

n_iters = examples
training_pairs = []
pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break

    # Extract & resize the MNIST samples image data for training
    #       int(time / dt)  -> length of spike train
    #       28 x 28         -> size of sample
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Train progress: (%d / %d)" % (i, n_iters))

    # Run network on sample image
    model.run(inputs={"X": datum}, time=time, input_time_dim=1, reward=1.0)
    training_pairs.append([spikes["Y"].get("s").sum(0), label])

    # Plot spiking activity using monitors
    if plot:
        inpt_axes, inpt_ims = plot_input(
            dataPoint["image"].view(28, 28),
            datum.view(int(time / dt), 784).sum(0).view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )

        plt.pause(1e-8)
    model.reset_state_variables()


### Classification ###
# Define logistic regression model using PyTorch.
# These neurons will take the reservoirs output as its input, and be trained to classify the images.
class NN(nn.Module):
    def __init__(self, input_size, num_classes):
        super(NN, self).__init__()
        # h = int(input_size/2)
        self.linear_1 = nn.Linear(input_size, num_classes)
        # self.linear_1 = nn.Linear(input_size, h)
        # self.linear_2 = nn.Linear(h, num_classes)

    def forward(self, x):
        out = torch.sigmoid(self.linear_1(x.float().view(-1)))
        # out = torch.sigmoid(self.linear_2(out))
        return out


# Create and train logistic regression model on reservoir outputs.
learning_model = NN(n_neurons, 10).to(device)
criterion = torch.nn.MSELoss(reduction="sum")
optimizer = torch.optim.SGD(learning_model.parameters(), lr=1e-4, momentum=0.9)

# Training the Model
print("\n Training the read out")
pbar = tqdm(enumerate(range(n_epochs)))
for epoch, _ in pbar:
    avg_loss = 0

    # Extract spike outputs from reservoir for a training sample
    #       i   -> Loop index
    #       s   -> Reservoir output spikes
    #       l   -> Image label
    for i, (s, l) in enumerate(training_pairs):
        # Reset gradients to 0
        optimizer.zero_grad()

        # Run spikes through logistic regression model
        outputs = learning_model(s)

        # Calculate MSE
        label = torch.zeros(1, 1, 10).float().to(device)
        label[0, 0, l] = 1.0
        loss = criterion(outputs.view(1, 1, -1), label)
        avg_loss += loss.data

        # Optimize parameters
        loss.backward()
        optimizer.step()

    pbar.set_description_str(
        "Epoch: %d/%d, Loss: %.4f"
        % (epoch + 1, n_epochs, avg_loss / len(training_pairs))
    )

# Run same simulation on reservoir with testing data instead of training data
# (see training section for intuition)
n_iters = examples
test_pairs = []
pbar = tqdm(enumerate(dataloader))
for i, dataPoint in pbar:
    if i > n_iters:
        break
    datum = dataPoint["encoded_image"].view(int(time / dt), 1, 1, 28, 28).to(device)
    label = dataPoint["label"]
    pbar.set_description_str("Testing progress: (%d / %d)" % (i, n_iters))

    model.run(inputs={"X": datum}, time=time, input_time_dim=1)
    test_pairs.append([spikes["Y"].get("s").sum(0), label])

    if plot:
        inpt_axes, inpt_ims = plot_input(
            dataPoint["image"].view(28, 28),
            datum.view(time, 784).sum(0).view(28, 28),
            label=label,
            axes=inpt_axes,
            ims=inpt_ims,
        )
        spike_ims, spike_axes = plot_spikes(
            {layer: spikes[layer].get("s").view(time, -1) for layer in spikes},
            axes=spike_axes,
            ims=spike_ims,
        )
        voltage_ims, voltage_axes = plot_voltages(
            {layer: voltages[layer].get("v").view(time, -1) for layer in voltages},
            ims=voltage_ims,
            axes=voltage_axes,
        )

        plt.pause(1e-8)
    model.reset_state_variables()

# Test learning model with previously trained logistic regression classifier
correct, total = 0, 0
for s, label in test_pairs:
    outputs = learning_model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    correct += int(predicted == label.long().to(device))

print(
    "\n Accuracy of the model on %d test images: %.2f %%"
    % (n_iters, 100 * correct / total)
)
