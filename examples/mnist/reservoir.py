import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from bindsnet.analysis.plotting import plot_input, plot_spikes, plot_voltages, plot_weights
from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson_loader
from bindsnet.network import Network, Input

# Build a simple two-layer, input-output network.
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.utils import get_square_weights

network = Network(dt=1.0)
inpt = Input(784, shape=(28, 28)); network.add_layer(inpt, name='I')
output = LIFNodes(625, thresh=-52 + torch.randn(625)); network.add_layer(output, name='O')
C1 = Connection(source=inpt, target=output, w=torch.randn(inpt.n, output.n));
C2 = Connection(source=output, target=output, w=0.5*torch.randn(output.n, output.n))

network.add_connection(C1, source='I', target='O')
network.add_connection(C2, source='O', target='O')

spikes = {}
for l in network.layers:
    spikes[l] = Monitor(network.layers[l], ['s'], time=250)
    network.add_monitor(spikes[l], name='%s_spikes' % l)

voltages = {'O' : Monitor(network.layers['O'], ['v'], time=250)}
network.add_monitor(voltages['O'], name='O_voltages')


# Get MNIST training images and labels.
images, labels = MNIST(path='../../data/MNIST',
                       download=True).get_train()
images *= 0.25

# Create lazily iterating Poisson-distributed data loader.
loader = zip(poisson_loader(images, time=250), iter(labels))

inpt_axes = None
inpt_ims = None
spike_axes = None
spike_ims = None
weights_im = None
weights_im2 = None
voltage_ims = None
voltage_axes = None

# Run training data on reservoir computer and store (spikes per neuron, label) per example.
n_iters = 500
training_pairs = []
for i, (datum, label) in enumerate(loader):
    if i % 100 == 0:
        print('Train progress: (%d / %d)' % (i, n_iters))
    
    network.run(inpts={'I' : datum}, time=250)
    training_pairs.append([spikes['O'].get('s').sum(-1), label])
    
    inpt_axes, inpt_ims = plot_input(images[i], datum.sum(0), label=label, axes=inpt_axes, ims=inpt_ims)
    spike_ims, spike_axes = plot_spikes({layer: spikes[layer].get('s').view(-1, 250) for layer in spikes},
                                        axes=spike_axes, ims=spike_ims)
    voltage_ims, voltage_axes = plot_voltages({layer: voltages[layer].get('v').view(-1, 250) for layer in voltages},
                                              ims=voltage_ims, axes=voltage_axes)
    weights_im = plot_weights(get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2)
    weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)
    
    plt.pause(1e-8)
    network.reset_()
    
    if i > n_iters:
        break


# Define logistic regression model using PyTorch.
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)
    
    def forward(self, x):
        out = self.linear(x)
        return out


# Create and train logistic regression model on reservoir outputs.
model = LogisticRegression(625, 10)
criterion = nn.CrossEntropyLoss()  
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)  

# Training the Model
for epoch in range(10):
    for i, (s, label) in enumerate(training_pairs):
        # Forward + Backward + Optimize
        optimizer.zero_grad()
        outputs = model(s)
        loss = criterion(outputs.unsqueeze(0), label.unsqueeze(0).long())
        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print('Epoch: [%d/%d], Step: [%d/%d], Loss: %.4f' % (epoch+1, 10, i+1, len(training_pairs), loss.data[0]))

# Get MNIST test images and labels.
images, labels = MNIST(path='../../data/MNIST',
                       download=True).get_test()
images *= 0.25

# Create lazily iterating Poisson-distributed data loader.
loader = zip(poisson_loader(images, time=250), iter(labels))

n_iters = 500
test_pairs = []
for i, (datum, label) in enumerate(loader):
    if i % 100 == 0:
        print('Test progress: (%d / %d)' % (i, n_iters))
    
    network.run(inpts={'I' : datum}, time=250)
    test_pairs.append([spikes['O'].get('s').sum(-1), label])
    
    inpt_axes, inpt_ims = plot_input(images[i], datum.sum(0), label=label, axes=inpt_axes, ims=inpt_ims)
    spike_ims, spike_axes = plot_spikes({layer: spikes[layer].get('s').view(-1, 250) for layer in spikes},
                                        axes=spike_axes, ims=spike_ims)
    voltage_ims, voltage_axes = plot_voltages({layer: voltages[layer].get('v').view(-1, 250) for layer in voltages},
                                              ims=voltage_ims, axes=voltage_axes)
    weights_im = plot_weights(get_square_weights(C1.w, 23, 28), im=weights_im, wmin=-2, wmax=2)
    weights_im2 = plot_weights(C2.w, im=weights_im2, wmin=-2, wmax=2)
    
    plt.pause(1e-8)
    network.reset_()
    
    if i > n_iters:
        break

# Test the Model
correct, total = 0, 0
for s, label in test_pairs:
    outputs = model(s)
    _, predicted = torch.max(outputs.data.unsqueeze(0), 1)
    total += 1
    correct += int(predicted == label.long())
    
print('Accuracy of the model on %d test images: %.2f %%' % (n_iters, 100 * correct / total))
