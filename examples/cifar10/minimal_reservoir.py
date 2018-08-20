import torch
import torch.nn as nn
from bindsnet.datasets import MNIST
from bindsnet.network import Network
from bindsnet.encoding import poisson_loader
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import LIFNodes, Input

# Define logistic regression model using PyTorch.
class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return self.linear(x)

# Build a simple, two layer, "input-output" network.
network = Network(dt=1.0)
inpt = Input(784, shape=(28, 28)); network.add_layer(inpt, name='I')
output = LIFNodes(625, thresh=-52 + torch.randn(625)); network.add_layer(output, name='O')
network.add_connection(Connection(inpt, output, w=torch.randn(inpt.n, output.n)), 'I', 'O')
network.add_connection(Connection(output, output, w=0.5 * torch.randn(output.n, output.n)), 'O', 'O')
network.add_monitor(Monitor(output, ['s'], time=250), name='output_spikes')

# Get MNIST training images and labels and create data loader.
images, labels = MNIST(path='../../data/MNIST').get_train()
loader = zip(poisson_loader(images * 0.25, time=250), iter(labels))

# Run training data on reservoir and store (spikes per neuron, label) pairs.
training_pairs = []
for i, (datum, label) in enumerate(loader):
    network.run(inpts={'I': datum}, time=250)
    training_pairs.append([network.monitors['output_spikes'].get('s').sum(-1), label])
    network.reset_()

    if (i + 1) % 50 == 0: print('Train progress: (%d / 500)' % (i + 1))
    if (i + 1) == 500: print(); break  # stop after 500 training examples

# Create and train logistic regression model on reservoir outputs.
model = LogisticRegression(625, 10); criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

# Train the logistic regression model on (spikes, label) pairs.
for epoch in range(10):
    for i, (s, label) in enumerate(training_pairs):
        optimizer.zero_grad(); output = model(s)
        loss = criterion(output.unsqueeze(0), label.unsqueeze(0).long())
        loss.backward(); optimizer.step()

# Get MNIST test images and labels and create data loader.
images, labels = MNIST(path='../../data/MNIST').get_test();
loader = zip(poisson_loader(images * 0.25, time=250), iter(labels))

# Run test data on reservoir and store (spikes per neuron, label) pairs.
test_pairs = []
for i, (datum, label) in enumerate(loader):
    network.run(inpts={'I': datum}, time=250)
    test_pairs.append([network.monitors['output_spikes'].get('s').sum(-1), label])
    network.reset_()

    if (i + 1) % 50 == 0: print('Test progress: (%d / 500)' % (i + 1))
    if (i + 1) == 500: print(); break  # stop after 500 test examples

# Test the logistic regresion model on (spikes, label) pairs.
correct, total = 0, 0
for s, label in test_pairs:
    output = model(s); _, predicted = torch.max(output.data.unsqueeze(0), 1)
    total += 1; correct += int(predicted == label.long())

print('Accuracy of logistic regression on 500 test examples: %.2f %%\n' % (100 * correct / total))