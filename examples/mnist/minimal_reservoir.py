import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

# Define logistic regression model using PyTorch.
from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder

from bindsnet.network import Network
from bindsnet.pipeline import DataLoaderPipeline
from bindsnet.network.monitors import Monitor
from bindsnet.network.topology import Connection
from bindsnet.network.nodes import LIFNodes, Input


class LogisticRegression(nn.Module):
    def __init__(self, input_size, num_classes):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        return F.sigmoid(self.linear(x)).unsqueeze(0)


class CustomPipeline(DataLoaderPipeline):
    def __init__(
        self,
        network: Network,
        train_ds: torch.utils.data.Dataset,
        test_ds: torch.utils.data.Dataset,
        logistic_reg: LogisticRegression,
        **kwargs
    ):
        super().__init__(network, train_ds, test_ds, **kwargs)

        self.model = logistic_reg
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1)
        self.loss = nn.CrossEntropyLoss()

    def init_fn(self):
        pass

    def step_(self, batch):
        self.network.reset_()
        inpts = {"I": batch["encoded_image"]}
        self.network.run(inpts, time=batch["encoded_image"].shape[0], input_time_dim=1)

        s = network.monitors["output_spikes"].get("s").sum(-1)
        label = batch["label"]

        # train the logistic regression model
        self.optimizer.zero_grad()
        output = self.model(s.float())
        loss = self.loss(output, label.long())
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def plots(self, input_batch, loss):
        print("Loss at step %d: %f" % (self.step_count, loss))

    def test_step(self):
        pass


# Build a simple, two layer, "input-output" network.
network = Network(dt=1.0)

inpt = Input(784, shape=(1, 1, 28, 28))
network.add_layer(inpt, name="I")
output = LIFNodes(625, thresh=-52 + torch.randn(625))
network.add_layer(output, name="O")
network.add_connection(
    Connection(inpt, output, w=torch.randn(inpt.n, output.n)), "I", "O"
)
network.add_connection(
    Connection(output, output, w=0.5 * torch.randn(output.n, output.n)), "O", "O"
)

# Specify dataset
mnist = MNIST(
    PoissonEncoder(time=250.0, dt=1.0),
    None,
    "../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128.0)]
    ),
)

log_reg_model = LogisticRegression(625, 10)

pipeline = CustomPipeline(
    network, mnist, mnist, log_reg_model, plot_interval=100, plot_length=10
)

pipeline.train()

# # Run test data on reservoir and store (spikes per neuron, label) pairs.
# test_pairs = []
# for i, (datum, label) in enumerate(loader):
#     network.run(inpts={'I' : datum}, time=250)
#     test_pairs.append([network.monitors['output_spikes'].get('s').sum(-1), label])
#     network.reset_()
#
#     if (i + 1) % 50 == 0: print('Test progress: (%d / 500)' % (i + 1))
#     if (i + 1) == 500: print(); break  # stop after 500 test examples
#
# # Test the logistic regression model on (spikes, label) pairs.
# correct, total = 0, 0
# for s, label in test_pairs:
#     output = model(s.float()); _, predicted = torch.max(output.data.unsqueeze(0), 1)
#     total += 1; correct += int(predicted == label.long())

# print('Accuracy of logistic regression on 500 test examples: %.2f %%\n' % (100 * correct / total))
