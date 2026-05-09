import torch

from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight, Bias
from bindsnet.learning.MCC_learning import PostPre


network = Network(dt=1.0)
neurons_number = 7
layer = LIFNodes(n=neurons_number, traces=True)
network.add_layer(layer, name="input")
network.add_layer(layer, name="output")
mask = ~torch.tril(torch.ones((neurons_number, neurons_number)), diagonal=-1).bool()
weight = Weight(
    name='weight_feature',
    value=torch.rand(neurons_number, neurons_number) * mask,
    learning_rule=PostPre,
    nu=(1e-4, 1e-2)
)
bias = Bias(
    name='bias_feature',
    value=torch.rand(neurons_number, neurons_number)
)
connection = MulticompartmentConnection(
    source=layer,
    target=layer,
    pipeline=[weight, bias],
    mask=mask,
    device='cpu'
)
network.add_connection(connection, source="input", target="output")
network.run(
    inputs={"input": torch.bernoulli(torch.rand(250, neurons_number)).byte()},
    time=250
)