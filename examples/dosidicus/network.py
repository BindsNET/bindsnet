import torch

from bindsnet.network import Network
from bindsnet.network.nodes import LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight, Bias
from bindsnet.learning.MCC_learning import PostPre

network = Network(dt=1.0)
source_layer = LIFNodes(n=5, traces=True)
target_layer = LIFNodes(n=5, traces=True)


network.add_layer(source_layer, name="input")
network.add_layer(target_layer, name="output")

weight = Weight(
    name='weight_feature',
    value=torch.rand(5, 5),
    learning_rule=PostPre,
    nu=(1e-4, 1e-2)
)
bias = Bias(name='bias_feature', value=torch.rand(5, 5))

connection = MulticompartmentConnection(
    source=source_layer,
    target=target_layer,
    pipeline=[weight, bias],
    device='cpu'
)
network.add_connection(connection, source="input", target="output")
print(connection.pipeline[0].value)
network.run(
    inputs={"input": torch.bernoulli(torch.rand(250, 5)).byte()},
    time=250,
    masks={
       ('input', 'output'): ~torch.tril(torch.ones((5, 5)), diagonal=-1).bool()
    }
)
print(connection.pipeline[0].value)