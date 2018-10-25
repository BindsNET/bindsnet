import torch
import torch.nn as nn

from typing import Union, Sequence

import bindsnet.network.nodes as nodes
import bindsnet.network.topology as topology

from bindsnet.network import Network


def ann_to_snn(ann: Union[nn.Module, str], input_shape: Sequence[int]) -> Network:
    # language=rst
    """
    Converts an artificial neural network (ANN) written as a ``torch.nn.Module`` into a near-equivalent spiking neural
    network.

    :param ann: Artificial neural network implemented in PyTorch. Accepts either ``torch.nn.Module`` or path to network
                saved using ``torch.save()``.
    :param input_shape: Shape of input data.
    :return: Spiking neural network implemented in PyTorch.
    """
    if isinstance(ann, str):
        ann = torch.load(ann)

    assert isinstance(ann, nn.Module)

    snn = Network()

    layer = nodes.RealInput(shape=input_shape)
    snn.add_layer(layer, name='Input')
    last = ('Input', layer)

    for name, module in ann.named_children():
        if isinstance(module, nn.Linear):
            layer = nodes.IFNodes(n=module.out_features, reset=0, thresh=1)
            snn.add_layer(layer, name=name)
            connection = topology.Connection(
                source=last[1], target=layer, w=module.weight.t(), b=module.bias
            )
            snn.add_connection(connection, source=last[0], target=name)
            last = (name, layer)
        elif isinstance(module, nn.Conv2d):
            pass

    return snn
