import torch
import torch.nn as nn

import numpy as np

from time import time as t
from typing import Union, Sequence, Optional

import bindsnet.network.nodes as nodes
import bindsnet.network.topology as topology

from bindsnet.network import Network


class FeatureExtractor(nn.Module):
    def __init__(self, submodule):
        super(FeatureExtractor, self).__init__()

        self.submodule = submodule

    def forward(self, x):
        activations = {'input': x}
        for name, module in self.submodule._modules.items():
            if isinstance(module, nn.Linear):
                x = x.view(-1, module.in_features)

            x = module(x)
            activations[name] = x

        return activations


class SubtractiveResetIFNodes(nodes.Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_ with using reset by
    subtraction.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Sequence[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False,
                 thresh: Union[float, torch.Tensor] = -52.0, reset: Union[float, torch.Tensor] = -65.0,
                 refrac: Union[int, torch.Tensor] = 5) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        :param thresh: Spike threshold voltage.
        :param reset: Post-spike reset voltage.
        :param refrac: Refractory (non-firing) period of the neuron.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        # Post-spike reset voltage.
        if isinstance(reset, float):
            self.reset = torch.tensor(reset)
        else:
            self.reset = reset

        # Spike threshold voltage.
        if isinstance(thresh, float):
            self.thresh = torch.tensor(thresh)
        else:
            self.thresh = thresh

        # Post-spike refractory period.
        if isinstance(refrac, float):
            self.refrac = torch.tensor(refrac)
        else:
            self.refrac = refrac

        self.v = self.reset * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)   # Refractory period counters.

    def step(self, inpts: torch.Tensor, dt: float) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Integrate input voltages.
        self.v += (self.refrac_count == 0).float() * inpts

        # Decrement refractory counters.
        self.refrac_count[self.refrac_count != 0] -= dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v[self.s] = self.v[self.s] - self.thresh

        super().step(inpts, dt)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()
        self.v = self.reset * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)   # Refractory period counters.


def data_based_normalization(ann: Union[nn.Module, str], data: torch.Tensor, percentile: float = 99.9):
    # language=rst
    """
    Use a dataset to rescale ANN weights and biases such that that the max ReLU activation is less than 1.

    :param ann: Artificial neural network implemented in PyTorch. Accepts either ``torch.nn.Module`` or path to network
                saved using ``torch.save()``.
    :param data: Data to use to perform data-based weight normalization``[n_examples, ...]``.
    :param percentile: Percentile (in ``[0, 100]``) of activations to scale by in data-based normalization scheme.
    :return: Artificial neural network with rescaled weights and biases according to activations on the dataset.
    """
    if isinstance(ann, str):
        ann = torch.load(ann)

    assert isinstance(ann, nn.Module)

    def set_requires_grad(module, value):
        for param in module.parameters():
            param.requires_grad = value

    set_requires_grad(ann, value=False)
    extractor = FeatureExtractor(ann)

    prev_name, prev_layer = None, None
    prev_factor = 1
    for name, layer in ann._modules.items():
        activations = extractor.forward(data)[name]
        if isinstance(layer, nn.ReLU):
            scale_factor = np.percentile(activations.cpu(), percentile)

            prev_layer.weight *= prev_factor / scale_factor
            prev_layer.bias /= scale_factor

            prev_factor = scale_factor

        prev_name, prev_layer = name, layer

    return ann


def ann_to_snn(ann: Union[nn.Module, str], input_shape: Sequence[int], data: Optional[torch.Tensor] = None) -> Network:
    # language=rst
    """
    Converts an artificial neural network (ANN) written as a ``torch.nn.Module`` into a near-equivalent spiking neural
    network.

    :param ann: Artificial neural network implemented in PyTorch. Accepts either ``torch.nn.Module`` or path to network
                saved using ``torch.save()``.
    :param input_shape: Shape of input data.
    :param data: Data to use to perform data-based weight normalization of shape ``[n_examples, ...]``.
    :return: Spiking neural network implemented in PyTorch.
    """
    if isinstance(ann, str):
        ann = torch.load(ann)

    assert isinstance(ann, nn.Module)

    if data is not None:
        print()
        print('Example data provided. Performing data-based normalization...')

        t0 = t()
        ann = data_based_normalization(ann=ann, data=data.detach())

        print(f'Elapsed: {t() - t0:.4f}')

    snn = Network()

    layer = nodes.RealInput(shape=input_shape)
    snn.add_layer(layer, name='Input')
    last = ('Input', layer)

    for name, module in ann.named_children():
        if isinstance(module, nn.Linear):
            layer = SubtractiveResetIFNodes(n=module.out_features, reset=0, thresh=1, refrac=0)
            connection = topology.Connection(
                source=last[1], target=layer, w=module.weight.t(), b=module.bias
            )

        elif isinstance(module, nn.Conv2d):
            input_height, input_width = last[1].shape[2], last[1].shape[3]
            out_channels, output_height, output_width = module.out_channels, last[1].shape[2], last[1].shape[3]

            width = (input_height - module.kernel_size[0] + 2 * module.padding[0]) / module.stride[0] + 1
            height = (input_width - module.kernel_size[1] + 2 * module.padding[1]) / module.stride[1] + 1
            shape = (1, out_channels, int(width), int(height))

            layer = SubtractiveResetIFNodes(
                shape=shape, reset=0, thresh=1, refrac=0
            )
            connection = topology.Conv2dConnection(
                source=last[1], target=layer, kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation, w=module.weight, b=module.bias
            )

        elif isinstance(module, nn.MaxPool2d):
            input_height, input_width = last[1].shape[2], last[1].shape[3]
            out_channels, output_height, output_width = module.out_channels, last[1].shape[2], last[1].shape[3]

            width = (input_height - module.kernel_size[0] + 2 * module.padding[0]) / module.stride[0] + 1
            height = (input_width - module.kernel_size[1] + 2 * module.padding[1]) / module.stride[1] + 1
            shape = (1, out_channels, int(width), int(height))
            layer = SubtractiveResetIFNodes(
                shape=shape, reset=0, thresh=1, refrac=0
            )
            connection = topology.Conv2dConnection(
                source=last[1], target=layer, kernel_size=module.kernel_size, stride=module.stride,
                padding=module.padding, dilation=module.dilation, w=module.weight, b=module.bias
            )

        else:
            continue

        snn.add_layer(layer, name=name)
        snn.add_connection(connection, source=last[0], target=name)
        last = (name, layer)

    return snn
