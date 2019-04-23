import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.modules.utils import _pair

from copy import deepcopy
from typing import Union, Sequence, Optional, Tuple, Dict

import bindsnet.network.nodes as nodes
import bindsnet.network.topology as topology

from bindsnet.network import Network


class Permute(nn.Module):
    # language=rst
    """
    PyTorch module for the explicit permutation of a tensor's dimensions in a parent
    module's ``forward`` pass (as opposed to ``torch.permute``).
    """

    def __init__(self, dims):
        # language=rst
        """
        Constructor for ``Permute`` module.

        :param dims: Ordering of dimensions for permutation.
        """
        super(Permute, self).__init__()

        self.dims = dims

    def forward(self, x):
        # language=rst
        """
        Forward pass of permutation module.

        :param x: Input tensor to permute.
        :return: Permuted input tensor.
        """
        return x.permute(*self.dims).contiguous()


class FeatureExtractor(nn.Module):
    # language=rst
    """
    Special-purpose PyTorch module for the extraction of child module's activations.
    """

    def __init__(self, submodule):
        # language=rst
        """
        Constructor for ``FeatureExtractor`` module.

        :param submodule: The module who's children modules are to be extracted.
        """
        super(FeatureExtractor, self).__init__()

        self.submodule = submodule

    def forward(self, x: torch.Tensor) -> Dict[nn.Module, torch.Tensor]:
        # language=rst
        """
        Forward pass of the feature extractor.

        :param x: Input data for the ``submodule''.
        :return: A dictionary mapping
        """
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
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        # Integrate input voltages.
        self.v += (self.refrac_count == 0).float() * x

        # Decrement refractory counters.
        self.refrac_count[self.refrac_count > 0] -= self.dt

        # Check for spiking neurons.
        self.s = self.v >= self.thresh

        # Refractoriness and voltage reset.
        self.refrac_count.masked_fill_(self.s, self.refrac)
        self.v[self.s] = self.v[self.s] - self.thresh

        super().forward(x)

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        super().reset_()

        self.v = self.reset * torch.ones(self.shape)  # Neuron voltages.
        self.refrac_count = torch.zeros(self.shape)  # Refractory period counters.

    def _compute_decays(self) -> None:
        # language=rst
        """
        Sets the relevant decays.
        """
        super()._compute_decays()


class PassThroughNodes(nodes.Nodes):
    # language=rst
    """
    Layer of `integrate-and-fire (IF) neurons <http://neuronaldynamics.epfl.ch/online/Ch1.S3.html>`_ with using reset by
    subtraction.
    """

    def __init__(self, n: Optional[int] = None, shape: Optional[Sequence[int]] = None, traces: bool = False,
                 trace_tc: Union[float, torch.Tensor] = 5e-2, sum_input: bool = False) -> None:
        # language=rst
        """
        Instantiates a layer of IF neurons.

        :param n: The number of neurons in the layer.
        :param shape: The dimensionality of the layer.
        :param traces: Whether to record spike traces.
        :param trace_tc: Time constant of spike trace decay.
        :param sum_input: Whether to sum all inputs.
        """
        super().__init__(n, shape, traces, trace_tc, sum_input)

        self.v = torch.zeros(self.shape)

    def forward(self, x: torch.Tensor) -> None:
        # language=rst
        """
        Runs a single simulation step.

        :param inpts: Inputs to the layer.
        :param dt: Simulation time step.
        """
        self.s = x

    def reset_(self) -> None:
        # language=rst
        """
        Resets relevant state variables.
        """
        self.s = torch.zeros(self.shape)


class PermuteConnection(topology.AbstractConnection):
    # language=rst
    """
    Special-purpose connection for emulating the custom ``Permute`` module in spiking neural networks.
    """

    def __init__(self, source: nodes.Nodes, target: nodes.Nodes, dims: Sequence,
                 nu: Optional[Union[float, Sequence[float]]] = None, weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``PermuteConnection``.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param dims: Order of dimensions to permute.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.dims = dims

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Permute input.

        :param s: Input.
        :return: Permuted input.
        """
        return s.permute(self.dims).float()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``update``.
        """
        pass

    def normalize(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``normalize``.
        """
        pass

    def reset_(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``reset_``.
        """
        pass


class ConstantPad2dConnection(topology.AbstractConnection):
    # language=rst
    """
    Special-purpose connection for emulating the ``ConstantPad2d`` PyTorch module in spiking neural networks.
    """

    def __init__(self, source: nodes.Nodes, target: nodes.Nodes, padding: Tuple,
                 nu: Optional[Union[float, Sequence[float]]] = None, weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``ConstantPad2dConnection``.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param padding: Padding of input tensors; passed to ``torch.nn.functional.pad``.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """

        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.padding = padding

    def compute(self, s: torch.Tensor):
        # language=rst
        """
        Pad input.

        :param s: Input.
        :return: Padding input.
        """
        return F.pad(s, self.padding).float()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``update``.
        """
        pass

    def normalize(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``normalize``.
        """
        pass

    def reset_(self) -> None:
        # language=rst
        """
        Dummy definition of abstract method ``reset_``.
        """
        pass


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
    all_activations = extractor.forward(data)

    prev_module = None
    prev_factor = 1
    for name, module in ann._modules.items():
        if isinstance(module, nn.Sequential):

            extractor2 = FeatureExtractor(module)
            all_activations2 = extractor2.forward(data)
            for name2, module2 in module.named_children():
                activations = all_activations2[name2]

                if isinstance(module2, nn.ReLU):
                    if prev_module is not None:
                        scale_factor = np.percentile(activations.cpu(), percentile)

                        prev_module.weight *= prev_factor / scale_factor
                        prev_module.bias /= scale_factor

                        prev_factor = scale_factor

                elif isinstance(module2, nn.Linear) or isinstance(module2, nn.Conv2d):
                    prev_module = module2

        else:
            activations = all_activations[name]
            if isinstance(module, nn.ReLU):
                if prev_module is not None:
                    scale_factor = np.percentile(activations.cpu(), percentile)

                    prev_module.weight *= prev_factor / scale_factor
                    prev_module.bias /= scale_factor

                    prev_factor = scale_factor

            elif isinstance(module, nn.Linear) or isinstance(module, nn.Conv2d):
                prev_module = module

    return ann


def _ann_to_snn_helper(prev, current, node_type, **kwargs):
    # language=rst
    """
    Helper function for main ``ann_to_snn`` method.

    :param prev: Previous PyTorch module in artificial neural network.
    :param current: Current PyTorch module in artificial neural network.
    :return: Spiking neural network layer and connection corresponding to ``prev`` and ``current`` PyTorch modules.
    """
    if isinstance(current, nn.Linear):
        layer = node_type(n=current.out_features, reset=0, thresh=1, refrac=0, **kwargs)
        connection = topology.Connection(
            source=prev, target=layer, w=current.weight.t(), b=current.bias
        )

    elif isinstance(current, nn.Conv2d):
        input_height, input_width = prev.shape[2], prev.shape[3]
        out_channels, output_height, output_width = current.out_channels, prev.shape[2], prev.shape[3]

        width = (input_height - current.kernel_size[0] + 2 * current.padding[0]) / current.stride[0] + 1
        height = (input_width - current.kernel_size[1] + 2 * current.padding[1]) / current.stride[1] + 1
        shape = (1, out_channels, int(width), int(height))

        layer = node_type(shape=shape, reset=0, thresh=1, refrac=0, **kwargs)
        connection = topology.Conv2dConnection(
            source=prev, target=layer, kernel_size=current.kernel_size, stride=current.stride,
            padding=current.padding, dilation=current.dilation, w=current.weight, b=current.bias
        )

    elif isinstance(current, nn.MaxPool2d):
        input_height, input_width = prev.shape[2], prev.shape[3]
        current.kernel_size = _pair(current.kernel_size)
        current.padding = _pair(current.padding)
        current.stride = _pair(current.stride)

        width = (input_height - current.kernel_size[0] + 2 * current.padding[0]) / current.stride[0] + 1
        height = (input_width - current.kernel_size[1] + 2 * current.padding[1]) / current.stride[1] + 1
        shape = (1, prev.shape[1], int(width), int(height))

        layer = PassThroughNodes(
            shape=shape
        )
        connection = topology.MaxPool2dConnection(
            source=prev, target=layer, kernel_size=current.kernel_size, stride=current.stride,
            padding=current.padding, dilation=current.dilation, decay=1
        )

    elif isinstance(current, Permute):
        layer = PassThroughNodes(
            shape=[
                prev.shape[current.dims[0]], prev.shape[current.dims[1]],
                prev.shape[current.dims[2]], prev.shape[current.dims[3]]
            ]
        )

        connection = PermuteConnection(
            source=prev, target=layer, dims=current.dims
        )

    elif isinstance(current, nn.ConstantPad2d):
        layer = PassThroughNodes(
            shape=[
                prev.shape[0], prev.shape[1],
                current.padding[0] + current.padding[1] + prev.shape[2],
                current.padding[2] + current.padding[3] + prev.shape[3]
            ]
        )

        connection = ConstantPad2dConnection(
            source=prev, target=layer, padding=current.padding
        )

    else:
        return None, None

    return layer, connection


def ann_to_snn(ann: Union[nn.Module, str], input_shape: Sequence[int], data: Optional[torch.Tensor] = None,
               percentile: float = 99.9, node_type: Optional[nodes.Nodes] = SubtractiveResetIFNodes,
               **kwargs) -> Network:
    # language=rst
    """
    Converts an artificial neural network (ANN) written as a ``torch.nn.Module`` into a near-equivalent spiking neural
    network.

    :param ann: Artificial neural network implemented in PyTorch. Accepts either ``torch.nn.Module`` or path to network
                saved using ``torch.save()``.
    :param input_shape: Shape of input data.
    :param data: Data to use to perform data-based weight normalization of shape ``[n_examples, ...]``.
    :param percentile: Percentile (in ``[0, 100]``) of activations to scale by in data-based normalization scheme.
    :return: Spiking neural network implemented in PyTorch.
    """
    if isinstance(ann, str):
        ann = torch.load(ann)
    else:
        ann = deepcopy(ann)

    assert isinstance(ann, nn.Module)

    if data is None:
        import warnings
        warnings.warn('Data is None. Weights will not be scaled.', RuntimeWarning)
    else:
        ann = data_based_normalization(
            ann=ann, data=data.detach(), percentile=percentile
        )

    snn = Network()

    input_layer = nodes.RealInput(shape=input_shape)
    snn.add_layer(input_layer, name='Input')

    children = []
    for c in ann.children():
        if isinstance(c, nn.Sequential):
            for c2 in list(c.children()):
                children.append(c2)
        else:
            children.append(c)

    i = 0
    prev = input_layer
    while i < len(children) - 1:
        current, nxt = children[i:i + 2]
        layer, connection = _ann_to_snn_helper(prev, current, node_type, **kwargs)

        i += 1

        if layer is None or connection is None:
            continue

        snn.add_layer(layer, name=str(i))
        snn.add_connection(connection, source=str(i - 1), target=str(i))

        prev = layer

    current = children[-1]
    layer, connection = _ann_to_snn_helper(prev, current, node_type, **kwargs)

    i += 1

    if layer is not None or connection is not None:
        snn.add_layer(layer, name=str(i))
        snn.add_connection(connection, source=str(i - 1), target=str(i))

    return snn
