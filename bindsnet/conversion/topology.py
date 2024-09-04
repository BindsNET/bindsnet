from typing import Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from bindsnet.network import nodes, topology


class PermuteConnection(topology.AbstractConnection):
    # language=rst
    """
    Special-purpose connection for emulating the custom ``Permute`` module in
    spiking neural networks.
    """

    def __init__(
        self,
        source: nodes.Nodes,
        target: nodes.Nodes,
        dims: Iterable,
        nu: Optional[Union[float, Iterable[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``PermuteConnection``.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param dims: Order of dimensions to permute.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each
            iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according
            to some rule.
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


class ConstantPad2dConnection(topology.AbstractConnection):
    # language=rst
    """
    Special-purpose connection for emulating the ``ConstantPad2d`` PyTorch
    module in spiking neural networks.
    """

    def __init__(
        self,
        source: nodes.Nodes,
        target: nodes.Nodes,
        padding: Tuple,
        nu: Optional[Union[float, Iterable[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``ConstantPad2dConnection``.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param padding: Padding of input tensors; passed to
            ``torch.nn.functional.pad``.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each
            iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according
            to some rule.
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
