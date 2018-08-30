import torch
import warnings
import numpy as np
import torch.nn.functional as F

from typing import Union, Tuple, Optional
from abc import ABC, abstractmethod
from torch.nn.modules.utils import _pair

from .nodes import Nodes


class AbstractConnection(ABC):
    # language=rst
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(self, source: Nodes, target: Nodes,
                 nu: Optional[Union[float, Tuple[float, float]]] = None, **kwargs) -> None:
        # language=rst
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param nu_pre: Learning rate for pre-synaptic events.
        :param nu_post: Learning rate for post-synpatic events.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        """
        self.w = None
        self.source = source
        self.target = target
        self.nu = nu

        assert isinstance(source, Nodes), 'Source is not a Nodes object'
        assert isinstance(target, Nodes), 'Target is not a Nodes object'

        from ..learning import NoOp

        self.update_rule = kwargs.get('update_rule', NoOp)
        self.wmin = kwargs.get('wmin', float('-inf'))
        self.wmax = kwargs.get('wmax', float('inf'))
        self.norm = kwargs.get('norm', None)
        self.decay = kwargs.get('decay', None)

        self.update_rule = self.update_rule(
            connection=self, nu=self.nu
        )

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        # language=rst
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """
        pass

    @abstractmethod
    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        reward = kwargs.get('reward', None)
        self.update_rule.update(reward=reward)

        mask = kwargs.get('mask', None)
        if mask is not None:
            self.w.masked_fill_(mask, 0)

    @abstractmethod
    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of incoming connection weights equal to ``self.norm``.
        """
        pass

    @abstractmethod
    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        pass


class Connection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Tuple[float, float]]] = None,
                 **kwargs) -> None:
        # language=rst
        """
        Instantiates a :code:`SimpleConnection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param nu_pre: Learning rate for pre-synaptic events.
        :param nu_post: Learning rate for post-synpatic events.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, **kwargs)

        self.w = kwargs.get('w', None)

        if self.w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                self.w = torch.rand(*source.shape, *target.shape)
            else:
                self.w = self.wmin + torch.rand(*source.shape, *target.shape) * (self.wmax - self.wmin)
        else:
            if torch.max(self.w) > self.wmax or torch.min(self.w) < self.wmin:
                warnings.warn(f'Weight matrix will be clamped between [{self.wmin}, {self.wmax}]')
                self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or with decaying spike activation).
        """
        # Decaying spike activation from previous iteration.
        if self.decay is not None:
            self.a_pre = self.a_pre * self.decay + s.float().view(-1)
        else:
            self.a_pre = s.float().view(-1)

        # Compute multiplication of pre-activations by connection weights.
        if self.w.shape[0] == self.source.n and self.w.shape[1] == self.target.n:
            return self.a_pre @ self.w
        else:
            a_post = self.a_pre @ self.w.view(self.source.n, self.target.n)
            return a_post.view(*self.target.shape)

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to ``self.norm``.
        """
        if self.norm is not None:
            self.w = self.w.view(self.source.n, self.target.n)
            self.w *= self.norm / self.w.sum(0).view(1, -1)
            self.w = self.w.view(*self.source.shape, *self.target.shape)

    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_()


class Conv2dConnection(AbstractConnection):
    # language=rst
    """
    Specifies convolutional synapses between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]] = 1, padding: Union[int, Tuple[int, int]] = 0,
                 dilation: Union[int, Tuple[int, int]] = 1, nu: Optional[Union[float, Tuple[float, float]]] = None,
                 **kwargs) -> None:
        # language=rst
        """
        Instantiates a ``Conv2dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param padding: Horizontal and vertical padding for convolution.
        :param dilation: Horizontal and vertical dilation for convolution.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param nu_pre: Learning rate for pre-synaptic events.
        :param nu_post: Learning rate for post-synpatic events.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, **kwargs)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        assert source.shape[0] == target.shape[0], 'Minibatch size not equal across source and target'

        minibatch = source.shape[0]
        self.in_channels, input_height, input_width = source.shape[1], source.shape[2], source.shape[3]
        self.out_channels, output_height, output_width = target.shape[1], target.shape[2], target.shape[3]

        error = 'Target dimensionality must be (minibatch, out_channels, \
                        (input_height - filter_height + 2 * padding_height) / stride_height + 1, \
                        (input_width - filter_width + 2 * padding_width) / stride_width + 1'

        width = (input_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        height = (input_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1
        shape = (minibatch, self.out_channels, width, height)

        assert tuple(target.shape) == shape, error

        self.w = kwargs.get('w', torch.rand(self.out_channels, self.in_channels, *self.kernel_size))
        self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Spikes multiplied by synapse weights.
        """
        return F.conv2d(s.float(), self.w, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights along the first axis according to total weight per target neuron.
        """
        if self.norm is not None:
            shape = self.w.size()
            self.w = self.w.view(self.w.size(0), self.w.size(2) * self.w.size(3))

            for fltr in range(self.w.size(0)):
                self.w[fltr] *= self.norm / self.w[fltr].sum(0)

            self.w = self.w.view(*shape)

    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_()


class SparseConnection(AbstractConnection):
    # language=rst
    """
    Specifies sparse synapses between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Tuple[float, float]]] = None,
                 **kwargs) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object with sparse weights.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param nu_pre: Learning rate for pre-synaptic events.
        :param nu_post: Learning rate for post-synpatic events.

        Keyword arguments:

        :param torch.Tensor w: Strengths of synapses.
        :param float sparsity: Fraction of sparse connections to use.
        :param function update_rule: Modifies connection parameters according to some rule.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, **kwargs)

        self.w = kwargs.get('w', None)
        self.sparsity = kwargs.get('sparsity', None)

        assert (self.w is not None and self.sparsity is None or
                self.w is None and self.sparsity is not None), 'Only one of "weights" or "sparsity" must be specified'

        if self.w is None and self.sparsity is not None:
            i = torch.bernoulli(1 - self.sparsity * torch.ones(*source.shape, *target.shape))
            v = self.wmin + (self.wmax - self.wmin) * torch.rand(*source.shape, *target.shape)[i.byte()]
            self.w = torch.sparse.FloatTensor(i.nonzero().t(), v)
        elif self.w is not None:
            assert self.w.is_sparse, 'Weight matrix is not sparse (see torch.sparse module)'

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Spikes multiplied by synapse weights.
        """
        s = s.float().view(-1)
        a = s @ self.w
        return a.view(*self.target.shape)

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        pass

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights along the first axis according to total weight per target neuron.
        """
        pass

    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_()
