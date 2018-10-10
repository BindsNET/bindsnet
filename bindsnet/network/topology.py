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
                 nu: Optional[Union[float, Tuple[float, float]]] = None, weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

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
        self.weight_decay = weight_decay

        assert isinstance(source, Nodes), 'Source is not a Nodes object'
        assert isinstance(target, Nodes), 'Target is not a Nodes object'

        from ..learning import NoOp

        self.update_rule = kwargs.get('update_rule', NoOp)
        self.wmin = kwargs.get('wmin', None)
        self.wmax = kwargs.get('wmax', None)
        self.norm = kwargs.get('norm', None)
        self.decay = kwargs.get('decay', None)

        if self.update_rule is None:
            self.update_rule = NoOp

        if self.decay is None:
            self.decay = 0.0  # No memory of previous spikes.

        self.a_pre = 0.0

        self.update_rule = self.update_rule(
            connection=self, nu=self.nu, weight_decay=weight_decay
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
    def update(self, dt, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        learning = kwargs.get('learning', True)
        reward = kwargs.get('reward', None)

        if learning:
            self.update_rule.update(dt=dt, reward=reward)

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
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.w = kwargs.get('w', None)

        if self.w is None:
            if self.wmin is None or self.wmax is None:
                self.w = torch.rand(source.n, target.n)
            elif self.wmin is not None and self.wmax is not None:
                self.w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if self.wmin is not None and self.wmax is not None:
                self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        # Decaying spike activations by decay constant.
        self.a_pre = self.a_pre * self.decay + s.float().view(-1)

        # Compute multiplication of spike activations by connection weights.
        a_post = self.a_pre @ self.w
        return a_post.view(*self.target.shape)

    def update(self, dt, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(dt=dt, **kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to ``self.norm``.
        """
        if self.norm is not None:
            self.w *= self.norm / self.w.abs().sum(0).view(1, -1)

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
                 weight_decay: float = 0.0, **kwargs) -> None:
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
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        assert source.shape[0] == target.shape[0], 'Minibatch size not equal across source and target'

        minibatch = source.shape[0]
        self.in_channels, input_height, input_width = source.shape[1], source.shape[2], source.shape[3]
        self.out_channels, output_height, output_width = target.shape[1], target.shape[2], target.shape[3]

        width = (input_height - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[0] + 1
        height = (input_width - self.kernel_size[1] + 2 * self.padding[1]) / self.stride[1] + 1
        shape = (minibatch, self.out_channels, width, height)

        error = 'Target dimensionality must be (minibatch, out_channels,' \
                '(input_height - filter_height + 2 * padding_height) / stride_height + 1,' \
                '(input_width - filter_width + 2 * padding_width) / stride_width + 1'

        assert tuple(target.shape) == shape, error

        self.w = kwargs.get('w', torch.rand(self.out_channels, self.in_channels, *self.kernel_size))
        if self.wmin is not None and self.wmax is not None:
            self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Spikes multiplied by synapse weights.
        """
        return F.conv2d(s.float(), self.w, stride=self.stride, padding=self.padding, dilation=self.dilation)

    def update(self, dt, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(dt=dt, **kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights along the first axis according to total weight per target neuron.
        """
        if self.norm is not None:
            shape = self.w.size()
            self.w = self.w.view(self.w.size(0) * self.w.size(1), self.w.size(2) * self.w.size(3))

            for fltr in range(self.w.size(0)):
                self.w[fltr] *= self.norm / self.w[fltr].sum(0)

            self.w = self.w.view(*shape)

    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_()


class LocallyConnectedConnection(AbstractConnection):
    # language=rst
    """
    Specifies a locally connected connection between one or two populations of neurons.
    """

    def __init__(self, source: Nodes, target: Nodes, kernel_size: Union[int, Tuple[int, int]],
                 stride: Union[int, Tuple[int, int]], n_filters: int,
                 nu: Optional[Union[float, Tuple[float, float]]] = None, weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Instantiates a ``LocallyConnectedConnection`` object. Source population should be two-dimensional.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param n_filters: Number of locally connected filters per pre-synaptic region.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        :param Tuple[int, int] input_shape: Shape of input population if it's not ``[sqrt, sqrt]``.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters

        shape = kwargs.get('input_shape', None)
        if shape is None:
            sqrt = int(np.sqrt(source.n))
            shape = _pair(sqrt)

        if kernel_size == shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((shape[0] - kernel_size[0]) / stride[0]) + 1, int((shape[1] - kernel_size[1]) / stride[1]) + 1
            )

        self.conv_size = conv_size

        conv_prod = int(np.prod(conv_size))
        kernel_prod = int(np.prod(kernel_size))

        assert target.n == n_filters * conv_prod, 'Target layer size must be n_filters * (kernel_size ** 2).'

        locations = torch.zeros(kernel_size[0], kernel_size[1], conv_size[0], conv_size[1]).long()
        for c1 in range(conv_size[0]):
            for c2 in range(conv_size[1]):
                for k1 in range(kernel_size[0]):
                    for k2 in range(kernel_size[1]):
                        location = c1 * stride[0] * shape[1] + c2 * stride[1] + k1 * shape[0] + k2
                        locations[k1, k2, c1, c2] = location

        self.locations = locations.view(kernel_prod, conv_prod)
        self.w = kwargs.get('w', None)

        if self.w is None:
            self.w = torch.zeros(source.n, target.n)
            for f in range(n_filters):
                for c in range(conv_prod):
                    for k in range(kernel_prod):
                        if self.wmin == -np.inf or self.wmax == np.inf:
                            self.w[self.locations[k, c], f * conv_prod + c] = np.random.rand()
                        else:
                            self.w[self.locations[k, c], f * conv_prod + c] = \
                                self.wmin + np.random.rand() * (self.wmax - self.wmin)
        else:
            if self.wmin is not None and self.wmax is not None:
                self.w = torch.clamp(self.w, self.wmin, self.wmax)

        self.mask = self.w == 0

        if self.norm is not None:
            self.norm *= kernel_prod

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or with decaying spike activation).
        """
        # Decaying spike activation from previous iteration.
        self.a_pre = self.a_pre * self.decay + s.float().view(-1)

        # Compute multiplication of pre-activations by connection weights.
        if self.w.shape[0] == self.source.n and self.w.shape[1] == self.target.n:
            return self.a_pre @ self.w
        else:
            a_post = self.a_pre @ self.w.view(self.source.n, self.target.n)
            return a_post.view(*self.target.shape)

    def update(self, dt, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        if kwargs['mask'] is None:
            kwargs['mask'] = self.mask

        super().update(dt=dt, **kwargs)

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


class MeanFieldConnection(AbstractConnection):
    # language=rst
    """
    A connection between one or two populations of neurons which computes a summary of the pre-synaptic population to
    use as weighted input to the post-synaptic population.
    """

    def __init__(self, source: Nodes, target: Nodes, nu: Optional[Union[float, Tuple[float, float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Instantiates a :code:`MeanFieldConnection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param function update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.w = kwargs.get('w', None)

        if self.w is None:
            self.w = (torch.randn(1)[0] + 1) / 10
        else:
            if self.wmin is not None and self.wmax is not None:
                self.w = torch.clamp(self.w, self.wmin, self.wmax)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or with decaying spike activation).
        """
        # Decaying spike activation from previous iteration.
        self.a_pre = self.a_pre * self.decay + s.float().mean()

        # Compute multiplication of mean-field pre-activation by connection weights.
        return self.a_pre * self.w

    def update(self, dt, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(dt=dt, **kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights so each target neuron has sum of connection weights equal to ``self.norm``.
        """
        if self.norm is not None:
            self.w = self.w.view(1, self.target.n)
            self.w *= self.norm / self.w.sum()
            self.w = self.w.view(1, *self.target.shape)

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
                 weight_decay: float = None, **kwargs) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object with sparse weights.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param nu_pre: Learning rate for pre-synaptic events.
        :param nu_post: Learning rate for post-synpatic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param torch.Tensor w: Strengths of synapses.
        :param float sparsity: Fraction of sparse connections to use.
        :param function update_rule: Modifies connection parameters according to some rule.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

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
        s = s.float().view(-1, 1)
        a = self.w.t().mm(s)
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
