from abc import ABC, abstractmethod
from typing import Union, Tuple, Optional, Sequence

import numpy as np
import torch
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from .nodes import Nodes


class AbstractConnection(ABC, Module):
    # language=rst
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param float wmin: The minimum value on the connection weights.
        :param float wmax: The maximum value on the connection weights.
        :param float norm: Total weight per target neuron normalization.
        :param ByteTensor norm_by_max: Normalize the weight of a neuron by its max weight.
        :param ByteTensor norm_by_max_with_shadow_weights: Normalize the weight of a neuron by its max weight by
                                                                original weights
        """
        super().__init__()

        assert isinstance(source, Nodes), "Source is not a Nodes object"
        assert isinstance(target, Nodes), "Target is not a Nodes object"

        self.source = source
        self.target = target

        self.nu = nu
        self.weight_decay = weight_decay

        from ..learning import NoOp

        self.update_rule = kwargs.get("update_rule", NoOp)
        self.wmin = kwargs.get("wmin", -np.inf)
        self.wmax = kwargs.get("wmax", np.inf)
        self.norm = kwargs.get("norm", None)
        self.decay = kwargs.get("decay", None)
        self.norm_by_max = kwargs.get("norm_by_max", False)
        self.norm_by_max_from_shadow_weights = kwargs.get(
            "norm_by_max_from_shadow_weights", False
        )

        if self.update_rule is None:
            self.update_rule = NoOp

        self.update_rule = self.update_rule(
            connection=self, nu=self.nu, weight_decay=weight_decay, **kwargs
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

        Keyword arguments:

        :param bool learning: Whether to allow connection updates.
        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        learning = kwargs.get("learning", True)

        if learning:
            self.update_rule.update(**kwargs)

        mask = kwargs.get("mask", None)
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

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        :param ByteTensor norm_by_max: Normalize the weight of a neuron by its max weight.
        :param ByteTensor norm_by_max_with_shadow_weights: Normalize the weight of a neuron by its max weight by
                                                           original weights.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(
                    torch.rand(source.n, target.n), self.wmin, self.wmax
                )
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (
                    self.wmax - self.wmin
                )
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, False)

        self.b = Parameter(kwargs.get("b",
            torch.zeros(target.n)), False)

        if self.norm_by_max_from_shadow_weights:
            self.shadow_w = self.w.clone().detach()
            self.prev_w = self.w.clone().detach()

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        # Compute multiplication of spike activations by connection weights and add bias.
        post = s.float().view(-1) @ self.w + self.b
        return post.view(*self.target.shape)

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
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def normalize_by_max(self) -> None:
        # language=rst
        """
        Normalize weights by the max weight of the target neuron.
        """
        if self.norm_by_max:
            w_max = self.w.abs().max(0)[0]
            w_max[w_max == 0] = 1.0
            self.w /= w_max

    def normalize_by_max_from_shadow_weights(self) -> None:
        # language=rst
        """
        Normalize weights by the max weight of the target neuron.
        """
        if self.norm_by_max_from_shadow_weights:
            self.shadow_w += self.w - self.prev_w
            w_max = self.shadow_w.abs().max(0)[0]
            w_max[w_max == 0] = 1.0
            self.w = self.shadow_w / w_max
            self.prev_w = self.w.clone().detach()

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

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
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

        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_channels, input_height, input_width = (
            source.shape[1],
            source.shape[2],
            source.shape[3],
        )
        self.out_channels, output_height, output_width = (
            target.shape[1],
            target.shape[2],
            target.shape[3],
        )

        width = (
            input_height - self.kernel_size[0] + 2 * self.padding[0]
        ) / self.stride[0] + 1
        height = (
            input_width - self.kernel_size[1] + 2 * self.padding[1]
        ) / self.stride[1] + 1
        shape = (self.in_channels, self.out_channels, int(width), int(height))

        error = (
            "Target dimensionality must be (out_channels, ?,"
            "(input_height - filter_height + 2 * padding_height) / stride_height + 1,"
            "(input_width - filter_width + 2 * padding_width) / stride_width + 1"
        )

        assert (
            target.shape[1] == shape[1]
            and target.shape[2] == shape[2]
            and target.shape[3] == shape[3]
        ), error

        w = kwargs.get("w", None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp(
                    torch.rand(self.out_channels, self.in_channels, *self.kernel_size),
                    self.wmin,
                    self.wmax,
                )
            else:
                w = (self.wmax - self.wmin) * torch.rand(
                    self.out_channels, self.in_channels, *self.kernel_size
                )
                w += self.wmin
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, False)

        self.b = Parameter(kwargs.get("b",
            torch.zeros(self.out_channels)), False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        return F.conv2d(
            s.float(),
            self.w,
            self.b,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

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
            # get a view and modify in place
            w = self.w.view(
                self.w.size(0) * self.w.size(1), self.w.size(2) * self.w.size(3)
            )

            for fltr in range(w.size(0)):
                w[fltr] *= self.norm / w[fltr].sum(0)

    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_()


class MaxPool2dConnection(AbstractConnection):
    # language=rst
    """
    Specifies max-pooling synapses between one or two populations of neurons by keeping online estimates of maximally
    firing neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        nu: Optional[Union[float, Tuple[float, float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a ``MaxPool2dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param padding: Horizontal and vertical padding for convolution.
        :param dilation: Horizontal and vertical dilation for convolution.

        Keyword arguments:

        :param decay: Decay rate of online estimates of average firing activity.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.register_buffer('firing_rates', torch.ones(source.shape))

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute max-pool pre-activations given spikes using online firing rate estimates.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        self.firing_rates -= self.decay * self.firing_rates
        self.firing_rates += s.float()

        _, indices = F.max_pool2d(
            self.firing_rates,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=True,
        )

        return s.take(indices).float()

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        No weights -> no normalization.
        """
        pass

    def reset_(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_()

        self.firing_rates = torch.zeros(self.source.shape)


class LocallyConnectedConnection(AbstractConnection):
    # language=rst
    """
    Specifies a locally connected connection between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_filters: int,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a ``LocallyConnectedConnection`` object. Source population should be two-dimensional.

        Neurons in the post-synaptic population are ordered by receptive field; that is, if there are ``n_conv`` neurons
        in each post-synaptic patch, then the first ``n_conv`` neurons in the post-synaptic population correspond to the
        first receptive field, the second ``n_conv`` to the second receptive field, and so on.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param n_filters: Number of locally connected filters per pre-synaptic region.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
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

        shape = kwargs.get("input_shape", None)
        if shape is None:
            sqrt = int(np.sqrt(source.n))
            shape = _pair(sqrt)

        if kernel_size == shape:
            conv_size = [1, 1]
        else:
            conv_size = (
                int((shape[0] - kernel_size[0]) / stride[0]) + 1,
                int((shape[1] - kernel_size[1]) / stride[1]) + 1,
            )

        self.conv_size = conv_size

        conv_prod = int(np.prod(conv_size))
        kernel_prod = int(np.prod(kernel_size))

        assert (
            target.n == n_filters * conv_prod
        ), "Target layer size must be n_filters * (kernel_size ** 2)."

        locations = torch.zeros(
            kernel_size[0], kernel_size[1], conv_size[0], conv_size[1]
        ).long()
        for c1 in range(conv_size[0]):
            for c2 in range(conv_size[1]):
                for k1 in range(kernel_size[0]):
                    for k2 in range(kernel_size[1]):
                        location = (
                            c1 * stride[0] * shape[1]
                            + c2 * stride[1]
                            + k1 * shape[0]
                            + k2
                        )
                        locations[k1, k2, c1, c2] = location

        self.register_buffer('locations', locations.view(kernel_prod, conv_prod))
        w = kwargs.get("w", None)

        if w is None:
            w = torch.zeros(source.n, target.n)
            for f in range(n_filters):
                for c in range(conv_prod):
                    for k in range(kernel_prod):
                        if self.wmin == -np.inf or self.wmax == np.inf:
                            w[self.locations[k, c], f * conv_prod + c] = np.clip(
                                np.random.rand(), self.wmin, self.wmax
                            )
                        else:
                            w[
                                self.locations[k, c], f * conv_prod + c
                            ] = self.wmin + np.random.rand() * (self.wmax - wmin)
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, False)

        self.register_buffer('mask', self.w == 0)

        self.b = Parameter(kwargs.get("b",
            torch.zeros(target.n)), False)

        if self.norm is not None:
            self.norm *= kernel_prod

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        # Compute multiplication of pre-activations by connection weights.
        if self.w.shape[0] == self.source.n and self.w.shape[1] == self.target.n:
            return s.float().view(-1) @ self.w + self.b
        else:
            a_post = (
                s.float().view(-1) @ self.w.view(self.source.n, self.target.n) + self.b
            )
            return a_post.view(*self.target.shape)

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.

        Keyword arguments:

        :param ByteTensor mask: Boolean mask determining which weights to clamp to zero.
        """
        if kwargs["mask"] is None:
            kwargs["mask"] = self.mask

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


class MeanFieldConnection(AbstractConnection):
    # language=rst
    """
    A connection between one or two populations of neurons which computes a summary of the pre-synaptic population to
    use as weighted input to the post-synaptic population.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = 0.0,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`MeanFieldConnection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if self.wmin == -np.inf or self.wmax == np.inf:
                w = torch.clamp((torch.randn(1)[0] + 1) / 10, self.wmin, self.wmax)
            else:
                w = self.wmin + ((torch.randn(1)[0] + 1) / 10) * (
                    self.wmax - self.wmin
                )
        else:
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        # Compute multiplication of mean-field pre-activation by connection weights.
        return s.float().mean() * self.w

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

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float]]] = None,
        weight_decay: float = None,
        **kwargs
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object with sparse weights.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param nu: Learning rate for both pre- and post-synaptic events.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param torch.Tensor w: Strengths of synapses.
        :param float sparsity: Fraction of sparse connections to use.
        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        self.sparsity = kwargs.get("sparsity", None)

        assert (
            w is not None
            and self.sparsity is None
            or w is None
            and self.sparsity is not None
        ), 'Only one of "weights" or "sparsity" must be specified'

        if w is None and self.sparsity is not None:
            i = torch.bernoulli(
                1 - self.sparsity * torch.ones(*source.shape, *target.shape)
            )
            if self.wmin == -np.inf or self.wmax == np.inf:
                v = torch.clamp(
                    torch.rand(*source.shape, *target.shape)[i.byte()],
                    self.wmin,
                    self.wmax,
                )
            else:
                v = self.wmin + torch.rand(*source.shape, *target.shape)[i.byte()] * (
                    self.wmax - self.wmin
                )
            w = torch.sparse.FloatTensor(i.nonzero().t(), v)
        elif w is not None and self.sparsity is None:
            assert (
                w.is_sparse
            ), "Weight matrix is not sparse (see torch.sparse module)"
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without decaying spike activation).
        """
        return torch.mm(self.w, s.unsqueeze(-1).float()).squeeze(-1)

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
