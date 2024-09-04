from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from bindsnet.utils import im2col_indices
from torch.nn import Module, Parameter
from torch.nn.modules.utils import _pair, _triple

from bindsnet.network.nodes import CSRMNodes, Nodes


class AbstractConnection(ABC, Module):
    # language=rst
    """
    Abstract base method for connections between ``Nodes``.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for abstract base class for connection objects.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Maximum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization.
        :param Union[bool, torch.Tensor] Dales_rule: Whether to enforce Dale's rule. input is boolean tensor in weight shape
            where True means force zero or positive values and False means force zero or negative values.
        """
        super().__init__()

        assert isinstance(source, Nodes), "Source is not a Nodes object"
        assert isinstance(target, Nodes), "Target is not a Nodes object"

        self.source = source
        self.target = target

        # self.nu = nu
        self.weight_decay = weight_decay
        self.reduction = reduction

        from ..learning import NoOp

        self.update_rule = kwargs.get("update_rule", None)

        # Float32 necessary for comparisons with +/-inf
        self.wmin = Parameter(
            torch.as_tensor(kwargs.get("wmin", -np.inf), dtype=torch.float32),
            requires_grad=False,
        )
        self.wmax = Parameter(
            torch.as_tensor(kwargs.get("wmax", np.inf), dtype=torch.float32),
            requires_grad=False,
        )
        self.norm = kwargs.get("norm", None)
        self.decay = kwargs.get("decay", None)

        if self.update_rule is None:
            self.update_rule = NoOp

        self.update_rule = self.update_rule(
            connection=self,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        self.Dales_rule = kwargs.get("Dales_rule", None)
        if self.Dales_rule is not None:
            self.Dales_rule = Parameter(
                torch.as_tensor(self.Dales_rule, dtype=torch.bool), requires_grad=False
            )

    @abstractmethod
    def compute(self, s: torch.Tensor) -> None:
        # language=rst
        """
        Compute pre-activations of downstream neurons given spikes of upstream neurons.

        :param s: Incoming spikes.
        """

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

        if self.Dales_rule is not None:
            # weight that are negative and should be positive are set to 0
            self.w[self.w < 0 * self.Dales_rule.to(torch.float)] = 0
            # weight that are positive and should be negative are set to 0
            self.w[self.w > 0 * 1 - self.Dales_rule.to(torch.float)] = 0

    @abstractmethod
    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """


class Connection(AbstractConnection):
    # language=rst
    """
    Specifies synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(torch.rand(source.n, target.n), self.wmin, self.wmax)
            else:
                w = self.wmin + torch.rand(source.n, target.n) * (self.wmax - self.wmin)
        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(torch.as_tensor(w), self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

        b = kwargs.get("b", None)
        if b is not None:
            self.b = Parameter(b, requires_grad=False)
        else:
            self.b = None

        if isinstance(self.target, CSRMNodes):
            self.s_w = None

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using connection weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
                 decaying spike activation).
        """
        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = s.view(s.size(0), -1).float() @ self.w
        else:
            post = s.view(s.size(0), -1).float() @ self.w + self.b
        return post.view(s.size(0), *self.target.shape)

    def compute_window(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """ """

        if self.s_w == None:
            # Construct a matrix of shape batch size * window size * dimension of layer
            self.s_w = torch.zeros(
                self.target.batch_size, self.target.res_window_size, *self.source.shape
            )

        # Add the spike vector into the first in first out matrix of windowed (res) spike trains
        self.s_w = torch.cat((self.s_w[:, 1:, :], s[:, None, :]), 1)

        # Compute multiplication of spike activations by weights and add bias.
        if self.b is None:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
            )
        else:
            post = (
                self.s_w.view(self.s_w.size(0), self.s_w.size(1), -1).float() @ self.w
                + self.b
            )

        return post.view(
            self.s_w.size(0), self.target.res_window_size, *self.target.shape
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
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w_abs_sum = self.w.abs().sum(0).unsqueeze(0)
            w_abs_sum[w_abs_sum == 0] = 1.0
            self.w *= self.norm / w_abs_sum

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class Conv1dConnection(AbstractConnection):
    # language=rst
    """
    Specifies one-dimensional convolutional synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a ``Conv1dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: the size of 1-D convolutional kernel.
        :param stride: stride for convolution.
        :param padding: padding for convolution.
        :param dilation: dilation for convolution.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Maximum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        if dilation != 1:
            raise NotImplementedError(
                "Dilation is not currently supported for 1-D spiking convolution."
            )

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.in_channels, input_size = (source.shape[0], source.shape[1])
        self.out_channels, output_size = (target.shape[0], target.shape[1])

        conv_size = (input_size - self.kernel_size + 2 * self.padding) / self.stride + 1
        shape = (self.in_channels, self.out_channels, int(conv_size))

        error = (
            "Target dimensionality must be (out_channels, ?,"
            "(input_size - filter_size + 2 * padding) / stride + 1,"
        )

        assert target.shape[0] == shape[1] and target.shape[1] == shape[2], error

        w = kwargs.get("w", None)
        inf = torch.tensor(np.inf)
        if w is None:
            if (self.wmin == -inf).any() or (self.wmax == inf).any():
                w = torch.clamp(
                    torch.rand(self.out_channels, self.in_channels, self.kernel_size),
                    self.wmin,
                    self.wmax,
                )
            else:
                w = (self.wmax - self.wmin) * torch.rand(
                    self.out_channels, self.in_channels, self.kernel_size
                )
                w += self.wmin
        else:
            if (self.wmin == -inf).any() or (self.wmax == inf).any():
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(
            kwargs.get("b", torch.zeros(self.out_channels)), requires_grad=False
        )

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        return F.conv1d(
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
        Normalize weights along the first axis according to total weight per target
        neuron.
        """
        if self.norm is not None:
            # get a view and modify in place
            w = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])

            for fltr in range(w.shape[0]):
                w[fltr] *= self.norm / w[fltr].sum(0)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class Conv2dConnection(AbstractConnection):
    # language=rst
    """
    Specifies two-dimensional convolutional synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
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
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Maximum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.in_channels, input_height, input_width = (
            source.shape[0],
            source.shape[1],
            source.shape[2],
        )
        self.out_channels, output_height, output_width = (
            target.shape[0],
            target.shape[1],
            target.shape[2],
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
            target.shape[0] == shape[1]
            and target.shape[1] == shape[2]
            and target.shape[2] == shape[3]
        ), error

        w = kwargs.get("w", None)
        inf = torch.tensor(np.inf)
        if w is None:
            if (self.wmin == -inf).any() or (self.wmax == inf).any():
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
            if (self.wmin == -inf).any() or (self.wmax == inf).any():
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(
            kwargs.get("b", torch.zeros(self.out_channels)), requires_grad=False
        )

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
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
        Normalize weights along the first axis according to total weight per target
        neuron.
        """
        if self.norm is not None:
            # get a view and modify in place
            w = self.w.view(
                self.w.shape[0] * self.w.shape[1], self.w.shape[2] * self.w.shape[3]
            )

            for fltr in range(w.shape[0]):
                w[fltr] *= self.norm / w[fltr].sum(0)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class Conv3dConnection(AbstractConnection):
    # language=rst
    """
    Specifies three-dimensional convolutional synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a ``Conv3dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Depth-wise, horizontal, and vertical  size of convolutional kernels.
        :param stride: Depth-wise, horizontal, and vertical stride for convolution.
        :param padding: Depth-wise, horizontal, and vertical  padding for convolution.
        :param dilation: Depth-wise, horizontal and vertical dilation for convolution.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Maximum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        if dilation != 1 and dilation != (1, 1, 1):
            raise NotImplementedError(
                "Dilation is not currently supported for 3-D spiking convolution."
            )

        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)

        self.in_channels, input_depth, input_height, input_width = (
            source.shape[0],
            source.shape[1],
            source.shape[2],
            source.shape[3],
        )
        self.out_channels, output_depth, output_height, output_width = (
            target.shape[0],
            target.shape[1],
            target.shape[2],
            target.shape[3],
        )

        depth = (input_depth - self.kernel_size[0] + 2 * self.padding[0]) / self.stride[
            0
        ] + 1
        width = (
            input_height - self.kernel_size[1] + 2 * self.padding[1]
        ) / self.stride[1] + 1
        height = (
            input_width - self.kernel_size[2] + 2 * self.padding[2]
        ) / self.stride[2] + 1

        shape = (
            self.in_channels,
            self.out_channels,
            int(depth),
            int(width),
            int(height),
        )

        error = (
            "Target dimensionality must be (out_channels, ?,"
            "(input_depth - filter_depth + 2 * padding_depth) / stride_depth + 1,"
            "(input_height - filter_height + 2 * padding_height) / stride_height + 1,"
            "(input_width - filter_width + 2 * padding_width) / stride_width + 1"
        )

        assert (
            target.shape[0] == shape[1]
            and target.shape[1] == shape[2]
            and target.shape[2] == shape[3]
            and target.shape[3] == shape[4]
        ), error

        w = kwargs.get("w", None)
        inf = torch.tensor(np.inf)
        if w is None:
            if (self.wmin == -inf).any() or (self.wmax == inf).any():
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
            if (self.wmin == -inf).any() or (self.wmax == inf).any():
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(
            kwargs.get("b", torch.zeros(self.out_channels)), requires_grad=False
        )

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        return F.conv3d(
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
        Normalize weights along the first axis according to total weight per target
        neuron.
        """
        if self.norm is not None:
            # get a view and modify in place
            w = self.w.view(
                self.w.shape[0] * self.w.shape[1],
                self.w.shape[2] * self.w.shape[3] * self.w.shape[4],
            )

            for fltr in range(w.shape[0]):
                w[fltr] *= self.norm / w[fltr].sum(0)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class MaxPool1dConnection(AbstractConnection):
    # language=rst
    """
    Specifies max-pooling synapses between one or two populations of neurons by keeping
    online estimates of maximally firing neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a ``MaxPool1dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: the size of 1-D convolutional kernel.
        :param stride: stride for convolution.
        :param padding: padding for convolution.
        :param dilation: dilation for convolution.

        Keyword arguments:

        :param decay: Decay rate of online estimates of average firing activity.
        """
        super().__init__(source, target, None, None, 0.0, **kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        self.register_buffer("firing_rates", torch.zeros(source.s.shape))

        from ..learning import NoOp

        # Initialize learning rule
        if self.update_rule is not None and (self.update_rule == NoOp):
            self.update_rule = self.update_rule()

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute max-pool pre-activations given spikes using online firing rate
        estimates.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        self.firing_rates -= self.decay * self.firing_rates
        self.firing_rates += s.float().squeeze()

        _, indices = F.max_pool1d(
            self.firing_rates,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=True,
        )

        return s.flatten(2).gather(2, indices.flatten(2)).view_as(indices).float()

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

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.firing_rates = torch.zeros(
            self.source.batch_size, *(self.source.s.shape[1:])
        )


class MaxPool2dConnection(AbstractConnection):
    # language=rst
    """
    Specifies max-pooling synapses between one or two populations of neurons by keeping
    online estimates of maximally firing neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        **kwargs,
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
        super().__init__(source, target, None, None, 0.0, **kwargs)

        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = _pair(padding)
        self.dilation = _pair(dilation)

        self.register_buffer("firing_rates", torch.zeros(source.s.shape))

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute max-pool pre-activations given spikes using online firing rate
        estimates.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        self.firing_rates -= self.decay * self.firing_rates
        self.firing_rates += s.float().squeeze()

        _, indices = F.max_pool2d(
            self.firing_rates,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=True,
        )

        return s.flatten(2).gather(2, indices.flatten(2)).view_as(indices).float()

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

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.firing_rates = torch.zeros(
            self.source.batch_size, *(self.source.s.shape[1:])
        )


class MaxPoo3dConnection(AbstractConnection):
    # language=rst
    """
    Specifies max-pooling synapses between one or two populations of neurons by keeping
    online estimates of maximally firing neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]] = 1,
        padding: Union[int, Tuple[int, int, int]] = 0,
        dilation: Union[int, Tuple[int, int, int]] = 1,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a ``MaxPool3dConnection`` object.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Depth-wise, horizontal and vertical size of convolutional kernels.
        :param stride: Depth-wise, horizontal and vertical stride for convolution.
        :param padding: Depth-wise, horizontal and vertical padding for convolution.
        :param dilation: Depth-wise, horizontal and vertical dilation for convolution.

        Keyword arguments:

        :param decay: Decay rate of online estimates of average firing activity.
        """
        super().__init__(source, target, None, None, 0.0, **kwargs)

        self.kernel_size = _triple(kernel_size)
        self.stride = _triple(stride)
        self.padding = _triple(padding)
        self.dilation = _triple(dilation)

        self.register_buffer("firing_rates", torch.zeros(source.s.shape))

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute max-pool pre-activations given spikes using online firing rate
        estimates.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        self.firing_rates -= self.decay * self.firing_rates
        self.firing_rates += s.float().squeeze()

        _, indices = F.max_pool3d(
            self.firing_rates,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            return_indices=True,
        )

        return s.flatten(2).gather(2, indices.flatten(2)).view_as(indices).float()

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

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.firing_rates = torch.zeros(
            self.source.batch_size, *(self.source.s.shape[1:])
        )


class LocalConnection(AbstractConnection):
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
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a ``LocalConnection2D`` object. Source population should have
        square size

        Neurons in the post-synaptic population are ordered by receptive field; that is,
        if there are ``n_conv`` neurons in each post-synaptic patch, then the first
        ``n_conv`` neurons in the post-synaptic population correspond to the first
        receptive field, the second ``n_conv`` to the second receptive field, and so on.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param n_filters: Number of locally connected filters per pre-synaptic region.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Maximum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        :param Tuple[int, int] input_shape: Shape of input population if it's not
            ``[sqrt, sqrt]``.
        """

        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

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

        assert target.n == n_filters * conv_prod, (
            f"Total neurons in target layer must be {n_filters * conv_prod}. "
            f"Got {target.n}."
        )

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

        self.register_buffer("locations", locations.view(kernel_prod, conv_prod))
        w = kwargs.get("w", None)

        if w is None:
            # Calculate unbounded weights
            w = torch.zeros(source.n, target.n)
            for f in range(n_filters):
                for c in range(conv_prod):
                    for k in range(kernel_prod):
                        w[self.locations[k, c], f * conv_prod + c] = np.random.rand()

            # Bind weights to given range
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(w, self.wmin, self.wmax)
            else:
                w = self.wmin + w * (self.wmax - self.wmin)

        else:
            if (self.wmin != -np.inf).any() or (self.wmax != np.inf).any():
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

        self.register_buffer("mask", self.w == 0)

        self.b = Parameter(kwargs.get("b", torch.zeros(target.n)), requires_grad=False)

        if self.norm is not None:
            self.norm *= kernel_prod

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        # Compute multiplication of pre-activations by connection weights.
        a_post = (
            s.float().view(s.size(0), -1) @ self.w.view(self.source.n, self.target.n)
            + self.b
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
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            w = self.w.view(self.source.n, self.target.n)
            w *= self.norm / self.w.sum(0).view(1, -1)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class LocalConnection1D(AbstractConnection):
    """
    Specifies a one-dimensional local connection between one or two population of neurons supporting multi-channel inputs with shape (C, H);
    The logic is different from the original LocalConnection implementation (where masks were used with normal dense connections).
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: int,
        stride: int,
        n_filters: int,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Instantiates a 'LocalConnection1D` object. Source population can be multi-channel.
        Neurons in the post-synaptic population are ordered by receptive field, i.e.,
        if there are `n_conv` neurons in each post-synaptic patch, then the first
        `n_conv` neurons in the post-synaptic population correspond to the first
        receptive field, the second ``n_conv`` to the second receptive field, and so on.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: size of convolutional kernels.
        :param stride: stride for convolution.
        :param n_filters: Number of locally connected filters per pre-synaptic region.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        Keyword arguments:
        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """

        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters

        self.in_channels, input_height = (source.shape[0], source.shape[1])

        height = int((input_height - self.kernel_size) / self.stride) + 1

        self.conv_size = height

        w = kwargs.get("w", None)

        error = (
            "Target dimensionality must be (in_channels,"
            "n_filters*conv_size,"
            "kernel_size)"
        )

        if w is None:
            w = torch.rand(
                self.in_channels, self.n_filters * self.conv_size, self.kernel_size
            )
        else:
            assert w.shape == (
                self.in_channels,
                self.out_channels * self.conv_size,
                self.kernel_size,
            ), error

        if self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", None), requires_grad=False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using layer weights.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        # Compute multiplication of pre-activations by connection weights
        # s: batch, ch_in, h_in => s_unfold: batch, ch_in, ch_out * h_out, k
        # w: ch_in, ch_out * h_out, k
        # a_post: batch, ch_in, ch_out * h_out, k => batch, ch_out * h_out (= target.n)

        batch_size = s.shape[0]

        self.s_unfold = (
            s.unfold(-1, self.kernel_size, self.stride)
            .reshape(s.shape[0], self.in_channels, self.conv_size, self.kernel_size)
            .repeat(1, 1, self.n_filters, 1)
        )

        a_post = self.s_unfold.to(self.w.device) * self.w

        return a_post.sum(-1).sum(1).view(batch_size, *self.target.shape)

    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            # get a view and modify in-place
            # w: ch_in, ch_out * h_out, k
            w = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])

            for fltr in range(w.shape[0]):
                w[fltr, :] *= self.norm / w[fltr, :].sum(0)

    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.target.reset_state_variables()


class LocalConnection2D(AbstractConnection):
    """
    Specifies a two-dimensional local connection between one or two population of neurons supporting multi-channel inputs with shape (C, H, W);
    The logic is different from the original LocalConnection implementation (where masks were used with normal dense connections)
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]],
        n_filters: int,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Instantiates a 'LocalConnection2D` object. Source population can be multi-channel.
        Neurons in the post-synaptic population are ordered by receptive field, i.e.,
        if there are `n_conv` neurons in each post-synaptic patch, then the first
        `n_conv` neurons in the post-synaptic population correspond to the first
        receptive field, the second ``n_conv`` to the second receptive field, and so on.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal and vertical size of convolutional kernels.
        :param stride: Horizontal and vertical stride for convolution.
        :param n_filters: Number of locally connected filters per pre-synaptic region.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        Keyword arguments:
        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """

        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        kernel_size = _pair(kernel_size)
        stride = _pair(stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters

        self.in_channels, input_height, input_width = (
            source.shape[0],
            source.shape[1],
            source.shape[2],
        )

        height = int((input_height - self.kernel_size[0]) / self.stride[0]) + 1
        width = int((input_width - self.kernel_size[1]) / self.stride[1]) + 1

        self.conv_size = (height, width)
        self.conv_prod = int(np.prod(self.conv_size))
        self.kernel_prod = int(np.prod(kernel_size))

        w = kwargs.get("w", None)

        error = (
            "Target dimensionality must be (in_channels,"
            "n_filters*conv_prod,"
            "kernel_prod)"
        )

        if w is None:
            w = torch.rand(
                self.in_channels, self.n_filters * self.conv_prod, self.kernel_prod
            )
        else:
            assert w.shape == (
                self.in_channels,
                self.out_channels * self.conv_prod,
                self.kernel_prod,
            ), error

        if self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", None), requires_grad=False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using layer weights.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        # Compute multiplication of pre-activations by connection weights
        # s: batch, ch_in, w_in, h_in => s_unfold: batch, ch_in, ch_out * w_out * h_out, k1*k2
        # w: ch_in, ch_out * w_out * h_out, k1*k2
        # a_post: batch, ch_in, ch_out * w_out * h_out, k1*k2 => batch, ch_out * w_out * h_out (= target.n)

        batch_size = s.shape[0]

        self.s_unfold = (
            s.unfold(-2, self.kernel_size[0], self.stride[0])
            .unfold(-2, self.kernel_size[1], self.stride[1])
            .reshape(s.shape[0], self.in_channels, self.conv_prod, self.kernel_prod)
            .repeat(1, 1, self.n_filters, 1)
        )

        a_post = self.s_unfold.to(self.w.device) * self.w

        return a_post.sum(-1).sum(1).view(batch_size, *self.target.shape)

    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            # get a view and modify in-place
            # w: ch_in, ch_out * w_out * h_out, k1 * k2
            w = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])

            for fltr in range(w.shape[0]):
                w[fltr, :] *= self.norm / w[fltr, :].sum(0)

    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.target.reset_state_variables()


class LocalConnection3D(AbstractConnection):
    """
    Specifies a three-dimensional local connection between one or two population of neurons supporting multi-channel inputs with shape (C, H, W, D);
    The logic is different from the original LocalConnection implementation (where masks were used with normal dense connections)
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        kernel_size: Union[int, Tuple[int, int, int]],
        stride: Union[int, Tuple[int, int, int]],
        n_filters: int,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        """
        Instantiates a 'LocalConnection3D` object. Source population can be multi-channel.
        Neurons in the post-synaptic population are ordered by receptive field, i.e.,
        if there are `n_conv` neurons in each post-synaptic patch, then the first
        `n_conv` neurons in the post-synaptic population correspond to the first
        receptive field, the second ``n_conv`` to the second receptive field, and so on.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
        :param kernel_size: Horizontal, vertical, and depth-wise size of convolutional kernels.
        :param stride: Horizontal, vertical, and depth-wise stride for convolution.
        :param n_filters: Number of locally connected filters per pre-synaptic region.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        Keyword arguments:
        :param LearningRule update_rule: Modifies connection parameters according to some rule.
        :param torch.Tensor w: Strengths of synapses.
        :param torch.Tensor b: Target population bias.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """

        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

        kernel_size = _triple(kernel_size)
        stride = _triple(stride)

        self.kernel_size = kernel_size
        self.stride = stride
        self.n_filters = n_filters

        self.in_channels, input_height, input_width, input_depth = (
            source.shape[0],
            source.shape[1],
            source.shape[2],
            source.shape[3],
        )

        height = int((input_height - self.kernel_size[0]) / self.stride[0]) + 1
        width = int((input_width - self.kernel_size[1]) / self.stride[1]) + 1
        depth = int((input_depth - self.kernel_size[2]) / self.stride[2]) + 1

        self.conv_size = (height, width, depth)
        self.conv_prod = int(np.prod(self.conv_size))
        self.kernel_prod = int(np.prod(kernel_size))

        w = kwargs.get("w", None)

        error = (
            "Target dimensionality must be (in_channels,"
            "n_filters*conv_prod,"
            "kernel_prod)"
        )

        if w is None:
            w = torch.rand(
                self.in_channels, self.n_filters * self.conv_prod, self.kernel_prod
            )
        else:
            assert w.shape == (
                self.in_channels,
                self.out_channels * self.conv_prod,
                self.kernel_prod,
            ), error

        if self.wmin != -np.inf or self.wmax != np.inf:
            w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)
        self.b = Parameter(kwargs.get("b", None), requires_grad=False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        """
        Compute pre-activations given spikes using layer weights.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        # Compute multiplication of pre-activations by connection weights
        # s: batch, ch_in, w_in, h_in, d_in => s_unfold: batch, ch_in, ch_out * w_out * h_out * d_out, k1*k2*k3
        # w: ch_in, ch_out * w_out * h_out * d_out, k1*k2*k3
        # a_post: batch, ch_in, ch_out * w_out * h_out * d_out, k1*k2*k3 => batch, ch_out * w_out * h_out * d_out (= target.n)

        batch_size = s.shape[0]

        self.s_unfold = (
            s.unfold(-3, self.kernel_size[0], self.stride[0])
            .unfold(-3, self.kernel_size[1], self.stride[1])
            .unfold(-3, self.kernel_size[2], self.stride[2])
            .reshape(s.shape[0], self.in_channels, self.conv_prod, self.kernel_prod)
            .repeat(1, 1, self.n_filters, 1)
        )

        a_post = self.s_unfold.to(self.w.device) * self.w

        return a_post.sum(-1).sum(1).view(batch_size, *self.target.shape)

    def update(self, **kwargs) -> None:
        """
        Compute connection's update rule.
        """
        super().update(**kwargs)

    def normalize(self) -> None:
        """
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            # get a view and modify in-place
            # w: ch_in, ch_out * w_out * h_out * d_out, k1*k2*k3
            w = self.w.view(self.w.shape[0] * self.w.shape[1], self.w.shape[2])

            for fltr in range(w.shape[0]):
                w[fltr, :] *= self.norm / w[fltr, :].sum(0)

    def reset_state_variables(self) -> None:
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()

        self.target.reset_state_variables()


class MeanFieldConnection(AbstractConnection):
    # language=rst
    """
    A connection between one or two populations of neurons which computes a summary of
    the pre-synaptic population to use as weighted input to the post-synaptic
    population.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`MeanFieldConnection` object.
        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        Keyword arguments:
        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param Union[float, torch.Tensor] w: Strengths of synapses. Can be single value or tensor of size ``target``
        :param Union[float, torch.Tensor] wmin: Minimum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param Union[float, torch.Tensor] wmax: Maximum allowed value(s) on the connection weights. Single value, or
            tensor of same size as w
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, weight_decay, **kwargs)

        w = kwargs.get("w", None)
        if w is None:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp((torch.randn(1)[0] + 1) / 10, self.wmin, self.wmax)
            else:
                w = self.wmin + ((torch.randn(1)[0] + 1) / 10) * (self.wmax - self.wmin)
        else:
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute pre-activations given spikes using layer weights.
        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
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
        Normalize weights so each target neuron has sum of connection weights equal to
        ``self.norm``.
        """
        if self.norm is not None:
            self.w = self.w.view(1, self.target.n)
            self.w *= self.norm / self.w.sum()
            self.w = self.w.view(1, *self.target.shape)

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()


class SparseConnection(AbstractConnection):
    # language=rst
    """
    Specifies sparse synapses between one or two populations of neurons.
    """

    def __init__(
        self,
        source: Nodes,
        target: Nodes,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Connection` object with sparse weights.

        :param source: A layer of nodes from which the connection originates.
        :param target: A layer of nodes to which the connection connects.
         :param nu: Learning rate for both pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param torch.Tensor w: Strengths of synapses. Must be in ``torch.sparse`` format
        :param float sparsity: Fraction of sparse connections to use.
        :param LearningRule update_rule: Modifies connection parameters according to
            some rule.
        :param float wmin: Minimum allowed value on the connection weights.
        :param float wmax: Maximum allowed value on the connection weights.
        :param float norm: Total weight per target neuron normalization constant.
        """
        super().__init__(source, target, nu, reduction, weight_decay, **kwargs)

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
            if (self.wmin == -np.inf).any() or (self.wmax == np.inf).any():
                v = torch.clamp(
                    torch.rand(*source.shape, *target.shape), self.wmin, self.wmax
                )[i.bool()]
            else:
                v = (
                    self.wmin
                    + torch.rand(*source.shape, *target.shape) * (self.wmax - self.wmin)
                )[i.bool()]
            w = torch.sparse.FloatTensor(i.nonzero().t(), v)
        elif w is not None and self.sparsity is None:
            assert w.is_sparse, "Weight matrix is not sparse (see torch.sparse module)"
            if self.wmin != -np.inf or self.wmax != np.inf:
                w = torch.clamp(w, self.wmin, self.wmax)

        self.w = Parameter(w, requires_grad=False)

    def compute(self, s: torch.Tensor) -> torch.Tensor:
        # language=rst
        """
        Compute convolutional pre-activations given spikes using layer weights.

        :param s: Incoming spikes.
        :return: Incoming spikes multiplied by synaptic weights (with or without
            decaying spike activation).
        """
        return torch.mm(self.w, s.view(s.shape[1], 1).float()).squeeze(-1)
        # return torch.mm(self.w, s.unsqueeze(-1).float()).squeeze(-1)

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute connection's update rule.
        """

    def normalize(self) -> None:
        # language=rst
        """
        Normalize weights along the first axis according to total weight per target
        neuron.
        """

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the connection.
        """
        super().reset_state_variables()
