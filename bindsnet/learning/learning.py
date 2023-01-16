import warnings
from abc import ABC
from typing import Optional, Sequence, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from bindsnet.utils import im2col_indices
from ..network.nodes import SRM0Nodes
from ..network.topology import (
    AbstractConnection,
    Connection,
    Conv1dConnection,
    Conv2dConnection,
    Conv3dConnection,
    LocalConnection,
    LocalConnection1D,
    LocalConnection2D,
    LocalConnection3D,
)


class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            self.nu = torch.tensor([0.0, 0.0], dtype=torch.float)
        elif isinstance(nu, (float, int)):
            self.nu = torch.tensor([nu, nu], dtype=torch.float)
        elif all(isinstance(element, (float, int)) for element in nu):
            self.nu = torch.tensor(nu, dtype=torch.float)
        else:
            self.nu = torch.stack(nu, dim=0).to(dtype=torch.float)

        if not self.nu.any() and not isinstance(self, NoOp):
            warnings.warn(
                f"nu is set to zeros for {type(self).__name__} learning rule. "
                + "It will disable the learning process."
            )

        # Parameter update reduction across minibatch dimension.
        if reduction is None:
            if self.source.batch_size == 1:
                self.reduction = torch.squeeze
            else:
                self.reduction = torch.sum
        else:
            self.reduction = reduction

        # Weight decay.
        self.weight_decay = 1.0 - weight_decay if weight_decay else 1.0

    def update(self) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        # Implement weight decay.
        if self.weight_decay:
            self.connection.w *= self.weight_decay

        # Bound weights.
        if (
            (self.connection.wmin != -np.inf).any()
            or (self.connection.wmax != np.inf).any()
        ) and not isinstance(self, NoOp):
            self.connection.w.clamp_(self.connection.wmin, self.connection.wmax)


class NoOp(LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        super().update()


class PostPre(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, LocalConnection1D):
            self.update = self._local_connection1d_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection2d_update
        elif isinstance(connection, LocalConnection3D):
            self.update = self._local_connection3d_update
        elif isinstance(connection, Conv1dConnection):
            self.update = self._conv1d_connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        elif isinstance(connection, Conv3dConnection):
            self.update = self._conv3d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        target_x = self.target.x.reshape(batch_size, out_channels * height_out, 1)
        target_x = target_x * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection2d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_x = target_x * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection3d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection3D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        kernel_depth = self.connection.kernel_size[2]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]
        depth_out = self.connection.conv_size[2]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_x = target_x * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_s = target_s * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_x = (
            self.source.x.unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Pre-synaptic update.
        if self.nu[0].any():
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]
            self.connection.w -= self.reduction(torch.bmm(source_s, target_x), dim=0)
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1].any():
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            self.connection.w += self.reduction(torch.bmm(source_x, target_s), dim=0)
            del source_x, target_s

        super().update()

    def _conv1d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv1dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, in_channels, kernel_size = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(self.source.x, _pair(padding))
        source_x = source_x.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(self.source.s.float(), _pair(padding))
        source_s = source_s.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            # print(self.nu[0].shape, self.connection.w.size())
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv3d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv3dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(
            self.source.x,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_x = (
            source_x.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(
            self.source.s,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_s = (
            source_s.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()
        # print(target_x.shape, source_s.shape, self.connection.w.shape)

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


class WeightDependentPostPre(LearningRule):
    # language=rst
    """
    STDP rule involving both pre- and post-synaptic spiking activity. The post-synaptic
    update is positive and the pre- synaptic update is negative, and both are dependent
    on the magnitude of the synaptic weights.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``WeightDependentPostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``WeightDependentPostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert self.source.traces, "Pre-synaptic nodes must record spike traces."
        assert (connection.wmin != -np.inf).any() and (
            connection.wmax != np.inf
        ).any(), "Connection must define finite wmin and wmax."

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, LocalConnection1D):
            self.update = self._local_connection1d_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection2d_update
        elif isinstance(connection, LocalConnection3D):
            self.update = self._local_connection3d_update
        elif isinstance(connection, Conv1dConnection):
            self.update = self._conv1d_connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        elif isinstance(connection, Conv3dConnection):
            self.update = self._conv3d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        update = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            outer_product = self.reduction(torch.bmm(source_s, target_x), dim=0)
            update -= self.nu[0] * outer_product * (self.connection.w - self.wmin)

        # Post-synaptic update.
        if self.nu[1].any():
            outer_product = self.reduction(torch.bmm(source_x, target_s), dim=0)
            update += self.nu[1] * outer_product * (self.wmax - self.connection.w)

        self.connection.w += update

        super().update()

    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        Weight-dependent post-pre learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        target_x = self.target.x.reshape(batch_size, out_channels * height_out, 1)
        target_x = target_x * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        update = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.w)
            )

        self.connection.w += update

        super().update()

    def _local_connection2d_update(self, **kwargs) -> None:
        # language=rst
        """
        Weight-dependent post-pre learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_x = target_x * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        update = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.w)
            )

        self.connection.w += update

        super().update()

    def _local_connection3d_update(self, **kwargs) -> None:
        # language=rst
        """
        Weight-dependent post-pre learning rule for ``LocalConnection3D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        kernel_depth = self.connection.kernel_size[2]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]
        depth_out = self.connection.conv_size[2]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_x = target_x * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_s = target_s * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_x = (
            self.source.x.unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        update = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )
        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.w)
            )

        self.connection.w += update

        super().update()

    def _conv1d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv1dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (out_channels, in_channels, kernel_size) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(self.source.x, _pair(padding))
        source_x = source_x.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(self.source.s.float(), _pair(padding))
        source_s = source_s.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        update = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.w)
            )

        self.connection.w += update

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        update = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(
                torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0
            )
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(
                torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0
            )
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.w)
            )

        self.connection.w += update

        super().update()

    def _conv3d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv3dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        (
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(
            self.source.x,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_x = (
            source_x.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(
            self.source.s,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_s = (
            source_s.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        update = 0

        # Pre-synaptic update.
        if self.nu[0].any():
            pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
            update -= (
                self.nu[0]
                * pre.view(self.connection.w.size())
                * (self.connection.w - self.wmin)
            )

        # Post-synaptic update.
        if self.nu[1].any():
            post = self.reduction(torch.bmm(target_s, source_x), dim=0)
            update += (
                self.nu[1]
                * post.view(self.connection.w.size())
                * (self.wmax - self.connection.w)
            )

        self.connection.w += update

        super().update()


class Hebbian(LearningRule):
    # language=rst
    """
    Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``Hebbian`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``Hebbian`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        assert (
            self.source.traces and self.target.traces
        ), "Both pre- and post-synaptic nodes must record spike traces."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, LocalConnection1D):
            self.update = self._local_connection1d_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection2d_update
        elif isinstance(connection, LocalConnection3D):
            self.update = self._local_connection3d_update
        elif isinstance(connection, Conv1dConnection):
            self.update = self._conv1d_connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        elif isinstance(connection, Conv3dConnection):
            self.update = self._conv3d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
        source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
        target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
        target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

        # Pre-synaptic update.
        update = self.reduction(torch.bmm(source_s, target_x), dim=0)
        self.connection.w += self.nu[0] * update

        # Post-synaptic update.
        update = self.reduction(torch.bmm(source_x, target_s), dim=0)
        self.connection.w += self.nu[1] * update

        super().update()

    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        target_x = self.target.x.reshape(batch_size, out_channels * height_out, 1)
        target_x = target_x * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection2d_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_x = target_x * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )
        source_x = (
            self.source.x.unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _local_connection3d_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``LocalConnection3D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        kernel_depth = self.connection.kernel_size[2]
        in_channels = self.connection.source.shape[0]
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]
        depth_out = self.connection.conv_size[2]

        target_x = self.target.x.reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_x = target_x * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_s = target_s * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)
        source_x = (
            self.source.x.unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv1d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv1dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        out_channels, in_channels, kernel_size = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(self.source.x, _pair(padding))
        source_x = source_x.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(self.source.s.float(), _pair(padding))
        source_s = source_s.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s.permute((0, 2, 1))), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x.permute((0, 2, 1))), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()

    def _conv3d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.
        """
        (
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride
        batch_size = self.source.batch_size

        # Reshaping spike traces and spike occurrences.
        source_x = F.pad(
            self.source.x,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_x = (
            source_x.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_x = self.target.x.view(batch_size, out_channels, -1)
        source_s = F.pad(
            self.source.s,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_s = (
            source_s.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Pre-synaptic update.
        pre = self.reduction(torch.bmm(target_x, source_s), dim=0)
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = self.reduction(torch.bmm(target_s, source_x), dim=0)
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


class MSTDP(LearningRule):
    # language=rst
    """
    Reward-modulated STDP. Adapted from `(Florian 2007)
    <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv1dConnection):
            self.update = self._conv1d_connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        elif isinstance(connection, Conv3dConnection):
            self.update = self._conv3d_connection_update
        elif isinstance(connection, LocalConnection1D):
            self.update = self._local_connection1d_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection2d_update
        elif isinstance(connection, LocalConnection3D):
            self.update = self._local_connection3d_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                # batch_size, *self.source.shape, device=self.source.s.device
                batch_size,
                self.source.n,
                device=self.source.s.device,
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                # batch_size, *self.target.shape, device=self.target.s.device
                batch_size,
                self.target.n,
                device=self.target.s.device,
            )
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = kwargs.get("a_plus", 1.0)
        if isinstance(a_plus, dict):
            for k, v in a_plus.items():
                a_plus[k] = torch.tensor(v, device=self.connection.w.device)
        else:
            a_plus = torch.tensor(a_plus, device=self.connection.w.device)
        a_minus = kwargs.get("a_minus", -1.0)
        if isinstance(a_minus, dict):
            for k, v in a_minus.items():
                a_minus[k] = torch.tensor(v, device=self.connection.w.device)
        else:
            a_minus = torch.tensor(a_minus, device=self.connection.w.device)

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * self.reduction(update, dim=0)

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            self.p_plus.unsqueeze(2), target_s.unsqueeze(1)
        ) + torch.bmm(source_s.unsqueeze(2), self.p_minus.unsqueeze(1))

        super().update()

    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """

        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.in_channels
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * self.reduction(update, dim=0)

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = (
                self.p_plus.unfold(-1, kernel_height, stride)
                .reshape(batch_size, height_out, in_channels * kernel_height)
                .repeat(1, out_channels, 1)
                .to(self.connection.w.device)
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.reshape(
                batch_size, out_channels * height_out, 1
            )
            self.p_minus = self.p_minus * torch.eye(out_channels * height_out).to(
                self.connection.w.device
            )

        # Reshaping spike occurrences.
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )

        self.eligibility = self.eligibility.view(batch_size, *self.connection.w.shape)

        super().update()

    def _local_connection2d_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        in_channels = self.connection.in_channels
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility

        self.connection.w += self.nu[0] * self.reduction(update, dim=0)

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = (
                self.p_plus.unfold(-2, kernel_height, stride[0])
                .unfold(-2, kernel_width, stride[1])
                .reshape(
                    batch_size,
                    height_out * width_out,
                    in_channels * kernel_height * kernel_width,
                )
                .repeat(1, out_channels, 1)
                .to(self.connection.w.device)
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.reshape(
                batch_size, out_channels * height_out * width_out, 1
            )
            self.p_minus = self.p_minus * torch.eye(
                out_channels * height_out * width_out
            ).to(self.connection.w.device)

        # Reshaping spike occurrences.
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )

        self.eligibility = self.eligibility.view(batch_size, *self.connection.w.shape)

        super().update()

    def _local_connection3d_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``LocalConnection3D`` subclass of
        ``AbstractConnection`` class.
        """

        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size[0]
        kernel_width = self.connection.kernel_size[1]
        kernel_depth = self.connection.kernel_size[2]
        in_channels = self.connection.in_channels
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]
        depth_out = self.connection.conv_size[2]

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * self.reduction(update, dim=0)

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = (
                self.p_plus.unfold(-3, kernel_height, stride[0])
                .unfold(-3, kernel_width, stride[1])
                .unfold(-3, kernel_depth, stride[2])
                .reshape(
                    batch_size,
                    height_out * width_out * depth_out,
                    in_channels * kernel_height * kernel_width * kernel_depth,
                )
                .repeat(1, out_channels, 1)
                .to(self.connection.w.device)
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.reshape(
                batch_size, out_channels * height_out * width_out * depth_out, 1
            )
            self.p_minus = self.p_minus * torch.eye(
                out_channels * height_out * width_out * depth_out
            ).to(self.connection.w.device)

        # Reshaping spike occurrences.
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_s = target_s * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )

        self.eligibility = self.eligibility.view(batch_size, *self.connection.w.shape)

        super().update()

    def _conv1d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Conv1dConnection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * torch.sum(update, dim=0)

        out_channels, in_channels, kernel_size = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = F.pad(self.p_plus, _pair(padding))
            self.p_plus = self.p_plus.unfold(-1, kernel_size, stride).reshape(
                batch_size, -1, in_channels * kernel_size
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = F.pad(self.source.s.float(), _pair(padding))
        source_s = source_s.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * torch.sum(update, dim=0)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            target_s, self.p_plus.permute((0, 2, 1))
        ) + torch.bmm(self.p_minus, source_s.permute((0, 2, 1)))
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()

    def _conv3d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Conv3dConnection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility
        self.connection.w += self.nu[0] * torch.sum(update, dim=0)

        (
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = F.pad(
                self.p_plus,
                (
                    padding[0],
                    padding[0],
                    padding[1],
                    padding[1],
                    padding[2],
                    padding[2],
                ),
            )
            self.p_plus = (
                self.p_plus.unfold(-3, kernel_width, stride[0])
                .unfold(-3, kernel_height, stride[1])
                .unfold(-3, kernel_depth, stride[2])
                .reshape(
                    batch_size,
                    -1,
                    in_channels * kernel_width * kernel_height * kernel_depth,
                )
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = F.pad(
            self.source.s,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_s = (
            source_s.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_s = self.target.s.view(batch_size, out_channels, -1).float()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()


class MSTDPET(LearningRule):
    # language=rst
    """
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDPET`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``MSTDPET`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.
        Keyword arguments:
        :param float tc_plus: Time constant for pre-synaptic firing trace.
        :param float tc_minus: Time constant for post-synaptic firing trace.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        elif isinstance(connection, LocalConnection1D):
            self.update = self._local_connection1d_update
        elif isinstance(connection, LocalConnection2D):
            self.update = self._local_connection2d_update
        elif isinstance(connection, LocalConnection3D):
            self.update = self._local_connection3d_update
        elif isinstance(connection, Conv1dConnection):
            self.update = self._conv1d_connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        elif isinstance(connection, Conv3dConnection):
            self.update = self._conv3d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility, eligibility trace, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros((self.source.n), device=self.source.s.device)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros((self.target.n), device=self.target.s.device)
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Compute weight update.
        self.connection.w += (
            self.nu[0] * self.connection.dt * reward * self.eligibility_trace
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.outer(self.p_plus, target_s) + torch.outer(
            source_s, self.p_minus
        )

        super().update()

    def _local_connection1d_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``LocalConnection1D`` subclass of
        ``AbstractConnection`` class.
        """

        # Get LC layer parameters.

        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_height = self.connection.kernel_size
        in_channels = self.connection.in_channels
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = (
                self.p_plus.unfold(-1, kernel_height, stride)
                .reshape(batch_size, height_out, in_channels * kernel_height)
                .repeat(1, out_channels, 1)
                .to(self.connection.w.device)
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.reshape(
                batch_size, out_channels * height_out, 1
            )
            self.p_minus = self.p_minus * torch.eye(out_channels * height_out).to(
                self.connection.w.device
            )

        # Reshaping spike occurrences.
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-1, kernel_height, stride)
            .reshape(batch_size, height_out, in_channels * kernel_height)
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # print(target_x.shape, source_s.shape)
        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out).to(
            self.connection.w.device
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )
        self.eligibility = self.eligibility.view(batch_size, *self.connection.w.shape)

        super().update()

    def _local_connection2d_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``LocalConnection2D`` subclass of
        ``AbstractConnection`` class.
        """
        # Get LC layer parameters.

        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_width = self.connection.kernel_size[0]
        kernel_height = self.connection.kernel_size[1]
        in_channels = self.connection.in_channels
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = (
                self.p_plus.unfold(-2, kernel_height, stride[0])
                .unfold(-2, kernel_width, stride[1])
                .reshape(
                    batch_size,
                    height_out * width_out,
                    in_channels * kernel_height * kernel_width,
                )
                .repeat(1, out_channels, 1)
                .to(self.connection.w.device)
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.reshape(
                batch_size, out_channels * height_out * width_out, 1
            )
            self.p_minus = self.p_minus * torch.eye(
                out_channels * height_out * width_out
            ).to(self.connection.w.device)

        # Reshaping spike occurrences.
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-2, kernel_height, stride[0])
            .unfold(-2, kernel_width, stride[1])
            .reshape(
                batch_size,
                height_out * width_out,
                in_channels * kernel_height * kernel_width,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # print(target_x.shape, source_s.shape)
        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out, 1
        )
        target_s = target_s * torch.eye(out_channels * height_out * width_out).to(
            self.connection.w.device
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )
        self.eligibility = self.eligibility.view(batch_size, *self.connection.w.shape)

        super().update()

    def _local_connection3d_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``LocalConnection3D`` subclass of
        ``AbstractConnection`` class.
        """

        # Get LC layer parameters.
        stride = self.connection.stride
        batch_size = self.source.batch_size
        kernel_width = self.connection.kernel_size[0]
        kernel_height = self.connection.kernel_size[1]
        kernel_depth = self.connection.kernel_size[2]
        in_channels = self.connection.in_channels
        out_channels = self.connection.n_filters
        height_out = self.connection.conv_size[0]
        width_out = self.connection.conv_size[1]
        depth_out = self.connection.conv_size[2]

        # Initialize eligibility.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = (
                self.p_plus.unfold(-3, kernel_height, stride[0])
                .unfold(-3, kernel_width, stride[1])
                .unfold(-3, kernel_depth, stride[2])
                .reshape(
                    batch_size,
                    height_out * width_out * depth_out,
                    in_channels * kernel_height * kernel_width * kernel_depth,
                )
                .repeat(1, out_channels, 1)
                .to(self.connection.w.device)
            )

        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.reshape(
                batch_size, out_channels * height_out * width_out * depth_out, 1
            )
            self.p_minus = self.p_minus * torch.eye(
                out_channels * height_out * width_out * depth_out
            ).to(self.connection.w.device)

        # Reshaping spike occurrences.
        source_s = (
            self.source.s.type(torch.float)
            .unfold(-3, kernel_height, stride[0])
            .unfold(-3, kernel_width, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                height_out * width_out * depth_out,
                in_channels * kernel_height * kernel_width * kernel_depth,
            )
            .repeat(1, out_channels, 1)
            .to(self.connection.w.device)
        )

        # print(target_x.shape, source_s.shape)
        target_s = self.target.s.type(torch.float).reshape(
            batch_size, out_channels * height_out * width_out * depth_out, 1
        )
        target_s = target_s * torch.eye(
            out_channels * height_out * width_out * depth_out
        ).to(self.connection.w.device)

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )
        self.eligibility = self.eligibility.view(batch_size, *self.connection.w.shape)

        super().update()

    def _conv1d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Conv1dConnection`` subclass of
        ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility and eligibility trace.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        out_channels, in_channels, kernel_size = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = F.pad(self.p_plus.float(), _pair(padding))
            self.p_plus = self.p_plus.unfold(-1, kernel_size, stride).reshape(
                batch_size, -1, in_channels * kernel_size
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = F.pad(self.source.s.float(), _pair(padding))
        source_s = source_s.unfold(-1, kernel_size, stride).reshape(
            batch_size, -1, in_channels * kernel_size
        )
        target_s = (
            self.target.s.permute(1, 2, 0).view(batch_size, out_channels, -1).float()
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Conv2dConnection`` subclass of
        ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility and eligibility trace.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(),
            kernel_height,
            kernel_width,
            padding=padding,
            stride=stride,
        )
        target_s = (
            self.target.s.permute(1, 2, 3, 0).view(batch_size, out_channels, -1).float()
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(
            target_s, self.p_plus.permute((0, 2, 1))
        ) + torch.bmm(self.p_minus, source_s.permute((0, 2, 1)))
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()

    def _conv3d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Conv3dConnection`` subclass of
        ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        batch_size = self.source.batch_size

        # Initialize eligibility and eligibility trace.
        if not hasattr(self, "eligibility"):
            self.eligibility = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                batch_size, *self.connection.w.shape, device=self.connection.w.device
            )

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.connection.w.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.connection.w.device
        )

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)

        # Compute weight update.
        update = reward * self.eligibility_trace
        self.connection.w += self.nu[0] * self.connection.dt * torch.sum(update, dim=0)

        (
            out_channels,
            in_channels,
            kernel_depth,
            kernel_height,
            kernel_width,
        ) = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros(
                batch_size, *self.source.shape, device=self.connection.w.device
            )
            self.p_plus = F.pad(
                self.p_plus,
                (
                    padding[0],
                    padding[0],
                    padding[1],
                    padding[1],
                    padding[2],
                    padding[2],
                ),
            )
            self.p_plus = (
                self.p_plus.unfold(-3, kernel_width, stride[0])
                .unfold(-3, kernel_height, stride[1])
                .unfold(-3, kernel_depth, stride[2])
                .reshape(
                    batch_size,
                    -1,
                    in_channels * kernel_width * kernel_height * kernel_depth,
                )
            )
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                batch_size, *self.target.shape, device=self.connection.w.device
            )
            self.p_minus = self.p_minus.view(batch_size, out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = F.pad(
            self.source.s,
            (padding[0], padding[0], padding[1], padding[1], padding[2], padding[2]),
        )
        source_s = (
            source_s.unfold(-3, kernel_width, stride[0])
            .unfold(-3, kernel_height, stride[1])
            .unfold(-3, kernel_depth, stride[2])
            .reshape(
                batch_size,
                -1,
                in_channels * kernel_width * kernel_height * kernel_depth,
            )
        )
        target_s = (
            self.target.s.permute(1, 2, 3, 4, 0)
            .view(batch_size, out_channels, -1)
            .float()
        )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.bmm(target_s, self.p_plus) + torch.bmm(
            self.p_minus, source_s
        )
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()


class Rmax(LearningRule):
    # language=rst
    """
    Reward-modulated learning rule derived from reward maximization principles. Adapted
    from `(Vasilaki et al., 2009)
    <https://intranet.physio.unibe.ch/Publikationen/Dokumente/Vasilaki2009PloSComputBio_1.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractConnection,
        nu: Optional[Union[float, Sequence[float], Sequence[torch.Tensor]]] = None,
        reduction: Optional[callable] = None,
        weight_decay: float = 0.0,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``R-max`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``R-max``
            learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively. It also
            accepts a pair of tensors to individualize learning rates of each neuron.
            In this case, their shape should be the same size as the connection weights.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param weight_decay: Coefficient controlling rate of decay of the weights each iteration.

        Keyword arguments:

        :param float tc_c: Time constant for balancing naive Hebbian and policy gradient
            learning.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection,
            nu=nu,
            reduction=reduction,
            weight_decay=weight_decay,
            **kwargs,
        )

        # Trace is needed for computing epsilon.
        assert (
            self.source.traces and self.source.traces_additive
        ), "Pre-synaptic nodes must use additive spike traces."

        # Derivation of R-max depends on stochastic SRM neurons!
        assert isinstance(
            self.target, SRM0Nodes
        ), "R-max needs stochastically firing neurons, use SRM0Nodes."

        if isinstance(connection, (Connection, LocalConnection)):
            self.update = self._connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_c = torch.tensor(
            kwargs.get("tc_c", 5.0)
        )  # 0 for pure naive Hebbian, inf for pure policy gradient.
        self.tc_e_trace = torch.tensor(kwargs.get("tc_e_trace", 25.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        R-max learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement
            learning task.
        """
        # Initialize eligibility trace.
        if not hasattr(self, "eligibility_trace"):
            self.eligibility_trace = torch.zeros(
                *self.connection.w.shape, device=self.connection.w.device
            )

        # Reshape variables.
        target_s = self.target.s.view(-1).float()
        target_s_prob = self.target.s_prob.view(-1)
        source_x = self.source.x.view(-1)

        # Parse keyword arguments.
        reward = kwargs["reward"]

        # New eligibility trace.
        self.eligibility_trace *= 1 - self.connection.dt / self.tc_e_trace
        self.eligibility_trace += (
            target_s
            - (target_s_prob / (1.0 + self.tc_c / self.connection.dt * target_s_prob))
        ) * source_x[:, None]

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * self.eligibility_trace

        super().update()
