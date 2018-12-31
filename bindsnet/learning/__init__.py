import torch
import numpy as np

from abc import ABC
from typing import Union, Tuple, Optional, Sequence

from ..utils import im2col_indices
from ..network.topology import AbstractConnection, Connection, Conv2dConnection, LocallyConnectedConnection


class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = [0.0, 0.0]
        elif isinstance(nu, float) or isinstance(nu, int):
            nu = [nu, nu]

        self.nu = nu

        # Weight decay.
        self.weight_decay = weight_decay

    def update(self) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        # Implement weight decay.
        if self.weight_decay:
            self.connection.w -= self.weight_decay * self.connection.w

        # Bound weights.
        if None not in [self.connection.wmin, self.connection.wmax] and not isinstance(self, NoOp):
            self.connection.w = torch.clamp(
                self.connection.w, self.connection.wmin, self.connection.wmax
            )


class NoOp(LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object which this learning rule will have no effect on.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
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
    Simple STDP rule involving both pre- and post-synaptic spiking activity. The pre-synaptic update is negative, while
    the post-synpatic update is positive.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        """
        super().update()

        source_s = self.source.s.view(-1).float()
        source_x = self.source.x.view(-1)
        target_s = self.target.s.view(-1).float()
        target_x = self.target.x.view(-1)

        # Pre-synaptic update.
        if self.nu[0]:
            self.connection.w -= self.nu[0] * torch.ger(source_s, target_x)

        # Post-synaptic update.
        if self.nu[1]:
            self.connection.w += self.nu[1] * torch.ger(source_x, target_s)

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        super().update()

        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        x_source = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        x_target = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        s_source = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        s_target = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0]:
            pre = x_target @ s_source.t()
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1]:
            post = s_target @ x_source.t()
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())


class WeightDependentPostPre(LearningRule):
    # language=rst
    """
    STDP rule involving both pre- and post-synaptic spiking activity. The post-synaptic update is positive and the pre-
    synaptic update is negative, and both are dependent on the magnitude of the synaptic weights.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``WeightDependentPostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``WeightDependentPostPre`` learning rule will
                           modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

        assert self.source.traces, 'Pre-synaptic nodes must record spike traces.'
        assert connection.wmin is not None and connection.wmax is not None, 'Connection must define wmin and wmax.'

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        """
        super().update()

        source_s = self.source.s.view(-1).float()
        source_x = self.source.x.view(-1)
        target_s = self.target.s.view(-1).float()
        target_x = self.target.x.view(-1)

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            update -= self.nu[0] * torch.ger(source_s, target_x) * (self.connection.w - self.wmin)

        # Post-synaptic update.
        if self.nu[1]:
            update += self.nu[1] * torch.ger(source_x, target_s) * (self.wmax - self.connection.w)

        self.connection.w += update

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        super().update()

        # Get convolutional layer parameters.
        out_channels, in_channels, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        x_source = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        x_target = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        s_source = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        s_target = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            pre = x_target @ s_source.t()
            update -= self.nu[0] * pre.view(self.connection.w.size())(self.connection.w - self.wmin)

        # Post-synaptic update.
        if self.nu[1]:
            post = s_target @ x_source.t()
            update += self.nu[1] * post.view(self.connection.w.size()) * (self.wmax - self.connection.wmin)

        self.connection.w += update


class Hebbian(LearningRule):
    # language=rst
    """
    Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``Hebbian`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``Hebbian`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Connection`` subclass of ``AbstractConnection`` class.
        """
        super().update()

        source_s = self.source.s.view(-1).float()
        source_x = self.source.x.view(-1)
        target_s = self.target.s.view(-1).float()
        target_x = self.target.x.view(-1)

        # Pre-synaptic update.
        self.connection.w += self.nu[0] * torch.ger(
            source_s, target_x
        )
        # Post-synaptic update.
        self.connection.w += self.nu[1] * torch.ger(
            source_x, target_s
        )

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        super().update()

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        x_source = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        x_target = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        s_source = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        s_target = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Pre-synaptic update.
        pre = x_target @ s_source.t()
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = s_target @ x_source.t()
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())


class MSTDP(LearningRule):
    # language=rst
    """
    Reward-modulated STDP. Adapted from `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``MSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        M-STDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param float reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        super().update()

        source_s = self.source.s.view(-1).float()
        source_x = self.source.x.view(-1)
        target_s = self.target.s.view(-1).float()
        target_x = self.target.x.view(-1)

        self.connection.w = self.connection.w.view(self.source.n, self.target.n)

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_minus', -1)

        # Get P^+ and P^- values (function of firing traces).
        p_plus = a_plus * source_x
        p_minus = a_minus * target_x

        # Calculate point eligibility value.
        eligibility = torch.ger(p_plus, target_s) + torch.ger(source_s, p_minus)

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * eligibility

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        M-STDP learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param float reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        super().update()

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_minus', -1)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        x_source = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        x_target = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        s_source = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        s_target = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Get P^+ and P^- values (function of firing traces), and reshape source and target spikes.
        p_plus = a_plus * x_source
        p_minus = a_minus * x_target

        # Pre- and post-synaptic updates.
        pre = (s_source @ p_minus.t()).view(self.connection.w.size())
        post = (p_plus @ s_target.t()).view(self.connection.w.size())

        # Calculate point eligibility value.
        eligibility = post + pre

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * eligibility


class MSTDPET(LearningRule):
    # language=rst
    """
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Sequence[float]]] = None,
                 weight_decay: float = 0.0, **kwargs) -> None:
        # language=rst
        """
        Constructor for ``MSTDPET`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDPET`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        :param weight_decay: Constant multiple to decay weights by on each iteration.

        Keyword arguments:

        :param float tc_e_trace: Time constant of the eligibility trace.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

        self.e_trace = torch.zeros(self.source.n, self.target.n)
        self.tc_e_trace = kwargs.get('tc_e_trace', 5e-2)

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        M-STDP-ET learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param float reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        super().update()

        source_s = self.source.s.view(-1).float()
        source_x = self.source.x.view(-1)
        target_s = self.target.s.view(-1).float()
        target_x = self.target.x.view(-1)

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_minus', -1)

        # Get P^+ and P^- values (function of firing traces).
        self.p_plus = a_plus * source_x
        self.p_minus = a_minus * target_x

        # Calculate value of eligibility trace.
        self.e_trace -= self.tc_e_trace * self.e_trace
        update = torch.ger(self.p_plus, target_s) + torch.ger(source_s, self.p_minus)
        self.e_trace[update != 0] = update[update != 0]

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * self.e_trace

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        M-STDP-ET learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param float reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        super().update()

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_minus', -1)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        x_source = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        x_target = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        s_source = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        s_target = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Get P^+ and P^- values (function of firing traces).
        self.p_plus = a_plus * x_source
        self.p_minus = a_minus * x_target

        # Post-synaptic and pre-synaptic updates.
        post = (self.p_plus @ s_target.t()).view(self.connection.w.size())
        pre = (s_source @ self.p_minus.t()).view(self.connection.w.size())
        update = post + pre

        # Calculate value of eligibility trace.
        self.e_trace[update != 0] = update[update != 0]

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * self.e_trace
