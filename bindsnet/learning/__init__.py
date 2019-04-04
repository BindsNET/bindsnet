from abc import ABC
from typing import Union, Optional, Sequence

import numpy as np
import torch

from ..network.topology import AbstractConnection, Connection, Conv2dConnection, LocallyConnectedConnection
from ..utils import im2col_indices


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
        if (self.connection.wmin != -np.inf or self.connection.wmax != np.inf) and not isinstance(self, NoOp):
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

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_s = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Pre-synaptic update.
        if self.nu[0]:
            pre = target_x @ source_s.t()
            self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        if self.nu[1]:
            post = target_s @ source_x.t()
            self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


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
        assert connection.wmin != -np.inf and connection.wmax != np.inf, 'Connection must define finite wmin and wmax.'

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

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, in_channels, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_x = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_s = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        update = 0

        # Pre-synaptic update.
        if self.nu[0]:
            pre = target_x @ source_s.t()
            update -= self.nu[0] * pre.view(self.connection.w.size()) * (self.connection.w - self.wmin)

        # Post-synaptic update.
        if self.nu[1]:
            post = target_s @ source_x.t()
            update += self.nu[1] * post.view(self.connection.w.size()) * (self.wmax - self.connection.wmin)

        self.connection.w += update

        super().update()


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

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        source_x = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )

        target_x = self.target.x.permute(1, 2, 3, 0).view(out_channels, -1)
        source_s = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_s = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Pre-synaptic update.
        pre = target_x @ source_s.t()
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = target_s @ source_x.t()
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        super().update()


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

        Keyword arguments:

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

        self.tc_plus = torch.tensor(kwargs.get('tc_plus', 20.0))
        self.tc_minus = torch.tensor(kwargs.get('tc_minus', 20.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility, P^+, and P^-.
        if not hasattr(self, 'p_plus'):
            self.p_plus = torch.zeros(self.source.n)
        if not hasattr(self, 'p_minus'):
            self.p_minus = torch.zeros(self.target.n)
        if not hasattr(self, 'eligibility'):
            self.eligibility = torch.zeros(*self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = torch.tensor(kwargs.get('a_plus', 1.0))
        a_minus = torch.tensor(kwargs.get('a_minus', -1.0))

        # Compute weight update based on the point eligibility value of the past timestep.
        self.connection.w += self.nu[0] * reward * self.eligibility

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.ger(self.p_plus, target_s) + \
                           torch.ger(source_s, self.p_minus)

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDP learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility.
        if not hasattr(self, 'eligibility'):
            self.eligibility = torch.zeros(*self.connection.w.shape)

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = torch.tensor(kwargs.get('a_plus', 1.0))
        a_minus = torch.tensor(kwargs.get('a_minus', -1.0))

        # Compute weight update based on the point eligibility value of the past timestep.
        self.connection.w += self.nu[0] * reward * self.eligibility

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, 'p_plus'):
            self.p_plus = torch.zeros(*self.source.s.size())
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, 'p_minus'):
            self.p_minus = torch.zeros(*self.target.s.size())
            self.p_minus = self.p_minus.view(out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_s = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = target_s @ self.p_plus.t() + self.p_minus @ source_s.t()
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()


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

        :param float tc_plus: Time constant for pre-synaptic firing trace.
        :param float tc_minus: Time constant for post-synaptic firing trace.
        :param float tc_e_trace: Time constant for the eligibility trace.
        """
        super().__init__(
            connection=connection, nu=nu, weight_decay=weight_decay, **kwargs
        )

        if isinstance(connection, (Connection, LocallyConnectedConnection)):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

        self.tc_plus = torch.tensor(kwargs.get('tc_plus', 20.0))
        self.tc_minus = torch.tensor(kwargs.get('tc_minus', 20.0))
        self.tc_e_trace = torch.tensor(kwargs.get('tc_e_trace', 25.0))

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility, eligibility trace, P^+, and P^-.
        if not hasattr(self, 'p_plus'):
            self.p_plus = torch.zeros(self.source.n)
        if not hasattr(self, 'p_minus'):
            self.p_minus = torch.zeros(self.target.n)
        if not hasattr(self, 'eligibility'):
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, 'eligibility_trace'):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = torch.tensor(kwargs.get('a_plus', 1.0))
        a_minus = torch.tensor(kwargs.get('a_minus', -1.0))

        # Calculate value of eligibility trace based on the value of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Compute weight update.
        self.connection.w += self.nu[0] * self.connection.dt * reward * self.eligibility_trace

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = torch.ger(self.p_plus, target_s) + \
                           torch.ger(source_s, self.p_minus)

        super().update()

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        MSTDPET learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Initialize eligibility and eligibility trace.
        if not hasattr(self, 'eligibility'):
            self.eligibility = torch.zeros(*self.connection.w.shape)
        if not hasattr(self, 'eligibility_trace'):
            self.eligibility_trace = torch.zeros(*self.connection.w.shape)

        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = torch.tensor(kwargs.get('a_plus', 1.0))
        a_minus = torch.tensor(kwargs.get('a_minus', -1.0))

        # Calculate value of eligibility trace based on the value of the point eligibility value of the past timestep.
        self.eligibility_trace *= torch.exp(-self.connection.dt / self.tc_e_trace)
        self.eligibility_trace += self.eligibility / self.tc_e_trace

        # Compute weight update.
        self.connection.w += self.nu[0] * self.connection.dt * reward * self.eligibility_trace

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Initialize P^+ and P^-.
        if not hasattr(self, 'p_plus'):
            self.p_plus = torch.zeros(*self.source.s.size())
            self.p_plus = im2col_indices(
                self.p_plus, kernel_height, kernel_width, padding=padding, stride=stride
            )
        if not hasattr(self, 'p_minus'):
            self.p_minus = torch.zeros(*self.target.s.size())
            self.p_minus = self.p_minus.view(out_channels, -1).float()

        # Reshaping spike occurrences.
        source_s = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        target_s = self.target.s.permute(1, 2, 3, 0).view(out_channels, -1).float()

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)
        self.p_plus += a_plus * source_s
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)
        self.p_minus += a_minus * target_s

        # Calculate point eligibility value.
        self.eligibility = target_s @ self.p_plus.t() + self.p_minus @ source_s.t()
        self.eligibility = self.eligibility.view(self.connection.w.size())

        super().update()
