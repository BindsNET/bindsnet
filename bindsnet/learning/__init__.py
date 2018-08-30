import torch

from abc import ABC
from typing import Union, Tuple, Optional

from ..utils import im2col_indices
from ..network.topology import AbstractConnection, Connection, Conv2dConnection


class LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Tuple[float, float]]] = None) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target

        self.wmin = connection.wmin
        self.wmax = connection.wmax

        # Learning rate(s).
        if nu is None:
            nu = 0.0

        if isinstance(nu, float):
            nu = (nu, nu)

        self.nu = nu

    def update(self) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        pass


class NoOp(LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Tuple[float, float]]] = None) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object which this learning rule will have no effect on.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        """
        super().__init__(
            connection=connection, nu=nu
        )

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """
        pass


class PostPre(LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. The pre-synpatic update is negative, while
    the post-synpatic update is positive.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Tuple[float, float]]] = None) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``PostPre`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        """
        super().__init__(
            connection=connection, nu=nu
        )

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, Connection):
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
        # Pre-synaptic update.
        self.connection.w -= self.nu[0] * torch.ger(
            self.source.s.float(), self.target.x
        )
        # Post-synaptic update.
        self.connection.w += self.nu[1] * torch.ger(
            self.source.x, self.target.s.float()
        )
        # Bound weights.
        self.connection.w = torch.clamp(
            self.connection.w, self.connection.wmin, self.connection.wmax
        )

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        # Get convolutional layer parameters.
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        x_source = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        x_target = self.target.x.permute(1, 2, 3, 0).reshape(out_channels, -1)
        s_source = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        s_target = self.target.s.permute(1, 2, 3, 0).reshape(out_channels, -1).float()

        # Pre-synaptic update.
        pre = x_target @ s_source.t()
        self.connection.w -= self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = s_target @ x_source.t()
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        # Bound weights.
        self.connection.w = torch.clamp(
            self.connection.w, self.connection.wmin, self.connection.wmax
        )


class Hebbian(LearningRule):
    # language=rst
    """
    Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Tuple[float, float]]] = None) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``Hebbian`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        """
        super().__init__(connection=connection, nu=nu)

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, Connection):
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
        # Pre-synaptic update.
        self.connection.w += self.nu[0] * torch.ger(
            self.source.s.float(), self.target.x
        )
        # Post-synaptic update.
        self.connection.w += self.nu[1] * torch.ger(
            self.source.x, self.target.s.float()
        )
        # Bound weights.
        self.connection.w = torch.clamp(
            self.connection.w, self.connection.wmin, self.connection.wmax
        )

    def _conv2d_connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Hebbian learning rule for ``Conv2dConnection`` subclass of ``AbstractConnection`` class.
        """
        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Reshaping spike traces and spike occurrences.
        x_source = im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        x_target = self.target.x.permute(1, 2, 3, 0).reshape(out_channels, -1)
        s_source = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        s_target = self.target.s.permute(1, 2, 3, 0).reshape(out_channels, -1).float()

        # Pre-synaptic update.
        pre = x_target @ s_source.t()
        self.connection.w += self.nu[0] * pre.view(self.connection.w.size())

        # Post-synaptic update.
        post = s_target @ x_source.t()
        self.connection.w += self.nu[1] * post.view(self.connection.w.size())

        # Bound weights.
        self.connection.w = torch.clamp(
            self.connection.w, self.connection.wmin, self.connection.wmax
        )


class MSTDP(LearningRule):
    # language=rst
    """
    Reward-modulated STDP. Adapted from `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Tuple[float, float]]] = None) -> None:
        # language=rst
        """
        Constructor for ``MSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        """
        super().__init__(connection=connection, nu=nu)

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, Connection):
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
        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_plus', -1)

        # Get P^+ and P^- values (function of firing traces).
        p_plus = a_plus * self.source.x
        p_minus = a_minus * self.target.x

        # Calculate point eligibility value.
        eligibility = torch.ger(p_plus, self.target.s.float()) + torch.ger(self.source.s.float(), p_minus)

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * eligibility

        # Bound weights.
        self.connection.w = torch.clamp(self.connection.w, self.wmin, self.wmax)

    def _conv2d_connection_update(self, **kwargs) -> None:
        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_plus', -1)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Get P^+ and P^- values (function of firing traces), and reshape source and target spikes.
        p_plus = a_plus * im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        p_minus = a_minus * self.target.x.permute(1, 2, 3, 0).reshape(out_channels, -1)
        pre_fire = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        post_fire = self.target.s.permute(1, 2, 3, 0).reshape(out_channels, -1).float()

        # Post-synaptic.
        post = (p_plus @ post_fire.t()).view(self.connection.w.size())
        if post.max() > 0:
            post = post / post.max()

        # Pre-synaptic.
        pre = (pre_fire @ p_minus.t()).view(self.connection.w.size())
        if pre.max() > 0:
            pre = pre / pre.max()

        # Calculate point eligibility value.
        eligibility = post + pre

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * eligibility

        # Bound weights.
        self.connection.w = torch.clamp(self.connection.w, self.wmin, self.wmax)


class MSTDPET(LearningRule):
    # language=rst
    """
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(self, connection: AbstractConnection, nu: Optional[Union[float, Tuple[float, float]]] = None) -> None:
        # language=rst
        """
        Constructor for ``MSTDPET`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDPET`` learning rule will modify.
        :param nu: Single or pair of learning rates for pre- and post-synaptic events, respectively.
        """
        super().__init__(connection=connection, nu=nu)

        assert self.source.traces and self.target.traces, 'Both pre- and post-synaptic nodes must record spike traces.'

        if isinstance(connection, Connection):
            self.update = self._connection_update
        elif isinstance(connection, Conv2dConnection):
            self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                'This learning rule is not supported for this Connection type.'
            )

        self.e_trace = 0
        self.tc_e_trace = 0.04
        self.p_plus = 0
        self.tc_plus = 0.05
        self.p_minus = 0
        self.tc_minus = 0.05

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        M-STDP-ET learning rule for ``Connection`` subclass of ``AbstractConnection`` class.

        Keyword arguments:

        :param float reward: Reward signal from reinforcement learning task.
        :param float a_plus: Learning rate (post-synaptic).
        :param float a_minus: Learning rate (pre-synaptic).
        """
        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_plus', -1)

        # Get P^+ and P^- values (function of firing traces).
        self.p_plus = -(self.tc_plus * self.p_plus) + a_plus * self.source.x
        self.p_minus = -(self.tc_minus * self.p_minus) + a_minus * self.target.x

        # Get pre- and post-synaptic spiking neurons.
        pre_fire = self.source.s.float()
        post_fire = self.target.s.float()

        # Calculate value of eligibility trace.
        self.e_trace -= self.tc_e_trace * self.e_trace
        self.e_trace += torch.ger(self.p_plus, post_fire) + torch.ger(pre_fire, self.p_minus)

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * self.e_trace

        # Bound weights.
        self.connection.w = torch.clamp(self.connection.w, self.wmin, self.wmax)

    def _conv2d_connection_update(self, **kwargs) -> None:
        # Parse keyword arguments.
        reward = kwargs['reward']
        a_plus = kwargs.get('a_plus', 1)
        a_minus = kwargs.get('a_plus', -1)

        out_channels, _, kernel_height, kernel_width = self.connection.w.size()
        padding, stride = self.connection.padding, self.connection.stride

        # Get P^+ and P^- values (function of firing traces).
        self.p_plus = -(self.tc_plus * self.p_plus) + a_plus * im2col_indices(
            self.source.x, kernel_height, kernel_width, padding=padding, stride=stride
        )
        self.p_minus = -(self.tc_minus * self.p_minus) + a_minus * \
                       self.target.x.permute(1, 2, 3, 0).reshape(out_channels, -1)

        # Get pre- and post-synaptic spiking neurons.
        pre_fire = im2col_indices(
            self.source.s.float(), kernel_height, kernel_width, padding=padding, stride=stride
        )
        post_fire = self.target.s.permute(1, 2, 3, 0).reshape(out_channels, -1).float()

        # Post-synaptic.
        post = (p_plus @ post_fire.t()).view(self.connection.w.size())
        if post.max() > 0:
            post = post / post.max()

        # Pre-synaptic.
        pre = (pre_fire @ p_minus.t()).view(self.connection.w.size())
        if pre.max() > 0:
            pre = pre / pre.max()

        # Calculate point eligibility value.
        self.e_trace += -(self.tc_e_trace * self.e_trace) + (post + pre)

        # Compute weight update.
        self.connection.w += self.nu[0] * reward * self.e_trace

        # Bound weights.
        self.connection.w = torch.clamp(self.connection.w, self.wmin, self.wmax)
