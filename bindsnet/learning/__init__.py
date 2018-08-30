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

        :param connection: An ``AbstractConnection`` object which this learning rule will have no effect on.
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
        Post-pre learning rule for ``Connection`` subclass of ``AbstractionConnection`` class.
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
        Post-pre learning rule for ``Conv2dConnection`` subclass of ``AbstractionConnection`` class.
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

        :param connection: An ``AbstractConnection`` object which this learning rule will have no effect on.
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
        Hebbian learning rule for ``Connection`` subclass of ``AbstractionConnection`` class.
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
        Hebbian learning rule for ``Conv2dConnection`` subclass of ``AbstractionConnection`` class.
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


def m_stdp(conn: AbstractConnection, **kwargs) -> None:
    # language=rst
    """
    Reward-modulated STDP. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.

    :param conn: An ``AbstractConnection`` object whose weights are to be modified by the post-pre STDP rule.

    Keyword arguments:

    :param float reward: Reward signal from reinforcement learning task.
    :param float a_plus: Learning rate (post-synaptic).
    :param float a_minus: Learning rate (pre-synaptic).
    """
    # Parse keyword arguments.
    reward = kwargs['reward']
    a_plus = kwargs.get('a_plus', 1)
    a_minus = kwargs.get('a_plus', -1)

    if isinstance(conn, Connection):
        # Get P^+ and P^- values (function of firing traces).
        p_plus = a_plus * conn.source.x.unsqueeze(-1)
        p_minus = a_minus * conn.target.x.unsqueeze(0)

        # Get pre- and post-synaptic spiking neurons.
        pre_fire = conn.source.s.float().unsqueeze(-1)
        post_fire = conn.target.s.float().unsqueeze(0)

        # Calculate point eligibility value.
        eligibility = p_plus * post_fire + pre_fire * p_minus

        # Compute weight update.
        conn.w += conn.nu * reward * eligibility

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
    else:
        out_channels, _, kernel_height, kernel_width = conn.w.size()
        padding, stride = conn.padding, conn.stride

        p_plus = a_plus * im2col_indices(conn.source.x, kernel_height, kernel_width, padding=padding, stride=stride)

        p_minus = a_minus * conn.target.x.permute(1, 2, 3, 0).reshape(out_channels, -1)
        pre_fire = im2col_indices(conn.source.s, kernel_height, kernel_width, padding=padding, stride=stride).float()
        post_fire = conn.target.s.permute(1, 2, 3, 0).reshape(out_channels, -1).float()

        # Post-synaptic.
        post = (p_plus @ post_fire.t()).view(conn.w.size())
        if post.max() > 0:
            post = post / post.max()

        # Pre-synaptic.
        pre = (pre_fire @ p_minus.t()).view(conn.w.size())
        if pre.max() > 0:
            pre = pre / pre.max()

        # Calculate point eligibility value.
        eligibility = post + pre

        # Compute weight update.
        conn.w += conn.nu * reward * eligibility

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)


def m_stdp_et(conn: AbstractConnection, **kwargs) -> None:
    # language=rst
    """
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.

    :param conn: An ``AbstractConnection`` object whose weights are to be modified by the post-pre STDP rule.

    Keyword arguments:

    :param float reward: Reward signal from reinforcement learning task.
    :param float a_plus: Learning rate (post-synaptic).
    :param float a_minus: Learning rate (pre-synaptic).
    """
    # Parse keyword arguments.
    reward = kwargs['reward']
    a_plus = kwargs.get('a_plus', 1)
    a_minus = kwargs.get('a_plus', -1)

    if isinstance(conn, Connection):
        # Get P^+ and P^- values (function of firing traces).
        conn.p_plus = -(conn.tc_plus * conn.p_plus) + a_plus * conn.source.x.unsqueeze(-1)
        conn.p_minus = -(conn.tc_minus * conn.p_minus) + a_minus * conn.target.x.unsqueeze(0)

        # Get pre- and post-synaptic spiking neurons.
        pre_fire = conn.source.s.float().unsqueeze(-1)
        post_fire = conn.target.s.float().unsqueeze(0)

        # Calculate value of eligibility trace.
        conn.e_trace += -(conn.tc_e_trace * conn.e_trace) + conn.p_plus * post_fire + pre_fire * conn.p_minus

        # Compute weight update.
        conn.w += conn.nu * reward * conn.e_trace

        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
    else:
        out_channels, _, kernel_height, kernel_width = conn.w.size()
        padding, stride = conn.padding, conn.stride
        
        p_plus = a_plus * im2col_indices(conn.source.x, kernel_height, kernel_width, padding=padding, stride=stride)
        p_minus = a_minus * conn.target.x.permute(1, 2, 3, 0).reshape(out_channels, -1)
        pre_fire = im2col_indices(conn.source.s, kernel_height, kernel_width, padding=padding, stride=stride).float()
        post_fire = conn.target.s.permute(1, 2, 3, 0).reshape(out_channels, -1).float()
        
        # Post-synaptic.
        post = (p_plus @ post_fire.t()).view(conn.w.size())
        if post.max() > 0:
            post = post / post.max()
        
        # Pre-synaptic.
        pre = (pre_fire @ p_minus.t()).view(conn.w.size())
        if pre.max() > 0:
            pre = pre / pre.max()

        # Calculate point eligibility value.
        conn.e_trace += -(conn.tc_e_trace * conn.e_trace) + (post + pre)
        
        # Compute weight update.
        conn.w += conn.nu * reward * conn.e_trace
            
        # Bound weights.
        conn.w = torch.clamp(conn.w, conn.wmin, conn.wmax)
