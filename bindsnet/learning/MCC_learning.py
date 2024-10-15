from abc import ABC, abstractmethod
from typing import Union, Optional, Sequence
import warnings

import torch
import numpy as np

from ..network.nodes import SRM0Nodes
from ..network.topology import (
    AbstractMulticompartmentConnection,
    MulticompartmentConnection,
)
from ..utils import im2col_indices


class MCC_LearningRule(ABC):
    # language=rst
    """
    Abstract base class for learning rules.
    """

    def __init__(
        self,
        connection: AbstractMulticompartmentConnection,
        feature_value: Union[float, int, torch.Tensor],
        range: Optional[Union[list, tuple]] = None,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        decay: float = 0.0,
        enforce_polarity: bool = False,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Abstract constructor for the ``LearningRule`` object.

        :param connection: An ``AbstractConnection`` object.
        :param feature_value: Value(s) to be updated. Can be only tensor (scalar currently not supported)
        :param range: Allowed range for :code:`feature_value`
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param decay: Coefficient controlling rate of decay of the weights each iteration.
        :param enforce_polarity: Will prevent synapses from changing signs if :code:`True`
        """
        # Connection parameters.
        self.connection = connection
        self.source = connection.source
        self.target = connection.target
        self.feature_value = feature_value
        self.enforce_polarity = enforce_polarity
        self.min, self.max = range

        # Learning rate(s).
        if nu is None:
            nu = [0.2, 0.1]
        elif isinstance(nu, (float, int)):
            nu = [nu, nu]

        # Keep track of polarities
        if enforce_polarity:
            self.polarities = torch.sign(self.feature_value)

        self.nu = torch.zeros(2, dtype=torch.float)
        self.nu[0] = nu[0]
        self.nu[1] = nu[1]

        if (self.nu == torch.zeros(2)).all() and not isinstance(self, NoOp):
            warnings.warn(
                f"nu is set to [0., 0.] for {type(self).__name__} learning rule. "
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
        self.decay = 1.0 - decay if decay else 1.0

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Abstract method for a learning rule update.
        """

        # Implement decay.
        if self.decay:
            self.feature_value *= self.decay

        # Enforce polarities
        if self.enforce_polarity:
            polarity_swaps = self.polarities == torch.sign(self.feature_value)
            self.feature_value[polarity_swaps == 0] = 0

        # Bound weights.
        if ((self.min is not None) or (self.max is not None)) and not isinstance(
            self, NoOp
        ):
            self.feature_value.clamp_(self.min, self.max)

    @abstractmethod
    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the feature.
        """
        pass


class NoOp(MCC_LearningRule):
    # language=rst
    """
    Learning rule with no effect.
    """

    def __init__(self, **args) -> None:
        # language=rst
        """
        No operation done during runtime
        """
        pass

    def update(self, **kwargs) -> None:
        # language=rst
        """
        No operation done during runtime
        """
        pass

    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the feature.
        """
        pass


class PostPre(MCC_LearningRule):
    # language=rst
    """
    Simple STDP rule involving both pre- and post-synaptic spiking activity. By default,
    pre-synaptic update is negative and the post-synaptic update is positive.
    """

    def __init__(
        self,
        connection: AbstractMulticompartmentConnection,
        feature_value: Union[torch.Tensor, float, int],
        range: Optional[Sequence[float]] = None,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        decay: float = 0.0,
        enforce_polarity: bool = False,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``PostPre`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``PostPre`` learning rule will modify.
        :param feature_value: The object which will be altered
        :param range: The domain for the feature
        :param nu: Single or pair of learning rates for pre- and post-synaptic events.
        :param reduction: Method for reducing parameter updates along the batch
            dimension.
        :param decay: Coefficient controlling rate of decay of the weights each iteration.
        :param enforce_polarity: Will prevent synapses from changing signs if :code:`True`

        Keyword arguments:
        :param average_update: Number of updates to average over, 0=No averaging, x=average over last x updates
        :param continues_update: If True, the update will be applied after every update, if False, only after the average_update buffer is full
        """
        super().__init__(
            connection=connection,
            feature_value=feature_value,
            range=[-1, +1] if range is None else range,
            nu=nu,
            reduction=reduction,
            decay=decay,
            enforce_polarity=enforce_polarity,
            **kwargs,
        )

        assert self.source.traces and self.target.traces, (
            "Both pre- and post-synaptic nodes must record spike traces "
            "(use traces='True' on source/target layers)"
        )

        if isinstance(connection, (MulticompartmentConnection)):
            self.update = self._connection_update
        # elif isinstance(connection, Conv2dConnection):
        #     self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        # Initialize variables for average update and continues update
        self.average_update = kwargs.get("average_update", 0)
        self.continues_update = kwargs.get("continues_update", False)

        if self.average_update > 0:
            self.average_buffer_pre = torch.zeros(
                self.average_update,
                *self.feature_value.shape,
                device=self.feature_value.device,
            )
            self.average_buffer_post = torch.zeros_like(self.average_buffer_pre)
            self.average_buffer_index_pre = 0
            self.average_buffer_index_post = 0

    def _connection_update(self, **kwargs) -> None:
        # language=rst
        """
        Post-pre learning rule for ``Connection`` subclass of ``AbstractConnection``
        class.
        """
        batch_size = self.source.batch_size

        # Pre-synaptic update.
        if self.nu[0]:
            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1) * self.nu[0]

            if self.average_update > 0:
                self.average_buffer_pre[self.average_buffer_index_pre] = self.reduction(
                    torch.bmm(source_s, target_x), dim=0
                )

                self.average_buffer_index_pre = (
                    self.average_buffer_index_pre + 1
                ) % self.average_update

                if self.continues_update:
                    self.feature_value -= (
                        torch.mean(self.average_buffer_pre, dim=0) * self.connection.dt
                    )
                elif self.average_buffer_index_pre == 0:
                    self.feature_value -= (
                        torch.mean(self.average_buffer_pre, dim=0) * self.connection.dt
                    )
            else:
                self.feature_value -= (
                    self.reduction(torch.bmm(source_s, target_x), dim=0)
                    * self.connection.dt
                )
            del source_s, target_x

        # Post-synaptic update.
        if self.nu[1]:
            target_s = (
                self.target.s.view(batch_size, -1).unsqueeze(1).float() * self.nu[1]
            )
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)

            if self.average_update > 0:
                self.average_buffer_post[self.average_buffer_index_post] = (
                    self.reduction(torch.bmm(source_x, target_s), dim=0)
                )

                self.average_buffer_index_post = (
                    self.average_buffer_index_post + 1
                ) % self.average_update

                if self.continues_update:
                    self.feature_value += (
                        torch.mean(self.average_buffer_post, dim=0) * self.connection.dt
                    )
                elif self.average_buffer_index_post == 0:
                    self.feature_value += (
                        torch.mean(self.average_buffer_post, dim=0) * self.connection.dt
                    )
            else:
                self.feature_value += (
                    self.reduction(torch.bmm(source_x, target_s), dim=0)
                    * self.connection.dt
                )
            del source_x, target_s

        super().update()

    def reset_state_variables(self):
        return

    class Hebbian(MCC_LearningRule):
        # language=rst
        """
        Simple Hebbian learning rule. Pre- and post-synaptic updates are both positive.
        """

        def __init__(
            self,
            connection: AbstractMulticompartmentConnection,
            feature_value: Union[torch.Tensor, float, int],
            nu: Optional[Union[float, Sequence[float]]] = None,
            reduction: Optional[callable] = None,
            decay: float = 0.0,
            **kwargs,
        ) -> None:
            # language=rst
            """
            Constructor for ``Hebbian`` learning rule.

            :param connection: An ``AbstractConnection`` object whose weights the
                ``Hebbian`` learning rule will modify.
            :param nu: Single or pair of learning rates for pre- and post-synaptic events.
            :param reduction: Method for reducing parameter updates along the batch
                dimension.
            :param decay: Coefficient controlling rate of decay of the weights each iteration.
            """
            super().__init__(
                connection=connection,
                feature_value=feature_value,
                nu=nu,
                reduction=reduction,
                decay=decay,
                **kwargs,
            )

            assert (
                self.source.traces and self.target.traces
            ), "Both pre- and post-synaptic nodes must record spike traces."

            if isinstance(MulticompartmentConnection):
                self.update = self._connection_update
                self.feature_value = feature_value
            # elif isinstance(connection, Conv2dConnection):
            #     self.update = self._conv2d_connection_update
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

            # Add polarities back to feature after updates
            if self.enforce_polarity:
                self.feature_value = torch.abs(self.feature_value)

            batch_size = self.source.batch_size

            source_s = self.source.s.view(batch_size, -1).unsqueeze(2).float()
            source_x = self.source.x.view(batch_size, -1).unsqueeze(2)
            target_s = self.target.s.view(batch_size, -1).unsqueeze(1).float()
            target_x = self.target.x.view(batch_size, -1).unsqueeze(1)

            # Pre-synaptic update.
            update = self.reduction(torch.bmm(source_s, target_x), dim=0)
            self.feature_value += self.nu[0] * update

            # Post-synaptic update.
            update = self.reduction(torch.bmm(source_x, target_s), dim=0)
            self.feature_value += self.nu[1] * update

            # Add polarities back to feature after updates
            if self.enforce_polarity:
                self.feature_value = self.feature_value * self.polarities

            super().update()

        def reset_state_variables(self):
            return


class MSTDP(MCC_LearningRule):
    # language=rst
    """
    Reward-modulated STDP. Adapted from `(Florian 2007)
    <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractMulticompartmentConnection,
        feature_value: Union[torch.Tensor, float, int],
        range: Optional[Sequence[float]] = None,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        decay: float = 0.0,
        enforce_polarity: bool = False,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDP`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the ``MSTDP``
            learning rule will modify.
        :param feature_value: The object which will be altered
        :param range: The domain for the feature
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param decay: Coefficient controlling rate of decay of the weights each iteration.
        :param enforce_polarity: Will prevent synapses from changing signs if :code:`True`

        Keyword arguments:

        :param average_update: Number of updates to average over, 0=No averaging, x=average over last x updates
        :param continues_update: If True, the update will be applied after every update, if False, only after the average_update buffer is full

        :param tc_plus: Time constant for pre-synaptic firing trace.
        :param tc_minus: Time constant for post-synaptic firing trace.
        """
        super().__init__(
            connection=connection,
            feature_value=feature_value,
            range=[-1, +1] if range is None else range,
            nu=nu,
            reduction=reduction,
            decay=decay,
            enforce_polarity=enforce_polarity,
            **kwargs,
        )

        if isinstance(connection, (MulticompartmentConnection)):
            self.update = self._connection_update
        # elif isinstance(connection, Conv2dConnection):
        #     self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(kwargs.get("tc_plus", 20.0))
        self.tc_minus = torch.tensor(kwargs.get("tc_minus", 20.0))

        # Initialize variables for average update and continues update
        self.average_update = kwargs.get("average_update", 0)
        self.continues_update = kwargs.get("continues_update", False)

        if self.average_update > 0:
            self.average_buffer = torch.zeros(
                self.average_update,
                *self.feature_value.shape,
                device=self.feature_value.device,
            )
            self.average_buffer_index = 0

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
                batch_size, *self.feature_value.shape, device=self.feature_value.device
            )

        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(batch_size, -1).float()
        target_s = self.target.s.view(batch_size, -1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = torch.tensor(
            kwargs.get("a_plus", 1.0), device=self.feature_value.device
        )
        a_minus = torch.tensor(
            kwargs.get("a_minus", -1.0), device=self.feature_value.device
        )

        # Compute weight update based on the eligibility value of the past timestep.
        update = reward * self.eligibility

        if self.average_update > 0:
            self.average_buffer[self.average_buffer_index] = self.reduction(
                update, dim=0
            )
            self.average_buffer_index = (
                self.average_buffer_index + 1
            ) % self.average_update

            if self.continues_update:
                self.feature_value += self.nu[0] * torch.mean(
                    self.average_buffer, dim=0
                )
            elif self.average_buffer_index == 0:
                self.feature_value += self.nu[0] * torch.mean(
                    self.average_buffer, dim=0
                )
        else:
            self.feature_value += self.nu[0] * self.reduction(update, dim=0)

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

    def reset_state_variables(self):
        return


class MSTDPET(MCC_LearningRule):
    # language=rst
    """
    Reward-modulated STDP with eligibility trace. Adapted from
    `(Florian 2007) <https://florian.io/papers/2007_Florian_Modulated_STDP.pdf>`_.
    """

    def __init__(
        self,
        connection: AbstractMulticompartmentConnection,
        feature_value: Union[torch.Tensor, float, int],
        range: Optional[Sequence[float]] = None,
        nu: Optional[Union[float, Sequence[float]]] = None,
        reduction: Optional[callable] = None,
        decay: float = 0.0,
        enforce_polarity: bool = False,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Constructor for ``MSTDPET`` learning rule.

        :param connection: An ``AbstractConnection`` object whose weights the
            ``MSTDPET`` learning rule will modify.
        :param feature_value: The object which will be altered
        :param range: The domain for the feature
        :param nu: Single or pair of learning rates for pre- and post-synaptic events,
            respectively.
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension.
        :param decay: Coefficient controlling rate of decay of the weights each iteration.
        :param enforce_polarity: Will prevent synapses from changing signs if :code:`True`

        Keyword arguments:

        :param float tc_plus: Time constant for pre-synaptic firing trace.
        :param float tc_minus: Time constant for post-synaptic firing trace.
        :param float tc_e_trace: Time constant for the eligibility trace.
        :param average_update: Number of updates to average over, 0=No averaging, x=average over last x updates
        :param continues_update: If True, the update will be applied after every update, if False, only after the average_update buffer is full

        """
        super().__init__(
            connection=connection,
            feature_value=feature_value,
            range=[-1, +1] if range is None else range,
            nu=nu,
            reduction=reduction,
            decay=decay,
            enforce_polarity=enforce_polarity,
            **kwargs,
        )

        if isinstance(connection, (MulticompartmentConnection)):
            self.update = self._connection_update
        # elif isinstance(connection, Conv2dConnection):
        #     self.update = self._conv2d_connection_update
        else:
            raise NotImplementedError(
                "This learning rule is not supported for this Connection type."
            )

        self.tc_plus = torch.tensor(
            kwargs.get("tc_plus", 20.0)
        )  # How long pos reinforcement effects weights
        self.tc_minus = torch.tensor(
            kwargs.get("tc_minus", 20.0)
        )  # How long neg reinforcement effects weights
        self.tc_e_trace = torch.tensor(
            kwargs.get("tc_e_trace", 25.0)
        )  # How long trace effects weights
        self.eligibility = torch.zeros(
            *self.feature_value.shape, device=self.feature_value.device
        )
        self.eligibility_trace = torch.zeros(
            *self.feature_value.shape, device=self.feature_value.device
        )

        # Initialize eligibility, eligibility trace, P^+, and P^-.
        if not hasattr(self, "p_plus"):
            self.p_plus = torch.zeros((self.source.n), device=self.feature_value.device)
        if not hasattr(self, "p_minus"):
            self.p_minus = torch.zeros(
                (self.target.n), device=self.feature_value.device
            )

        # Initialize variables for average update and continues update
        self.average_update = kwargs.get("average_update", 0)
        self.continues_update = kwargs.get("continues_update", False)
        if self.average_update > 0:
            self.average_buffer = torch.zeros(
                self.average_update,
                *self.feature_value.shape,
                device=self.feature_value.device,
            )
            self.average_buffer_index = 0

    # @profile
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
        # Reshape pre- and post-synaptic spikes.
        source_s = self.source.s.view(-1).float()
        target_s = self.target.s.view(-1).float()

        # Parse keyword arguments.
        reward = kwargs["reward"]
        a_plus = kwargs.get("a_plus", 1.0)
        # if isinstance(a_plus, dict):
        #     for k, v in a_plus.items():
        #         a_plus[k] = torch.tensor(v, device=self.feature_value.device)
        # else:
        a_plus = torch.tensor(a_plus, device=self.feature_value.device)
        a_minus = kwargs.get("a_minus", -1.0)
        # if isinstance(a_minus, dict):
        #     for k, v in a_minus.items():
        #         a_minus[k] = torch.tensor(v, device=self.feature_value.device)
        # else:
        a_minus = torch.tensor(a_minus, device=self.feature_value.device)

        # Calculate value of eligibility trace based on the value
        # of the point eligibility value of the past timestep.
        # Note: eligibility = [source.n, target.n] > 0 where source and target spiked
        # Note: high negs. ->
        self.eligibility_trace *= torch.exp(
            -self.connection.dt / self.tc_e_trace
        )  # Decay
        self.eligibility_trace += self.eligibility / self.tc_e_trace  # Additive changes
        # ^ Also effected by delay in last step

        # Compute weight update.

        if self.average_update > 0:
            self.average_buffer[self.average_buffer_index] = (
                self.nu[0] * self.connection.dt * reward * self.eligibility_trace
            )
            self.average_buffer_index = (
                self.average_buffer_index + 1
            ) % self.average_update

            if self.continues_update:
                self.feature_value += torch.mean(self.average_buffer, dim=0)
            elif self.average_buffer_index == 0:
                self.feature_value += torch.mean(self.average_buffer, dim=0)
        else:
            self.feature_value += (
                self.nu[0] * self.connection.dt * reward * self.eligibility_trace
            )

        # Update P^+ and P^- values.
        self.p_plus *= torch.exp(-self.connection.dt / self.tc_plus)  # Decay
        self.p_plus += a_plus * source_s  # Scaled source spikes
        self.p_minus *= torch.exp(-self.connection.dt / self.tc_minus)  # Decay
        self.p_minus += a_minus * target_s  # Scaled target spikes

        # Notes:
        #
        # a_plus -> How much a spike in src contributes to the eligibility
        # a_minus -> How much a spike in trg contributes to the eligibility (neg)
        # p_plus -> +a_plus every spike, with decay
        # p_minus -> +a_minus every spike, with decay

        # Calculate point eligibility value.
        self.eligibility = torch.outer(self.p_plus, target_s) + torch.outer(
            source_s, self.p_minus
        )

        super().update()

    def reset_state_variables(self) -> None:
        self.eligibility.zero_()
        self.eligibility_trace.zero_()
        return
