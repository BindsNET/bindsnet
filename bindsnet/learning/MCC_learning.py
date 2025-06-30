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

#MCC much faster and durable
# Old is practical connection between 2 objects and you can only use it as a weight
# If theres a mask you want to add, you need to add another object, or another weight
# run the experiments 
#Object pipe it can be delayed weighted

class MCC_LearningRule(ABC): #multicompartment connection 
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

# Remove the MyBackpropVariant class and replace with:

class ForwardForwardMCCLearning(MCC_LearningRule):
    """
    Forward-Forward learning rule for MulticompartmentConnection.
    
    This MCC learning rule wrapper integrates the Forward-Forward algorithm
    with the MulticompartmentConnection architecture, enabling layer-wise
    learning without backpropagation through time.
    
    The learning rule works by:
    1. Computing goodness scores from target layer activity
    2. Collecting positive and negative sample statistics
    3. Applying contrastive weight updates based on Forward-Forward loss
    """
    
    def __init__(
        self,
        alpha_loss: float = 0.6,
        goodness_fn: str = "mean_squared",
        nu: float = 0.001,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        **kwargs
    ):
        """
        Initialize Forward-Forward MCC learning rule.
        
        Args:
            alpha_loss: Forward-Forward loss threshold parameter
            goodness_fn: Goodness score computation method ("mean_squared", "sum_squared")
            nu: Learning rate for weight updates
            momentum: Momentum factor for weight updates
            weight_decay: Weight decay factor for regularization
            **kwargs: Additional arguments passed to parent MCC_LearningRule
        """
        super().__init__(nu=nu, **kwargs)
        
        self.alpha_loss = alpha_loss
        self.goodness_fn = goodness_fn
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        # State tracking for Forward-Forward learning
        self.positive_goodness = None
        self.negative_goodness = None
        self.positive_activations = None
        self.negative_activations = None
        
        # Momentum state
        self.velocity = None
        
        # Sample type tracking
        self.current_sample_type = None
        self.samples_processed = 0
        
    def update(
        self,
        connection: 'MulticompartmentConnection',
        source_s: torch.Tensor,
        target_s: torch.Tensor,
        **kwargs
    ) -> None:
        """
        Perform Forward-Forward learning update.
        
        This method is called by MCC during each simulation step. It accumulates
        statistics for positive and negative samples, then applies contrastive
        updates when both sample types are available.
        
        Args:
            connection: Parent MulticompartmentConnection
            source_s: Source layer spikes [batch_size, source_neurons]
            target_s: Target layer spikes [batch_size, target_neurons]
            **kwargs: Additional arguments including 'sample_type'
        """
        # Check if learning is enabled
        if not connection.w.requires_grad:
            return
        
        # Get sample type from kwargs
        sample_type = kwargs.get('sample_type', self.current_sample_type)
        if sample_type is None:
            # Default to positive for backward compatibility
            sample_type = "positive"
        
        # Compute goodness score for current batch
        current_goodness = self._compute_goodness(target_s)
        
        # Store activations and goodness based on sample type
        if sample_type == "positive":
            self.positive_goodness = current_goodness.detach()
            self.positive_activations = {
                'source': source_s.detach(),
                'target': target_s.detach()
            }
            
        elif sample_type == "negative":
            self.negative_goodness = current_goodness.detach()
            self.negative_activations = {
                'source': source_s.detach(),
                'target': target_s.detach()
            }
            
        else:
            raise ValueError(f"Invalid sample_type: {sample_type}. Must be 'positive' or 'negative'")
        
        self.samples_processed += 1
        
        # Apply contrastive update if we have both positive and negative samples
        if (self.positive_goodness is not None and 
            self.negative_goodness is not None and
            self.positive_activations is not None and
            self.negative_activations is not None):
            
            self._apply_forward_forward_update(connection)
            self._reset_accumulated_data()
    
    def _compute_goodness(self, target_activity: torch.Tensor) -> torch.Tensor:
        """
        Compute Forward-Forward goodness score from target layer activity.
        
        Args:
            target_activity: Target neuron spikes [batch_size, neurons]
            
        Returns:
            Goodness scores [batch_size]
        """
        if self.goodness_fn == "mean_squared":
            # Mean squared activity across neurons (original FF paper)
            goodness = torch.mean(target_activity ** 2, dim=1)
            
        elif self.goodness_fn == "sum_squared":
            # Sum of squared activity across neurons
            goodness = torch.sum(target_activity ** 2, dim=1)
            
        else:
            raise ValueError(f"Unknown goodness function: {self.goodness_fn}")
        
        return goodness
    
    def _apply_forward_forward_update(self, connection: 'MulticompartmentConnection'):
        """
        Apply Forward-Forward contrastive weight update.
        
        The update follows the Forward-Forward principle:
        - Strengthen weights that increase goodness for positive samples
        - Weaken weights that increase goodness for negative samples
        
        Args:
            connection: Parent MulticompartmentConnection
        """
        # Get weight tensor
        w = connection.w
        
        # Compute Forward-Forward loss (for monitoring)
        ff_loss = self._compute_ff_loss(self.positive_goodness, self.negative_goodness)
        
        # Compute weight update based on activity correlations
        pos_source = self.positive_activations['source']
        pos_target = self.positive_activations['target']
        neg_source = self.negative_activations['source']
        neg_target = self.negative_activations['target']
        
        # Positive update: strengthen weights for positive samples
        # ΔW_pos = η * s_pos^T * t_pos / batch_size
        delta_w_pos = torch.mm(pos_source.t(), pos_target) / pos_source.shape[0]
        
        # Negative update: weaken weights for negative samples  
        # ΔW_neg = -η * s_neg^T * t_neg / batch_size
        delta_w_neg = -torch.mm(neg_source.t(), neg_target) / neg_source.shape[0]
        
        # Combined Forward-Forward update
        delta_w = self.nu * (delta_w_pos + delta_w_neg)
        
        # Add weight decay if specified
        if self.weight_decay > 0:
            delta_w = delta_w - self.weight_decay * w
        
        # Apply momentum if specified
        if self.momentum > 0:
            if self.velocity is None:
                self.velocity = torch.zeros_like(w)
            
            self.velocity = self.momentum * self.velocity + delta_w
            delta_w = self.velocity
        
        # Apply weight update
        with torch.no_grad():
            w.add_(delta_w)
        
        # Apply weight constraints if they exist
        self._apply_weight_constraints(connection)
    
    def _compute_ff_loss(
        self,
        goodness_pos: torch.Tensor,
        goodness_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Forward-Forward contrastive loss for monitoring.
        
        L = log(1 + exp(-g_pos + α)) + log(1 + exp(g_neg - α))
        
        Args:
            goodness_pos: Goodness scores for positive samples
            goodness_neg: Goodness scores for negative samples
            
        Returns:
            Forward-Forward loss (scalar)
        """
        # Positive loss: encourage high goodness for positive samples
        loss_pos = torch.log(1 + torch.exp(-goodness_pos + self.alpha_loss))
        
        # Negative loss: encourage low goodness for negative samples
        loss_neg = torch.log(1 + torch.exp(goodness_neg - self.alpha_loss))
        
        # Return mean loss across batch
        total_loss = loss_pos + loss_neg
        return torch.mean(total_loss)
    
    def _apply_weight_constraints(self, connection: 'MulticompartmentConnection'):
        """
        Apply weight constraints (bounds, normalization) if specified.
        
        Args:
            connection: Parent connection with constraint parameters
        """
        w = connection.w
        
        # Apply weight bounds if specified
        if hasattr(connection, 'wmin') and hasattr(connection, 'wmax'):
            with torch.no_grad():
                w.clamp_(connection.wmin, connection.wmax)
        
        # Apply normalization if specified
        if hasattr(connection, 'norm') and connection.norm is not None:
            with torch.no_grad():
                if connection.norm == "l2":
                    # L2 normalize each output neuron's weights
                    w.div_(w.norm(dim=0, keepdim=True) + 1e-8)
                elif connection.norm == "l1":
                    # L1 normalize each output neuron's weights
                    w.div_(w.abs().sum(dim=0, keepdim=True) + 1e-8)
    
    def _reset_accumulated_data(self):
        """Reset accumulated positive and negative sample data."""
        self.positive_goodness = None
        self.negative_goodness = None
        self.positive_activations = None
        self.negative_activations = None
    
    def set_sample_type(self, sample_type: str):
        """
        Set the current sample type for subsequent updates.
        
        Args:
            sample_type: Either "positive" or "negative"
        """
        if sample_type not in ["positive", "negative"]:
            raise ValueError(f"Invalid sample_type: {sample_type}")
        
        self.current_sample_type = sample_type
    
    def get_goodness_scores(self) -> dict:
        """Get current goodness scores for positive and negative samples."""
        return {
            'positive_goodness': self.positive_goodness,
            'negative_goodness': self.negative_goodness
        }
    
    def get_ff_loss(self) -> torch.Tensor:
        """Compute and return current Forward-Forward loss if data available."""
        if self.positive_goodness is not None and self.negative_goodness is not None:
            return self._compute_ff_loss(self.positive_goodness, self.negative_goodness)
        else:
            return torch.tensor(0.0)
    
    def reset_state(self):
        """Reset all learning rule state."""
        self._reset_accumulated_data()
        self.velocity = None
        self.current_sample_type = None
        self.samples_processed = 0
    
    def get_learning_stats(self) -> dict:
        """Get learning rule statistics and configuration."""
        return {
            'learning_rule_type': 'ForwardForwardMCCLearning',
            'alpha_loss': self.alpha_loss,
            'goodness_fn': self.goodness_fn,
            'learning_rate': self.nu,
            'momentum': self.momentum,
            'weight_decay': self.weight_decay,
            'samples_processed': self.samples_processed,
            'current_sample_type': self.current_sample_type,
            'has_positive_data': self.positive_goodness is not None,
            'has_negative_data': self.negative_goodness is not None
        }
    
    def __repr__(self):
        """String representation of the learning rule."""
        return (
            f"ForwardForwardMCCLearning("
            f"nu={self.nu}, "
            f"alpha_loss={self.alpha_loss}, "
            f"goodness_fn='{self.goodness_fn}', "
            f"momentum={self.momentum}, "
            f"weight_decay={self.weight_decay})"
        )