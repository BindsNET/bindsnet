from abc import ABC, abstractmethod
from bindsnet.learning.learning import NoOp
from typing import Union, Tuple, Optional, Sequence, TYPE_CHECKING

import numpy as np
import torch

if TYPE_CHECKING:
    from bindsnet.network.topology import MulticompartmentConnection
from torch import device
from torch.nn import Parameter
import torch.nn.functional as F
import torch.nn as nn
import bindsnet.learning

class AbstractFeature(ABC):
    # language=rst
    """
    Features to operate on signals traversing a connection.
    """

    @abstractmethod
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        range: Optional[Union[list, tuple]] = None,
        clamp_frequency: Optional[int] = 1,
        norm: Optional[Union[torch.Tensor, float, int]] = None,
        learning_rule: Optional[bindsnet.learning.LearningRule] = None,
        nu: Optional[Union[list, tuple, int, float]] = None,
        reduction: Optional[callable] = None,
        enforce_polarity: Optional[bool] = False,
        decay: float = 0.0,
        parent_feature=None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Feature` object. Will assign all incoming arguments as class variables
        :param name: Name of the feature
        :param value: Core numeric object for the feature. This parameters function will vary depending on the feature
        :param range: Range of acceptable values for the :code:`value` parameter
        :param norm: Value which all values in :code:`value` will sum to. Normalization of values occurs after each
            sample and after the value has been updated by the learning rule (if there is one)
        :param learning_rule: Rule which will modify the :code:`value` after each sample
        :param nu: Learning rate for the learning rule
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension
        :param decay: Constant multiple to decay weights by on each iteration
        :param parent_feature: Parent feature to inherit :code:`value` from
        """

        #### Initialize class variables ####
        ## Args ##
        self.name = name
        self.value = value
        self.range = [-1.0, 1.0] if range is None else range
        self.clamp_frequency = clamp_frequency
        self.norm = norm
        self.learning_rule = learning_rule
        self.nu = nu
        self.reduction = reduction
        self.decay = decay
        self.parent_feature = parent_feature
        self.kwargs = kwargs

        ## Backend ##
        self.is_primed = False

        from ..learning.MCC_learning import (
            NoOp,
            PostPre,
            MSTDP,
            MSTDPET,
        )

        supported_rules = [
            NoOp,
            PostPre,
            MSTDP,
            MSTDPET,
        ]

        #### Assertions ####
        # Assert correct instance of feature values
        assert isinstance(name, str), "Feature {0}'s name should be of type str".format(
            name
        )
        assert value is None or isinstance(
            value, (torch.Tensor, float, int)
        ), "Feature {0} should be of type float, int, or torch.Tensor, not {1}".format(
            name, type(value)
        )
        assert norm is None or isinstance(
            norm, (torch.Tensor, float, int)
        ), "Feature {0}'s norm should be of type float, int, or torch.Tensor, not {1}".format(
            name, type(norm)
        )
        assert learning_rule is None or (
            learning_rule in supported_rules
        ), "Feature {0}'s learning_rule should be of type bindsnet.LearningRule not {1}".format(
            name, type(learning_rule)
        )
        assert nu is None or isinstance(
            nu, (list, tuple)
        ), "Feature {0}'s nu should be of type list or tuple, not {1}".format(
            name, type(nu)
        )
        assert reduction is None or isinstance(
            reduction, callable
        ), "Feature {0}'s reduction should be of type callable, not {1}".format(
            name, type(reduction)
        )
        assert decay is None or isinstance(
            decay, float
        ), "Feature {0}'s decay should be of type float, not {1}".format(
            name, type(decay)
        )

        self.assert_valid_range()
        if value is not None:
            self.assert_feature_in_range()

    @abstractmethod
    def reset_state_variables(self) -> None:
        # language=rst
        """
        Contains resetting logic for the feature.
        """
        if self.learning_rule:
            self.learning_rule.reset_state_variables()
        pass

    @abstractmethod
    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:
        # language=rst
        """
        Computes the feature being operated on a set of incoming signals.
        """
        pass

    def prime_feature(self, connection, device, **kwargs) -> None:
        # language=rst
        """
        Prepares a feature after it has been placed in a connection. This takes care of learning rules, feature
        value initialization, and asserting that features have proper shape. Should occur after primary constructor.
        """

        # Note: DO NOT move NoOp to global; cyclical dependency
        from ..learning.MCC_learning import NoOp

        # Check if feature is already primed
        if self.is_primed:
            return
        self.is_primed = True

        # Check if feature is a child feature
        if self.parent_feature is not None:
            self.link(self.parent_feature)
            self.learning_rule = NoOp(connection=connection)
            return

        # Check if values/norms are the correct shape
        if isinstance(self.value, torch.Tensor):
            assert tuple(self.value.shape) == (connection.source.n, connection.target.n)

        if self.norm is not None and isinstance(self.norm, torch.Tensor):
            assert self.norm.shape[0] == connection.target.n

        #### Initialize feature value ####
        if self.value is None:
            self.value = (
                self.initialize_value()
            )  # This should be defined per the sub-class

        if isinstance(self.value, (int, float)):
            self.value = torch.Tensor([self.value])

        # Parameterize and send to proper device
        # Note: Floating is used here to avoid dtype conflicts
        self.value = Parameter(self.value, requires_grad=False).to(device)

        ##### Initialize learning rule #####

        # Default is NoOp
        if self.learning_rule is None:
            self.learning_rule = NoOp

        self.learning_rule = self.learning_rule(
            connection=connection,
            feature_value=self.value,
            range=self.range,
            nu=self.nu,
            reduction=self.reduction,
            decay=self.decay,
            **kwargs,
        )

        #### Recycle unnecessary variables ####
        del self.nu, self.reduction, self.decay, self.range

    def update(self, **kwargs) -> None:
        # language=rst
        """
        Compute feature's update rule
        """

        self.learning_rule.update(**kwargs)

    def normalize(self) -> None:
        # language=rst
        """
        Normalize feature so each target neuron has sum of feature values equal to
        ``self.norm``.
        """

        if self.norm is not None:
            abs_sum = self.value.sum(0).unsqueeze(0)
            abs_sum[abs_sum == 0] = 1.0
            self.value *= self.norm / abs_sum

    def degrade(self) -> None:
        # language=rst
        """
        Degrade the value of the propagated spikes according to the features value. A lambda function should be passed
        into the constructor which takes a single argument (which represent the value), and returns a value which will
        be *subtracted* from the propagated spikes.
        """

        return self.degrade(self.value)

    def link(self, parent_feature) -> None:
        # language=rst
        """
        Allow two features to share tensor values
        """

        valid_features = (Probability, Weight, Bias, Intensity)

        assert isinstance(self, valid_features), f"A {self} cannot use feature linking"
        assert isinstance(
            parent_feature, valid_features
        ), f"A {parent_feature} cannot use feature linking"
        assert self.is_primed, f"Prime feature before linking: {self}"
        assert (
            parent_feature.is_primed
        ), f"Prime parent feature before linking: {parent_feature}"

        # Link values, disable learning for this feature
        self.value = parent_feature.value
        self.learning_rule = NoOp

    def assert_valid_range(self):
        # language=rst
        """
        Default range verifier (within [-1, +1])
        """

        r = self.range

        ## Check dtype ##
        assert isinstance(
            self.range, (list, tuple)
        ), f"Invalid range for feature {self.name}: range should be a list or tuple, not {type(self.range)}"
        assert (
            len(r) == 2
        ), f"Invalid range for feature {self.name}: range should have a length of 2"

        ## Check min/max relation ##
        if isinstance(r[0], torch.Tensor) or isinstance(r[1], torch.Tensor):
            assert (
                r[0] < r[1]
            ).all(), f"Invalid range for feature {self.name}: a min is larger than an adjacent max"
        else:
            assert (
                r[0] < r[1]
            ), f"Invalid range for feature {self.name}: the min value is larger than the max value"

    def assert_feature_in_range(self):
        r = self.range
        f = self.value

        if isinstance(r[0], torch.Tensor) or isinstance(f, torch.Tensor):
            assert (
                f >= r[0]
            ).all(), f"Feature out of range for {self.name}: Features values not in [{r[0]}, {r[1]}]"
        else:
            assert (
                f >= r[0]
            ), f"Feature out of range for {self.name}: Features values not in [{r[0]}, {r[1]}]"

        if isinstance(r[1], torch.Tensor) or isinstance(f, torch.Tensor):
            assert (
                f <= r[1]
            ).all(), f"Feature out of range for {self.name}: Features values not in [{r[0]}, {r[1]}]"
        else:
            assert (
                f <= r[1]
            ), f"Feature out of range for {self.name}: Features values not in [{r[0]}, {r[1]}]"

    def assert_valid_shape(self, source_shape, target_shape, f):
        # Multidimensional feat
        if len(f.shape) > 1:
            assert f.shape == (
                source_shape,
                target_shape,
            ), f"Feature {self.name} has an incorrect shape of {f.shape}. Should be of shape {(source_shape, target_shape)}"
        # Else assume scalar, which is a valid shape


class Probability(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        range: Optional[Sequence[float]] = None,
        norm: Optional[Union[torch.Tensor, float, int]] = None,
        learning_rule: Optional[bindsnet.learning.LearningRule] = None,
        nu: Optional[Union[list, tuple]] = None,
        reduction: Optional[callable] = None,
        decay: float = 0.0,
        parent_feature=None,
    ) -> None:
        # language=rst
        """
        Will run a bernoulli trial using :code:`value` to determine if a signal will successfully traverse the synapse
        :param name: Name of the feature
        :param value: Number(s) in [0, 1] which represent the probability of a signal traversing a synapse. Tensor values
            assume that probabilities will be matched to adjacent synapses in the connection. Scalars will be applied to
            all synapses.
        :param range: Range of acceptable values for the :code:`value` parameter. Should be in [0, 1]
        :param norm: Value which all values in :code:`value` will sum to. Normalization of values occurs after each sample
            and after the value has been updated by the learning rule (if there is one)
        :param learning_rule: Rule which will modify the :code:`value` after each sample
        :param nu: Learning rate for the learning rule
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension
        :param decay: Constant multiple to decay weights by on each iteration
        :param parent_feature: Parent feature to inherit :code:`value` from
        """

        ### Assertions ###
        super().__init__(
            name=name,
            value=value,
            range=[0, 1] if range is None else range,
            norm=norm,
            learning_rule=learning_rule,
            nu=nu,
            reduction=reduction,
            decay=decay,
            parent_feature=parent_feature,
        )

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:
        return conn_spikes * torch.bernoulli(self.value)

    def reset_state_variables(self) -> None:
        pass

    def prime_feature(self, connection, device, **kwargs) -> None:
        ## Initialize value ###
        if self.value is None:
            self.initialize_value = lambda: torch.clamp(
                torch.rand(connection.source.n, connection.target.n, device=device),
                self.range[0],
                self.range[1],
            )

        super().prime_feature(connection, device, **kwargs)

    def assert_valid_range(self):
        super().assert_valid_range()

        r = self.range

        ## Check min greater than 0 ##
        if isinstance(r[0], torch.Tensor):
            assert (
                r[0] >= 0
            ).all(), (
                f"Invalid range for feature {self.name}: a min value is less than 0"
            )
        elif isinstance(r[0], (float, int)):
            assert (
                r[0] >= 0
            ), f"Invalid range for feature {self.name}: the min value is less than 0"
        else:
            assert (
                False
            ), f"Invalid range for feature {self.name}: the min value must be of type torch.Tensor, float, or int"


class Mask(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
    ) -> None:
        # language=rst
        """
        Boolean mask which determines whether or not signals are allowed to traverse certain synapses.
        :param name: Name of the feature
        :param value: Boolean mask. :code:`True` means a signal can pass, :code:`False` means the synapse is impassable
        """

        ### Assertions ###
        if isinstance(value, torch.Tensor):
            assert (
                value.dtype == torch.bool
            ), "Mask must be of type bool, not {}".format(value.dtype)
        elif value is not None:
            assert isinstance(value, bool), "Mask must be of type bool, not {}".format(
                value.dtype
            )

            # Send boolean to tensor (priming wont work if it's not a tensor)
            value = torch.tensor(value)

        super().__init__(
            name=name,
            value=value,
        )

        self.name = name
        self.value = value

    def compute(self, conn_spikes) -> torch.Tensor:
        return conn_spikes * self.value

    def reset_state_variables(self) -> None:
        pass

    def prime_feature(self, connection, device, **kwargs) -> None:
        # Check if feature is already primed
        if self.is_primed:
            return
        self.is_primed = True

        #### Initialize feature value ####
        if self.value is None:
            self.value = (
                torch.rand(connection.source.n, connection.target.n) > 0.99
            ).to(device=device)
        self.value = Parameter(self.value, requires_grad=False).to(device)

        #### Assertions ####
        # Check if tensor values are the correct shape
        if isinstance(self.value, torch.Tensor):
            self.assert_valid_shape(
                connection.source.n, connection.target.n, self.value
            )

        ##### Initialize learning rule #####
        # Note: DO NOT move NoOp to global; cyclical dependency
        from ..learning.MCC_learning import NoOp

        # Default is NoOp
        if self.learning_rule is None:
            self.learning_rule = NoOp

        self.learning_rule = self.learning_rule(
            connection=connection,
            feature=self.value,
            range=self.range,
            nu=self.nu,
            reduction=self.reduction,
            decay=self.decay,
            **kwargs,
        )


class MeanField(AbstractFeature):
    def __init__(self) -> None:
        # language=rst
        """
        Takes the mean of all outgoing signals, and outputs that mean across every synapse in the connection
        """
        pass

    def reset_state_variables(self) -> None:
        pass

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:
        return conn_spikes.mean() * torch.ones(
            self.source_n * self.target_n, device=self.device
        )

    def prime_feature(self, connection, device, **kwargs) -> None:
        self.source_n = connection.source.n
        self.target_n = connection.target.n

        super().prime_feature(connection, device, **kwargs)


class Weight(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        range: Optional[Sequence[float]] = None,
        norm: Optional[Union[torch.Tensor, float, int]] = None,
        norm_frequency: Optional[str] = "sample",
        learning_rule: Optional[bindsnet.learning.LearningRule] = None,
        nu: Optional[Union[list, tuple]] = None,
        reduction: Optional[callable] = None,
        enforce_polarity: Optional[bool] = False,
        decay: float = 0.0,
    ) -> None:
        # language=rst
        """
        Multiplies signals by scalars
        :param name: Name of the feature
        :param value: Values to scale signals by
        :param range: Range of acceptable values for the :code:`value` parameter
        :param norm: Value which all values in :code:`value` will sum to. Normalization of values occurs after each sample
            and after the value has been updated by the learning rule (if there is one)
        :param norm_frequency: How often to normalize weights:
            * 'sample': weights normalized after each sample
            * 'time step': weights normalized after each time step
        :param learning_rule: Rule which will modify the :code:`value` after each sample
        :param nu: Learning rate for the learning rule
        :param reduction: Method for reducing parameter updates along the minibatch
            dimension
        :param enforce_polarity: Will prevent synapses from changing signs if :code:`True`
        :param decay: Constant multiple to decay weights by on each iteration
        """

        self.norm_frequency = norm_frequency
        self.enforce_polarity = enforce_polarity
        super().__init__(
            name=name,
            value=value,
            range=[-torch.inf, +torch.inf] if range is None else range,
            norm=norm,
            learning_rule=learning_rule,
            nu=nu,
            reduction=reduction,
            decay=decay,
        )

    def reset_state_variables(self) -> None:
        pass

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:
        if self.enforce_polarity:
            pos_mask = ~torch.logical_xor(self.value > 0, self.positive_mask)
            neg_mask = ~torch.logical_xor(self.value < 0, ~self.positive_mask)
            self.value = self.value * torch.logical_or(pos_mask, neg_mask)
            self.value[~pos_mask] = 0.0001
            self.value[~neg_mask] = -0.0001

        return_val = self.value * conn_spikes
        if self.norm_frequency == "time step":
            self.normalize(time_step_norm=True)

        return return_val

    def prime_feature(self, connection, device, **kwargs) -> None:
        #### Initialize value ####
        if self.value is None:
            self.initialize_value = lambda: torch.rand(
                connection.source.n, connection.target.n
            )

        super().prime_feature(
            connection, device, enforce_polarity=self.enforce_polarity, **kwargs
        )
        if self.enforce_polarity:
            self.positive_mask = ((self.value > 0).sum(1) / self.value.shape[1]) > 0.5
            tmp = torch.zeros_like(self.value)
            tmp[self.positive_mask, :] = 1
            self.positive_mask = tmp.bool()

    def normalize(self, time_step_norm=False) -> None:
        # 'time_step_norm' will indicate if normalize is being called from compute()
        # or from network.py (after a sample is completed)

        if self.norm_frequency == "time step" and time_step_norm:
            super().normalize()

        if self.norm_frequency == "sample" and not time_step_norm:
            super().normalize()


class Bias(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        range: Optional[Sequence[float]] = None,
        norm: Optional[Union[torch.Tensor, float, int]] = None,
    ) -> None:
        # language=rst
        """
        Adds scalars to signals
        :param name: Name of the feature
        :param value: Values to add to the signals
        :param range: Range of acceptable values for the :code:`value` parameter
        :param norm: Value which all values in :code:`value` will sum to. Normalization of values occurs after each sample
            and after the value has been updated by the learning rule (if there is one)
        """

        super().__init__(
            name=name,
            value=value,
            range=[-torch.inf, +torch.inf] if range is None else range,
            norm=norm,
        )

    def reset_state_variables(self) -> None:
        pass

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:
        return conn_spikes + self.value

    def prime_feature(self, connection, device, **kwargs) -> None:
        #### Initialize value ####
        if self.value is None:
            self.initialize_value = lambda: torch.rand(
                connection.source.n, connection.target.n
            )

        super().prime_feature(connection, device, **kwargs)


class Intensity(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        range: Optional[Sequence[float]] = None,
    ) -> None:
        # language=rst
        """
        Adds scalars to signals
        :param name: Name of the feature
        :param value: Values to scale signals by
        """

        super().__init__(name=name, value=value, range=range)

    def reset_state_variables(self) -> None:
        pass

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:
        return conn_spikes * self.value

    def prime_feature(self, connection, device, **kwargs) -> None:
        #### Initialize value ####
        if self.value is None:
            self.initialize_value = lambda: torch.clamp(
                torch.sign(
                    torch.randint(-1, +2, (connection.source.n, connection.target.n))
                ),
                self.range[0],
                self.range[1],
            )

        super().prime_feature(connection, device, **kwargs)


class Degradation(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        degrade_function: callable = None,
        parent_feature: Optional[AbstractFeature] = None,
    ) -> None:
        # language=rst
        """
        Degrades propagating spikes according to :code:`degrade_function`.
        Note: If :code:`parent_feature` is provided, it will override :code:`value`.
        :param name: Name of the feature
        :param value: Value used to degrade feature
        :param degrade_function: Callable function which takes a single argument (:code:`value`) and returns a tensor or
        constant to be *subtracted* from the propagating spikes.
        :param parent_feature: Parent feature with desired :code:`value` to inherit
        """

        # Note: parent_feature will override value. See abstract constructor
        super().__init__(name=name, value=value, parent_feature=parent_feature)

        self.degrade_function = degrade_function

    def reset_state_variables(self) -> None:
        pass

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:
        return conn_spikes - self.degrade_function(self.value)


class AdaptationBaseSynapsHistory(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        ann_values: Union[list, tuple] = None,
        const_update_rate: float = 0.1,
        const_decay: float = 0.001,
    ) -> None:
        # language=rst
        """
        The ANN will be use on each synaps to messure the previous activity of the neuron and descide to close or open connection.

        :param name: Name of the feature
        :param ann_values: Values to be use to build an ANN that will adapt the connectivity of the layer.
        :param value: Values to be use to build an initial mask for the synapses.
        :param const_update_rate: The mask upatate rate of the ANN decision.
        :param const_decay: The spontaneous activation of the synapses.
        """

        # Define the ANN
        class ANN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(ANN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
                self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.tanh(self.fc2(x))  # MUST HAVE output between -1 and 1
                return x

        self.init_value = value.clone().detach()  # initial mask
        self.mask = value  # final decision of the ANN
        value = torch.zeros_like(value)  # initial mask
        self.ann = ANN(ann_values[0].shape[0], ann_values[0].shape[1], 1)

        # load weights from ann_values
        with torch.no_grad():
            self.ann.fc1.weight.data = ann_values[0]
            self.ann.fc2.weight.data = ann_values[1]
        self.ann.to(ann_values[0].device)

        self.spike_buffer = torch.zeros(
            (value.numel(), ann_values[0].shape[1]),
            device=ann_values[0].device,
            dtype=torch.bool,
        )
        self.counter = 0
        self.start_counter = False
        self.const_update_rate = const_update_rate
        self.const_decay = const_decay

        super().__init__(name=name, value=value)

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:

        # Update the spike buffer
        if self.start_counter == False or conn_spikes.sum() > 0:
            self.start_counter = True
            self.spike_buffer[:, self.counter % self.spike_buffer.shape[1]] = (
                conn_spikes.flatten()
            )
            self.counter += 1

        # Update the masks
        if self.counter % self.spike_buffer.shape[1] == 0:
            with torch.no_grad():
                ann_decision = self.ann(self.spike_buffer.to(torch.float32))
            self.mask += (
                ann_decision.view(self.mask.shape) * self.const_update_rate
            )  # update mask with learning rate fraction
            self.mask += self.const_decay  # spontaneous activate synapses
            self.mask = torch.clamp(self.mask, -1, 1)  # cap the mask

            # self.mask = torch.clamp(self.mask, -1, 1)
            self.value = (self.mask > 0).float()

        return conn_spikes * self.value

    def reset_state_variables(
        self,
    ):
        self.spike_buffer = torch.zeros_like(self.spike_buffer)
        self.counter = 0
        self.start_counter = False
        self.value = self.init_value.clone().detach()  # initial mask
        pass


class AdaptationBaseOtherSynaps(AbstractFeature):
    def __init__(
        self,
        name: str,
        value: Union[torch.Tensor, float, int] = None,
        ann_values: Union[list, tuple] = None,
        const_update_rate: float = 0.1,
        const_decay: float = 0.01,
    ) -> None:
        # language=rst
        """
        The ANN will be use on each synaps to messure the previous activity of the neuron and descide to close or open connection.

        :param name: Name of the feature
        :param ann_values: Values to be use to build an ANN that will adapt the connectivity of the layer.
        :param value: Values to be use to build an initial mask for the synapses.
        :param const_update_rate: The mask upatate rate of the ANN decision.
        :param const_decay: The spontaneous activation of the synapses.
        """

        # Define the ANN
        class ANN(nn.Module):
            def __init__(self, input_size, hidden_size, output_size):
                super(ANN, self).__init__()
                self.fc1 = nn.Linear(input_size, hidden_size, bias=False)
                self.fc2 = nn.Linear(hidden_size, output_size, bias=False)

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.tanh(self.fc2(x))  # MUST HAVE output between -1 and 1
                return x

        self.init_value = value.clone().detach()  # initial mask
        self.mask = value  # final decision of the ANN
        value = torch.zeros_like(value)  # initial mask
        self.ann = ANN(ann_values[0].shape[0], ann_values[0].shape[1], 1)

        # load weights from ann_values
        with torch.no_grad():
            self.ann.fc1.weight.data = ann_values[0]
            self.ann.fc2.weight.data = ann_values[1]
        self.ann.to(ann_values[0].device)

        self.spike_buffer = torch.zeros(
            (value.numel(), ann_values[0].shape[1]),
            device=ann_values[0].device,
            dtype=torch.bool,
        )
        self.counter = 0
        self.start_counter = False
        self.const_update_rate = const_update_rate
        self.const_decay = const_decay

        super().__init__(name=name, value=value)

    def compute(self, conn_spikes) -> Union[torch.Tensor, float, int]:

        # Update the spike buffer
        if self.start_counter == False or conn_spikes.sum() > 0:
            self.start_counter = True
            self.spike_buffer[:, self.counter % self.spike_buffer.shape[1]] = (
                conn_spikes.flatten()
            )
            self.counter += 1

        # Update the masks
        if self.counter % self.spike_buffer.shape[1] == 0:
            with torch.no_grad():
                ann_decision = self.ann(self.spike_buffer.to(torch.float32))
            self.mask += (
                ann_decision.view(self.mask.shape) * self.const_update_rate
            )  # update mask with learning rate fraction
            self.mask += self.const_decay  # spontaneous activate synapses
            self.mask = torch.clamp(self.mask, -1, 1)  # cap the mask

            # self.mask = torch.clamp(self.mask, -1, 1)
            self.value = (self.mask > 0).float()

        return conn_spikes * self.value

    def reset_state_variables(
        self,
    ):
        self.spike_buffer = torch.zeros_like(self.spike_buffer)
        self.counter = 0
        self.start_counter = False
        self.value = self.init_value.clone().detach()  # initial mask
        pass


### Sub Features ###


class AbstractSubFeature(ABC):
    # language=rst
    """
    A way to inject a features methods (like normalization, learning, etc.) into the pipeline for user controlled
    execution.
    """

    @abstractmethod
    def __init__(
        self,
        name: str,
        parent_feature: AbstractFeature,
    ) -> None:
        # language=rst
        """
        Instantiates a :code:`Augment` object. Will assign all incoming arguments as class variables.
        :param name: Name of the augment
        :param parent_feature: Primary feature which the augment will modify
        """

        self.name = name
        self.parent = parent_feature
        self.sub_feature = None  # <-- Defined in non-abstract constructor

    def compute(self, _) -> None:
        # language=rst
        """
        Proxy function to catch a pipeline execution from topology.py's :code:`compute` function. Allows :code:`SubFeature`
        objects to be executed like real features in the pipeline.
        """

        # sub_feature should be defined in the non-abstract constructor
        self.sub_feature()


class Normalization(AbstractSubFeature):
    # language=rst
    """
    Normalize parent features values so each target neuron has sum of feature values equal to a desired value :code:`norm`.
    """
    def __init__(
        self,
        name: str,
        parent_feature: AbstractFeature,
    ) -> None:
        super().__init__(name, parent_feature)

        self.sub_feature = self.parent.normalize


class Updating(AbstractSubFeature):
    # language=rst
    """
    Update parent features values using the assigned update rule.
    """

    def __init__(
        self,
        name: str,
        parent_feature: AbstractFeature,
    ) -> None:
        super().__init__(name, parent_feature)

        self.sub_feature = self.parent.update




#FF Related
class ArctangentSurrogateFeature(AbstractFeature):
    """
    Arctangent surrogate gradient feature for spiking neural networks.
    
    This feature handles the complete spiking computation pipeline using
    arctangent surrogate gradients to enable backpropagation through
    discrete spike events.
    """
    
    def __init__(
        self,
        spike_threshold: float = 1.0,
        alpha: float = 2.0,
        dt: float = 1.0,
        reset_mechanism: str = "subtract",
        **kwargs
    ):
        """
        Initialize arctangent surrogate feature.
        
        Args:
            spike_threshold: Voltage threshold for spike generation
            alpha: Steepness parameter for surrogate gradient (higher = steeper)
            dt: Integration time step
            reset_mechanism: Post-spike reset ("subtract", "zero", "none")
            **kwargs: Additional arguments for AbstractFeature
        """
        super().__init__(**kwargs)
        
        self.spike_threshold = spike_threshold
        self.alpha = alpha
        self.dt = dt
        self.reset_mechanism = reset_mechanism
        
        # State variables
        self.v_membrane = None
        self.batch_size = None
        self.target_size = None
        self.initialized = False
        
    def compute(
        self, 
        connection: 'MulticompartmentConnection',
        source_s: torch.Tensor,
        **kwargs
    ) -> torch.Tensor:
        """
        Compute forward pass with arctangent surrogate gradients.
        
        Args:
            connection: Parent MulticompartmentConnection
            source_s: Source layer spikes [batch_size, source_neurons]
            **kwargs: Additional arguments
            
        Returns:
            Target spikes with differentiable surrogate gradients [batch_size, target_neurons]
        """
        # Step 1: Compute synaptic input
        synaptic_input = torch.mm(source_s.float(), connection.w)
        
        # Step 2: Initialize membrane potential if needed
        if not self.initialized:
            self._initialize_state(synaptic_input)
        
        # Step 3: Integrate membrane potential
        self.v_membrane = self.v_membrane + synaptic_input * self.dt
        
        # Step 4: Generate spikes with arctangent surrogate gradients
        spikes = self.arctangent_surrogate_spike(
            self.v_membrane, 
            self.spike_threshold, 
            self.alpha
        )
        
        # Step 5: Apply reset mechanism
        self._apply_reset(spikes)
        
        return spikes
    
    def arctangent_surrogate_spike(
        self, 
        v: torch.Tensor, 
        threshold: float, 
        alpha: float
    ) -> torch.Tensor:
        """
        Arctangent surrogate gradient spike function.
        
        Forward pass: spikes = (v >= threshold)
        Backward pass: grad = 1 / (α * |v - threshold| + 1)
        
        Args:
            v: Membrane potential [batch_size, neurons]
            threshold: Spike threshold
            alpha: Surrogate gradient steepness parameter
            
        Returns:
            Binary spikes with differentiable surrogate gradients
        """
        return _ArctangentSurrogateSpike.apply(v, threshold, alpha)
    
    def _initialize_state(self, reference_tensor: torch.Tensor):
        """Initialize state tensors based on input dimensions."""
        self.batch_size, self.target_size = reference_tensor.shape
        self.v_membrane = torch.zeros_like(reference_tensor)
        self.initialized = True
    
    def _apply_reset(self, spikes: torch.Tensor):
        """Apply post-spike reset mechanism."""
        if self.reset_mechanism == "subtract":
            # Subtract threshold from spiking neurons
            self.v_membrane = self.v_membrane - spikes * self.spike_threshold
        elif self.reset_mechanism == "zero":
            # Reset spiking neurons to zero
            self.v_membrane = self.v_membrane * (1 - spikes)
        elif self.reset_mechanism == "none":
            # No reset - membrane potential continues to integrate
            pass
        else:
            raise ValueError(f"Unknown reset mechanism: {self.reset_mechanism}")
    
    def reset_state_variables(self):
        """Reset all internal state variables."""
        self.v_membrane = None
        self.batch_size = None
        self.target_size = None
        self.initialized = False
    
    def get_membrane_potential(self) -> Optional[torch.Tensor]:
        """Get current membrane potential."""
        return self.v_membrane
    
    def set_spike_threshold(self, threshold: float):
        """Set spike threshold."""
        self.spike_threshold = threshold
    
    def set_alpha(self, alpha: float):
        """Set surrogate gradient steepness."""
        self.alpha = alpha
    
    def get_feature_info(self) -> dict:
        """Get feature configuration information."""
        return {
            'feature_type': 'ArctangentSurrogateFeature',
            'spike_threshold': self.spike_threshold,
            'alpha': self.alpha,
            'dt': self.dt,
            'reset_mechanism': self.reset_mechanism,
            'initialized': self.initialized,
            'batch_size': self.batch_size,
            'target_size': self.target_size
        }
    
    def __repr__(self) -> str:
        """String representation."""
        return (
            f"ArctangentSurrogateFeature("
            f"spike_threshold={self.spike_threshold}, "
            f"alpha={self.alpha}, "
            f"dt={self.dt}, "
            f"reset_mechanism='{self.reset_mechanism}')"
        )


class _ArctangentSurrogateSpike(torch.autograd.Function):
    """
    Arctangent surrogate gradient autograd function.
    
    This function implements the arctangent surrogate gradient method
    for training spiking neural networks with backpropagation.
    """
    
    @staticmethod
    def forward(ctx, input_v, threshold, alpha):
        """
        Forward pass: generate binary spikes.
        
        Args:
            ctx: Context for saving variables
            input_v: Input membrane potential
            threshold: Spike threshold
            alpha: Surrogate gradient parameter
            
        Returns:
            Binary spikes (0 or 1)
        """
        # Save variables for backward pass
        ctx.save_for_backward(input_v)
        ctx.threshold = threshold
        ctx.alpha = alpha
        
        # Generate binary spikes
        spikes = (input_v >= threshold).float()
        
        return spikes
    
    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass: compute surrogate gradients.
        
        Args:
            ctx: Context with saved variables
            grad_output: Gradient from downstream layers
            
        Returns:
            Gradients w.r.t. input_v, threshold, alpha
        """
        # Retrieve saved variables
        input_v, = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = ctx.alpha
        
        # Compute arctangent surrogate gradient
        # Surrogate gradient: 1 / (α * |v - v_th| + 1)
        surrogate_grad = 1.0 / (alpha * torch.abs(input_v - threshold) + 1.0)
        
        # Apply chain rule
        grad_input = grad_output * surrogate_grad
        
        # Return gradients (input_v, threshold, alpha)
        # threshold and alpha gradients are None (not optimized)
        return grad_input, None, None


# Helper function for easy creation
def create_arctangent_surrogate_connection(
    source,
    target,
    w: Optional[torch.Tensor] = None,
    spike_threshold: float = 1.0,
    alpha: float = 2.0,
    dt: float = 1.0,
    reset_mechanism: str = "subtract",
    **mcc_kwargs
):
    """
    Create MulticompartmentConnection with ArctangentSurrogateFeature.
    
    Args:
        source: Source population
        target: Target population
        w: Weight matrix (initialized randomly if None)
        spike_threshold: Spike threshold
        alpha: Surrogate gradient steepness
        dt: Integration time step
        reset_mechanism: Post-spike reset mechanism
        **mcc_kwargs: Additional MulticompartmentConnection arguments
        
    Returns:
        MulticompartmentConnection with ArctangentSurrogateFeature
    """
    from bindsnet.network.topology import MulticompartmentConnection
    
    # Initialize weights if not provided
    if w is None:
        w = 0.1 * torch.randn(source.n, target.n)
    
    # Create arctangent surrogate feature
    surrogate_feature = ArctangentSurrogateFeature(
        spike_threshold=spike_threshold,
        alpha=alpha,
        dt=dt,
        reset_mechanism=reset_mechanism
    )
    
    # Create connection with feature
    connection = MulticompartmentConnection(
        source=source,
        target=target,
        w=w,
        features=[surrogate_feature],
        **mcc_kwargs
    )
    
    return connection


