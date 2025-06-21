from abc import ABC, abstractmethod
from bindsnet.learning.learning import NoOp
from typing import Union, Tuple, Optional, Sequence

import numpy as np
import torch
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




class ForwardForwardWeight(AbstractFeature):
    """
    Forward-Forward learning weight feature for MulticompartmentConnection.
    
    Implements the Forward-Forward algorithm with surrogate gradients, enabling
    layer-wise learning without backpropagation through time. This feature adds:
    - Arctangent surrogate gradient computation
    - Membrane potential tracking for goodness scores
    - Forward-Forward loss computation capabilities
    
    Compatible with the MCC architecture and composable with other features.
    """
    
    def __init__(
        self,
        spike_threshold: float = 1.0,
        alpha: float = 2.0,
        alpha_loss: float = 0.6,
        dt: float = 1.0,
        **kwargs
    ):
        """
        Initialize Forward-Forward weight feature.
        
        Args:
            spike_threshold: Threshold for spike generation and surrogate gradient
            alpha: Arctangent surrogate gradient steepness parameter  
            alpha_loss: Forward-Forward loss threshold parameter
            dt: Time step size for membrane potential integration
            **kwargs: Additional arguments passed to parent WeightFeature
        """
        super().__init__(**kwargs)
        
        self.spike_threshold = spike_threshold
        self.alpha = alpha
        self.alpha_loss = alpha_loss
        self.dt = dt
        
        # Membrane potential state for goodness computation
        self.v_membrane = None
        
    def reset_state(self):
        """Reset membrane potential state."""
        self.v_membrane = None
        
    def forward(
        self, 
        s: torch.Tensor, 
        connection: 'MulticompartmentConnection',
        **kwargs
    ) -> torch.Tensor:
        """
        Forward pass through weight feature with surrogate gradients.
        
        This method integrates with the MCC forward pass pipeline.
        
        Args:
            s: Input spikes [batch_size, source_neurons]
            connection: Parent MulticompartmentConnection instance
            **kwargs: Additional arguments from MCC forward pass
            
        Returns:
            Weighted synaptic input with surrogate gradient computation
        """
        # Get connection weights (handled by parent MCC)
        w = connection.w
        
        # Compute synaptic input: I = s * W
        synaptic_input = torch.mm(s.float(), w)
        
        # Track this for goodness score computation if needed
        if hasattr(self, '_track_activity') and self._track_activity:
            self._last_synaptic_input = synaptic_input.detach()
        
        return synaptic_input
    
    def compute_spikes_with_surrogate(
        self, 
        synaptic_input: torch.Tensor,
        target_layer: 'AbstractPopulation'
    ) -> torch.Tensor:
        """
        Generate spikes with surrogate gradients from synaptic input.
        
        This method should be called after the weight forward pass to convert
        synaptic input to spikes using the Forward-Forward surrogate gradient.
        
        Args:
            synaptic_input: Weighted input [batch_size, target_neurons]
            target_layer: Target neuron population
            
        Returns:
            Spikes with surrogate gradients [batch_size, target_neurons]
        """
        # Initialize or update membrane potential
        if self.v_membrane is None:
            self.v_membrane = torch.zeros_like(synaptic_input)
        
        # Integrate synaptic input (simple Euler integration)
        self.v_membrane = self.v_membrane + synaptic_input * self.dt
        
        # Generate spikes using arctangent surrogate gradient
        spikes = ArctangentSurrogate.apply(
            self.v_membrane, 
            self.spike_threshold, 
            self.alpha
        )
        
        # Optional: Reset membrane potential where spikes occurred
        # self.v_membrane = self.v_membrane * (1 - spikes)
        
        return spikes
    
    def compute_goodness_score(self, spike_activity: torch.Tensor) -> torch.Tensor:
        """
        Compute Forward-Forward goodness score from spike activity.
        
        Args:
            spike_activity: Spike traces [batch_size, time_steps, neurons] or
                          spike counts [batch_size, neurons]
            
        Returns:
            Goodness scores [batch_size]
        """
        if spike_activity.dim() == 3:
            # Sum over time dimension if time traces provided
            spike_counts = torch.sum(spike_activity, dim=1)  # [batch_size, neurons]
        else:
            spike_counts = spike_activity  # Already summed
        
        # Forward-Forward goodness: mean squared spike activity
        goodness = torch.mean(spike_counts ** 2, dim=1)  # [batch_size]
        
        return goodness
    
    def compute_ff_loss(
        self,
        goodness_pos: torch.Tensor,
        goodness_neg: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute Forward-Forward contrastive loss.
        
        Loss = log(1 + exp(-g_pos + α)) + log(1 + exp(g_neg - α))
        
        Args:
            goodness_pos: Goodness scores for positive samples [batch_size]
            goodness_neg: Goodness scores for negative samples [batch_size]
            
        Returns:
            Forward-Forward loss [batch_size]
        """
        # Positive loss: encourage high goodness for true labels
        loss_pos = torch.log(1 + torch.exp(-goodness_pos + self.alpha_loss))
        
        # Negative loss: encourage low goodness for false labels
        loss_neg = torch.log(1 + torch.exp(goodness_neg - self.alpha_loss))
        
        return loss_pos + loss_neg
    
    def get_feature_info(self) -> dict:
        """Get information about this Forward-Forward feature."""
        return {
            'feature_type': 'ForwardForwardWeight',
            'spike_threshold': self.spike_threshold,
            'alpha_surrogate': self.alpha,
            'alpha_loss': self.alpha_loss,
            'dt': self.dt,
            'surrogate_function': 'arctangent',
            'compatible_with': ['MCC', 'other_weight_features']
        }


class ArctangentSurrogate(torch.autograd.Function):
    """
    Arctangent surrogate gradient function for Forward-Forward training.
    
    Forward pass: spikes = (membrane_potential >= threshold)
    Backward pass: gradient = 1 / (α * |membrane_potential - threshold| + 1)
    
    This enables gradient-based learning in spiking neural networks by
    providing a smooth approximation of the non-differentiable spike function.
    """
    
    @staticmethod
    def forward(
        ctx, 
        membrane_potential: torch.Tensor, 
        threshold: float, 
        alpha: float
    ) -> torch.Tensor:
        """
        Forward pass: generate binary spikes.
        
        Args:
            membrane_potential: Neuron membrane potentials
            threshold: Spike threshold
            alpha: Surrogate gradient steepness parameter
            
        Returns:
            Binary spike tensor (0 or 1)
        """
        # Save tensors and parameters for backward pass
        ctx.save_for_backward(membrane_potential)
        ctx.threshold = threshold
        ctx.alpha = alpha
        
        # Generate spikes (heaviside step function)
        spikes = (membrane_potential >= threshold).float()
        
        return spikes
    
    @staticmethod
    def backward(
        ctx, 
        grad_output: torch.Tensor
    ) -> Tuple[torch.Tensor, None, None]:
        """
        Backward pass: compute surrogate gradients.
        
        Uses arctangent-based surrogate: 1 / (α * |v - threshold| + 1)
        
        Args:
            grad_output: Gradient from subsequent layers
            
        Returns:
            Tuple of (grad_membrane_potential, None, None)
        """
        membrane_potential, = ctx.saved_tensors
        threshold = ctx.threshold
        alpha = ctx.alpha
        
        # Compute arctangent surrogate gradient
        # grad = 1 / (α * |v - v_th| + 1)
        surrogate_grad = 1.0 / (alpha * torch.abs(membrane_potential - threshold) + 1.0)
        
        # Apply chain rule with incoming gradients
        grad_membrane_potential = grad_output * surrogate_grad
        
        # Return gradients (only for first argument)
        return grad_membrane_potential, None, None


# Add this helper function to create FF-enabled MCC connections
def create_ff_connection(
    source: 'AbstractPopulation',
    target: 'AbstractPopulation', 
    w: Optional[torch.Tensor] = None,
    spike_threshold: float = 1.0,
    alpha: float = 2.0,
    alpha_loss: float = 0.6,
    dt: float = 1.0,
    **mcc_kwargs
) -> 'MulticompartmentConnection':
    """
    Helper function to create MulticompartmentConnection with ForwardForwardWeight feature.
    
    Args:
        source: Source neuron population
        target: Target neuron population
        w: Connection weights (if None, will be initialized)
        spike_threshold: FF spike threshold
        alpha: FF surrogate gradient parameter
        alpha_loss: FF loss threshold parameter
        dt: Time step size
        **mcc_kwargs: Additional arguments for MulticompartmentConnection
        
    Returns:
        MCC with ForwardForwardWeight feature attached
    """
    from bindsnet.network.topology import MulticompartmentConnection
    
    # Create ForwardForwardWeight feature
    ff_feature = ForwardForwardWeight(
        spike_threshold=spike_threshold,
        alpha=alpha,
        alpha_loss=alpha_loss,
        dt=dt
    )
    
    # Initialize weights if not provided
    if w is None:
        w = 0.1 * torch.randn(source.n, target.n)
    
    # Create MCC with FF feature
    connection = MulticompartmentConnection(
        source=source,
        target=target,
        w=w,
        features=[ff_feature],
        **mcc_kwargs
    )
    
    return connection