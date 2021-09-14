from abc import ABC, abstractmethod
from typing import Any, Dict, Tuple

import gym
import numpy as np
import torch

from bindsnet.datasets.preprocess import binary_image, crop, gray_scale, subsample
from bindsnet.encoding import Encoder, NullEncoder


class Environment(ABC):
    # language=rst
    """
    Abstract environment class.
    """

    @abstractmethod
    def step(self, a: int) -> Tuple[Any, ...]:
        # language=rst
        """
        Abstract method head for ``step()``.

        :param a: Integer action to take in environment.
        """

    @abstractmethod
    def reset(self) -> None:
        # language=rst
        """
        Abstract method header for ``reset()``.
        """

    @abstractmethod
    def render(self) -> None:
        # language=rst
        """
        Abstract method header for ``render()``.
        """

    @abstractmethod
    def close(self) -> None:
        # language=rst
        """
        Abstract method header for ``close()``.
        """

    @abstractmethod
    def preprocess(self) -> None:
        # language=rst
        """
        Abstract method header for ``preprocess()``.
        """


class GymEnvironment(Environment):
    # language=rst
    """
    A wrapper around the OpenAI ``gym`` environments.
    """

    def __init__(self, name: str, encoder: Encoder = NullEncoder(), **kwargs) -> None:
        # language=rst
        """
        Initializes the environment wrapper. This class makes the
        assumption that the OpenAI ``gym`` environment will provide an image
        of format HxW or CxHxW as an observation (we will add the C
        dimension to HxW tensors) or a 1D observation in which case no
        dimensions will be added.

        :param name: The name of an OpenAI ``gym`` environment.
        :param encoder: Function to encode observations into spike trains.

        Keyword arguments:

        :param float max_prob: Maximum spiking probability.
        :param bool clip_rewards: Whether or not to use ``np.sign`` of rewards.

        :param int history: Number of observations to keep track of.
        :param int delta: Step size to save observations in history.
        :param bool add_channel_dim: Allows for the adding of the channel dimension in
            2D inputs.
        """
        self.name = name
        self.env = gym.make(name)
        self.action_space = self.env.action_space

        self.encoder = encoder

        # Keyword arguments.
        self.max_prob = kwargs.get("max_prob", 1.0)
        self.clip_rewards = kwargs.get("clip_rewards", True)

        self.history_length = kwargs.get("history_length", None)
        self.delta = kwargs.get("delta", 1)
        self.add_channel_dim = kwargs.get("add_channel_dim", True)

        if self.history_length is not None and self.delta is not None:
            self.history = {
                i: torch.Tensor()
                for i in range(1, self.history_length * self.delta + 1, self.delta)
            }
        else:
            self.history = {}

        self.episode_step_count = 0
        self.history_index = 1

        self.obs = None
        self.reward = None

        assert (
            0.0 < self.max_prob <= 1.0
        ), "Maximum spiking probability must be in (0, 1]."

    def step(self, a: int) -> Tuple[torch.Tensor, float, bool, Dict[Any, Any]]:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``step()`` function.

        :param a: Action to take in the environment.
        :return: Observation, reward, done flag, and information dictionary.
        """
        # Call gym's environment step function.
        self.obs, self.reward, self.done, info = self.env.step(a)

        if self.clip_rewards:
            self.reward = np.sign(self.reward)

        self.preprocess()

        # Add the raw observation from the gym environment into the info
        # for debugging and display.
        info["gym_obs"] = self.obs

        # Store frame of history and encode the inputs.
        if len(self.history) > 0:
            self.update_history()
            self.update_index()
            # Add the delta observation into the info for debugging and display.
            info["delta_obs"] = self.obs

        # The new standard for images is BxTxCxHxW.
        # The gym environment doesn't follow exactly the same protocol.
        #
        # 1D observations will be left as is before the encoder and will become BxTxL.
        # 2D observations are assumed to be mono images will become BxTx1xHxW
        # 3D observations will become BxTxCxHxW
        if self.obs.dim() == 2 and self.add_channel_dim:
            # We want CxHxW, it is currently HxW.
            self.obs = self.obs.unsqueeze(0)

        # The encoder will add time - now Tx...
        if self.encoder is not None:
            self.obs = self.encoder(self.obs)

        # Add the batch - now BxTx...
        self.obs = self.obs.unsqueeze(0)

        self.episode_step_count += 1

        # Return converted observations and other information.
        return self.obs, self.reward, self.done, info

    def reset(self) -> torch.Tensor:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``reset()`` function.

        :return: Observation from the environment.
        """
        # Call gym's environment reset function.
        self.obs = self.env.reset()
        self.preprocess()

        self.history = {i: torch.Tensor() for i in self.history}

        self.episode_step_count = 0

        return self.obs

    def render(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``render()`` function.
        """
        self.env.render()

    def close(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI ``gym`` environment ``close()`` function.
        """
        self.env.close()

    def preprocess(self) -> None:
        # language=rst
        """
        Pre-processing step for an observation from a ``gym`` environment.
        """
        if self.name == "SpaceInvaders-v0":
            self.obs = subsample(gray_scale(self.obs), 84, 110)
            self.obs = self.obs[26:104, :]
            self.obs = binary_image(self.obs)
        elif self.name == "BreakoutDeterministic-v4":
            self.obs = subsample(gray_scale(crop(self.obs, 34, 194, 0, 160)), 80, 80)
            self.obs = binary_image(self.obs)
        else:  # Default pre-processing step.
            pass

        self.obs = torch.from_numpy(self.obs).float()

    def update_history(self) -> None:
        # language=rst
        """
        Updates the observations inside history by performing subtraction from most
        recent observation and the sum of previous observations. If there are not enough
        observations to take a difference from, simply store the observation without any
        differencing.
        """
        # Recording initial observations.
        if self.episode_step_count < len(self.history) * self.delta:
            # Store observation based on delta value.
            if self.episode_step_count % self.delta == 0:
                self.history[self.history_index] = self.obs
        else:
            # Take difference between stored frames and current frame.
            temp = torch.clamp(self.obs - sum(self.history.values()), 0, 1)

            # Store observation based on delta value.
            if self.episode_step_count % self.delta == 0:
                self.history[self.history_index] = self.obs

            assert (
                len(self.history) == self.history_length
            ), "History size is out of bounds"
            self.obs = temp

    def update_index(self) -> None:
        # language=rst
        """
        Updates the index to keep track of history. For example: ``history = 4``,
        ``delta = 3`` will produce ``self.history = {1, 4, 7, 10}`` and
        ``self.history_index`` will be updated according to ``self.delta`` and will wrap
        around the history dictionary.
        """
        if self.episode_step_count % self.delta == 0:
            if self.history_index != max(self.history.keys()):
                self.history_index += self.delta
            else:
                # Wrap around the history.
                self.history_index = (self.history_index % max(self.history.keys())) + 1
