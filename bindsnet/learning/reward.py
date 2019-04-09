from abc import ABC, abstractmethod
from typing import Union

import torch


class AbstractReward(ABC):
    # language=rst
    """
    Abstract base class for reward computation.
    """

    def __init__(self):
        # language=rst
        """
        Constructor for abstract reward class.
        """
        self.reward_predict = torch.tensor(0.0)  # Predicted reward (per step).
        self.reward_predict_episode = torch.tensor(0.0)  # Predicted reward per episode.
        self.rewards_predict_episode = None  # List of predicted rewards per episode (used for plotting).

    @abstractmethod
    def compute(self, **kwargs) -> None:
        # language=rst
        """
        Computes/modifies reward.
        """
        pass

    @abstractmethod
    def update(self, accumulated_reward: Union[float, torch.Tensor], steps: int, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per episode.

        :param accumulated_reward: Reward accumulated over one episode.
        :param steps: Steps in that episode.
        """
        pass


class MovingAvgRPE(AbstractReward):
    # language=rst
    """
    Calculates reward prediction error (RPE) based on an exponential moving average (EMA) of past rewards.
    """

    def __init__(self) -> None:
        # language=rst
        """
        Constructor for EMA reward prediction error.
        """
        super().__init__()
        self.rewards_predict_episode = []

    def compute(self, **kwargs) -> torch.Tensor:
        # language=rst
        """
        Computes the reward prediction error using EMA.

        Keyword arguments:

        :param Union[float, torch.Tensor] reward: Current reward.
        :return: Reward prediction error.
        """
        # Get keyword arguments.
        reward = kwargs['reward']

        return reward - self.reward_predict

    def update(self, accumulated_reward: Union[float, torch.Tensor], steps: int, **kwargs) -> None:
        # language=rst
        """
        Updates the EMAs. Called once per episode.

        :param accumulated_reward: Reward accumulated over one episode.
        :param steps: Steps in that episode.

        Keyword arguments:

        :param float ema_window: Width of the averaging window.
        """
        # Get keyword arguments.
        ema_window = kwargs.get('ema_window', 10.0)

        # Compute average reward per step.
        reward = accumulated_reward / steps

        # Update EMAs.
        self.reward_predict = (1 - 1 / ema_window) * self.reward_predict + 1 / ema_window * reward
        self.reward_predict_episode = (1 - 1 / ema_window) * self.reward_predict_episode + \
                                      1 / ema_window * accumulated_reward
        self.rewards_predict_episode.append(self.reward_predict_episode.item())
