from abc import ABC, abstractmethod
from typing import Union

import torch


class AbstractRPE(ABC):
    # language=rst
    """
    Abstract base class for reward prediction error (RPE).
    """

    def __init__(self):
        # language=rst
        """
        Constructor for abstract RPE class.
        """
        self.reward_predict = torch.tensor(0.0)  # Predicted reward (per step).
        self.reward_predict_episode = torch.tensor(0.0)  # Predicted reward per episode.
        self.rewards_predict_episode = []  # List of predicted rewards per episode (used for plotting).

    @abstractmethod
    def compute(self, reward: Union[float, torch.Tensor], **kwargs) -> torch.Tensor:
        # language=rst
        """
        Computes the RPE.

        :param reward: Current reward.
        :return: Reward prediction error.
        """
        return reward - self.reward_predict

    @abstractmethod
    def update(self, accumulated_reward: Union[float, torch.Tensor], steps: int, **kwargs) -> None:
        # language=rst
        """
        Updates internal variables needed to modify reward. Usually called once per episode.

        :param accumulated_reward: Reward accumulated over one episode.
        :param steps: Steps in that episode.
        """
        pass


class MovingAvgRPE(AbstractRPE):
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

    def compute(self, reward: Union[float, torch.Tensor], **kwargs) -> torch.Tensor:
        # language=rst
        """
        Computes the reward prediction error using EMA.

        :param reward: Current reward.
        :return: Reward prediction error.
        """
        return super().compute(reward, **kwargs)

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
