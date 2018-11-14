import gym
import torch
import numpy as np
import pdb

from typing import Tuple, Dict, Any
from abc import ABC, abstractmethod

from ..datasets import Dataset, MNIST, CIFAR10, CIFAR100, SpokenMNIST
from ..datasets.preprocess import subsample, gray_scale, binary_image, crop


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
        pass

    @abstractmethod
    def reset(self) -> None:
        # language=rst
        """
        Abstract method header for ``reset()``.
        """
        pass

    @abstractmethod
    def render(self) -> None:
        # language=rst
        """
        Abstract method header for ``render()``.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        # language=rst
        """
        Abstract method header for ``close()``.
        """
        pass

    @abstractmethod
    def preprocess(self) -> None:
        # language=rst
        """
        Abstract method header for ``preprocess()``.
        """
        pass

    @abstractmethod
    def reshape(self) -> None:
        # language=rst
        """
        Abstract method header for ``reshape()``.
        """
        pass


class DatasetEnvironment(Environment):
    # language=rst
    """
    A wrapper around any object from the ``datasets`` module to pass to the ``Pipeline`` object.
    """

    def __init__(self, dataset: Dataset, train: bool = True, time: int = 350, **kwargs):
        # language=rst
        """
        Initializes the environment wrapper around the dataset.

        :param dataset: Object from datasets module.
        :param train: Whether to use train or test dataset.
        :param time: Length of spike train per example.
        :param kwargs: Raw data is multiplied by this value.
        """
        self.dataset = dataset
        self.train = train
        self.time = time

        # Keyword arguments.
        self.intensity = kwargs.get('intensity', 1)
        self.max_prob = kwargs.get('max_prob', 1)

        assert 0 < self.max_prob <= 1, 'Maximum spiking probability must be in (0, 1].'

        self.obs = None

        if train:
            self.data, self.labels = self.dataset.get_train()
            self.label_loader = iter(self.labels)
        else:
            self.data, self.labels = self.dataset.get_test()
            self.label_loader = iter(self.labels)

        self.env = iter(self.data)

    def step(self, a: int = None) -> Tuple[torch.Tensor, int, bool, Dict[str, int]]:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``step()`` function.

        :param a: There is no interaction of the network the dataset.
        :return: Observation, reward (fixed to 0), done (fixed to False), and information dictionary.
        """
        try:
            # Attempt to fetch the next observation.
            self.obs = next(self.env)
        except StopIteration:
            # If out of samples, reload data and label generators.
            self.env = iter(self.data)
            self.label_loader = iter(self.labels)
            self.obs = next(self.env)

        # Preprocess observation.
        self.preprocess()

        # Info dictionary contains label of MNIST digit.
        info = {'label' : next(self.label_loader)}

        return self.obs, 0, False, info

    def reset(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``reset()`` function.
        """
        # Reload data and label generators.
        self.env = iter(self.data)
        self.label_loader = iter(self.labels)

    def render(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``render()`` function.
        """
        pass

    def close(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``close()`` function.
        """
        pass

    def preprocess(self) -> None:
        # language=rst
        """
        Preprocessing step for a state specific to dataset objects.
        """
        self.obs = self.obs.view(-1)
        self.obs *= self.intensity

    def reshape(self) -> torch.Tensor:
        # language=rst
        """
        Get reshaped observation for plotting purposes.

        :return: Reshaped observation to plot in ``plt.imshow()`` call.
        """
        if type(self.dataset) == MNIST:
            return self.obs.view(28, 28)
        elif type(self.dataset) in [CIFAR10, CIFAR100]:
            temp = self.obs.view(32, 32, 3).cpu().numpy() / self.intensity
            return temp / temp.max()
        elif type(self.dataset) in SpokenMNIST:
            return self.obs.view(-1, 40)


class MNISTEnvironment(DatasetEnvironment):
    """
    Environment for reward-based MNIST classification.
    """

    def __init__(self, duration_mean: float = 100.0, duration_uc: float = 10.0,
            delay_mean: float = 100.0, delay_uc: float = 10.0):
        """
        Initializes the environment with hyperparameters.

        :param duration_mean: Mean of the duration of each sample.
        :param duration_uc: Uncertainty of the duration. Uniform distribution is
            used.
        :param delay_mean: Mean of the delay of the reward.
        :param duration_uc: Uncertainty of the delay of the reward. Uniform
            distribution is used.
        """
        self.duration_mean = duration_mean
        self.duration_uc = duration_uc
        self.delay_mean = delay_mean
        self.delay_uc = delay_uc

        # Each step should run a single timestep in order to achieve online
        # interaction between the agent and the environment. This could be get
        # better by exploiting the timing structure.
        # TODO Increase simulation running timestep by exploiting the duration/
        # delay timing structure. Only if it improves the performance
        # significantly.
        self.time = 1

        self.dataset = MNIST(path='../../data/MNIST')
        self.obs = None

        self.data, self.labels = self.dataset.get_train()
        self.label_loader = iter(self.labels)
        self.env = iter(self.data)

        self.delayed_reward_queue = torch.zeros(self.delay_mean+self.delay_uc)
        self.current_time_step = 0

    def reward_function(self, action: torch.Tensor = None,
                        label: torch.Tensor = None) -> torch.Tensor:
        pass

    def step(self, a: torch.Tensor = None) -> Tuple[torch.Tensor, int, bool, Dict[str, int]]:
        # language=rst
        """
        Take action of the network(agent) and provide observation and delayed reward.

        :param a: There is no interaction of the network the dataset.
        :return: Observation, reward (fixed to 0), done (fixed to False), and information dictionary.
        """
        try:
            # Attempt to fetch the next observation.
            self.obs = next(self.env)
        except StopIteration:
            # If out of samples, reload data and label generators.
            self.env = iter(self.data)
            self.label_loader = iter(self.labels)
            self.obs = next(self.env)

        # Preprocess observation.
        self.preprocess()

        # Calculate reward of the current timestep.
        label = next(self.label_loader)
        self.reward = self.reward_fuction(action, label)

        # Info dictionary contains label of MNIST digit.
        info = {'label' : label}

        return self.obs, self.reward, False, info

    def reset(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``reset()`` function.
        """
        # Reload data and label generators.
        self.env = iter(self.data)
        self.label_loader = iter(self.labels)

        # Reset delayed_reward_queue and current timestep.
        self.delayed_reward_queue = [0] * (self.delay_mean + self.delay_uc)
        self.current_time_step = 0

    def render(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``render()`` function.
        """
        pass

    def close(self) -> None:
        # language=rst
        """
        Dummy function for OpenAI Gym environment's ``close()`` function.
        """
        pass

    def preprocess(self) -> None:
        # language=rst
        """
        Preprocessing step for a state specific to dataset objects.
        """
        self.obs = self.obs.view(-1)
        self.obs *= self.intensity

    def reshape(self) -> torch.Tensor:
        # language=rst
        """
        Get reshaped observation for plotting purposes.

        :return: Reshaped observation to plot in ``plt.imshow()`` call.
        """
        if type(self.dataset) == MNIST:
            return self.obs.view(28, 28)
        elif type(self.dataset) in [CIFAR10, CIFAR100]:
            temp = self.obs.view(32, 32, 3).cpu().numpy() / self.intensity
            return temp / temp.max()
        elif type(self.dataset) in SpokenMNIST:
            return self.obs.view(-1, 40)


class GymEnvironment(Environment):
    # language=rst
    """
    A wrapper around the OpenAI ``gym`` environments.
    """

    def __init__(self, name: str, **kwargs) -> None:
        # language=rst
        """
        Initializes the environment wrapper.

        :param name: The name of an OpenAI :code:`gym` environment.

        Keyword arguments:

        :param max_prob: Maximum spiking probability.
        """
        self.name = name
        self.env = gym.make(name)
        self.action_space = self.env.action_space

        # Keyword arguments.
        self.max_prob = kwargs.get('max_prob', 1)

        self.obs = None
        self.reward = None

        assert 0 < self.max_prob <= 1, 'Maximum spiking probability must be in (0, 1].'

    def step(self, a: int) -> Tuple[torch.Tensor, float, bool, Dict[Any, Any]]:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`step()` function.

        :param a: Action to take in the environment.
        :return: Observation, reward, done flag, and information dictionary.
        """
        # Call gym's environment step function.
        self.obs, self.reward, self.done, info = self.env.step(a)
        print(self.obs)
        self.preprocess()
        # Return converted observations and other information.
        return self.obs, self.reward, self.done, info

    def reset(self) -> torch.Tensor:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`reset()` function.

        :return: Observation from the environment.
        """
        # Call gym's environment reset function.
        self.obs = self.env.reset()
        self.preprocess()

        return self.obs

    def render(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`render()` function.
        """
        self.env.render()

    def close(self) -> None:
        # language=rst
        """
        Wrapper around the OpenAI Gym environment :code:`close()` function.
        """
        self.env.close()

    def cartpole_preprocess(self, obs: np.array) -> np.array:
        # language=rst
        X_RANGE = 2.4
        X_DOT_RANGE = 2 * X_RANGE
        THETA_RANGE = 0.25
        THETA_DOT_RANGE = 15 * THETA_RANGE
        X_LEVEL = 5
        X_DOT_LEVEL = 5
        THETA_LEVEL = 5
        THETA_DOT_LEVEL = 10
        N_FEATURE = X_LEVEL * X_DOT_LEVEL * THETA_LEVEL * THETA_DOT_LEVEL
        X_SIG = X_RANGE * 2 / X_LEVEL
        X_DOT_SIG = X_DOT_RANGE * 2 / X_DOT_LEVEL
        THETA_SIG = THETA_RANGE * 2 / THETA_LEVEL
        THETA_DOT_SIG = THETA_DOT_RANGE * 2 / THETA_DOT_LEVEL

        THETA_COEFF = (np.pi / 2) / THETA_RANGE
        x, x_dot, theta, theta_dot = obs
        cos_reward = np.cos(THETA_COEFF*theta)
        input_rate = np.zeros([N_FEATURE])
        for i in range(X_LEVEL):
            for j in range(X_DOT_LEVEL):
                for k in range(THETA_LEVEL):
                    for l in range(THETA_DOT_LEVEL):
                        x_center = -X_RANGE + i * 2*X_RANGE/(X_LEVEL-1)
                        x_dot_center = -X_DOT_RANGE + j * 2*X_DOT_RANGE/(X_DOT_LEVEL-1)
                        theta_center = -THETA_RANGE + k * 2*THETA_RANGE/(THETA_LEVEL-1)
                        theta_dot_center = -THETA_DOT_RANGE + l * 2*THETA_DOT_RANGE/(THETA_DOT_LEVEL-1)
                        exponent = - (x - x_center)**2 / (2 * X_SIG**2) \
                                   - (x_dot - x_dot_center)**2 / (2 * X_DOT_SIG**2) \
                                   - (theta - theta_center)**2 / (2 * THETA_SIG**2) \
                                   - (theta_dot - theta_dot_center)**2 / (2 * THETA_DOT_SIG**2)
                        input_rate[i*X_DOT_LEVEL*THETA_LEVEL*THETA_DOT_LEVEL
                                  +j*THETA_LEVEL*THETA_DOT_LEVEL
                                  +k*THETA_DOT_LEVEL + l] = np.exp(exponent)
        return input_rate, cos_reward

    def preprocess(self) -> None:
        # language=rst
        """
        Pre-processing step for an observation from a Gym environment.
        """
        if self.name == 'CartPole-v0':
            self.obs, self.reward = self.cartpole_preprocess(self.obs)
        elif self.name == 'SpaceInvaders-v0':
            self.obs = subsample(gray_scale(self.obs), 84, 110)
            self.obs = self.obs[26:104, :]
            self.obs = binary_image(self.obs)
        elif self.name == 'BreakoutDeterministic-v4':
            self.obs = subsample(gray_scale(crop(self.obs, 34, 194, 0, 160)), 80, 80)
            self.obs = binary_image(self.obs)
        else: # Default pre-processing step
            self.obs = subsample(gray_scale(self.obs), 84, 110)
            self.obs = binary_image(self.obs)

        self.obs = torch.from_numpy(self.obs).float()

    def reshape(self) -> torch.Tensor:
        # language=rst
        """
        Reshape observation for plotting purposes.

        :return: Reshaped observation to plot in ``plt.imshow()`` call.
        """
        return self.obs
