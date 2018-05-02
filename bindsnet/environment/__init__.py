import os
import sys
import gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from bindsnet.datasets.preprocess import *
from bindsnet.encoding            import *
from skimage.measure              import block_reduce


class Games(ABC):
    '''
    Abstract base class for OpenAI gym environments.
    '''
    def __init__(self):
        '''
        Abstract constructor for the Nodes class.
        '''
        super().__init__()

    @abstractmethod
    def preprocess(self):
        '''
        Pre-processing steps for every observation
        '''
        pass
    
    def get_observation(self):
        '''
        Returns the observation for current timestep
        '''
        return self.obs

    def get_reward(self):
        '''
        Returns the reward for current timestep
        '''
        return self.reward
    
    def close(self):
        '''
        Wrapper around the OpenAI Gym environment :code:`close()` function.
        '''
        self.env.close()

class SpaceInvaders(Games):
    '''
    A wrapper around the :code:`SpaceInvaders-v0` OpenAI gym environment.
    '''
    def __init__(self, max_prob=0.25, diffs=True):
        '''
        Initializes the OpenAI Gym Space Invaders environment wrapper.
        
        Inputs:
            | :code:`max_prob` (:code:`float`): Specifies the maximum Bernoulli spiking probability.
            | :code:`diffs` (:code:`bool`): Whether to record previous frame and take differences.
        '''
        super().__init__()
        
        self.max_prob = max_prob
        self.env = gym.make('SpaceInvaders-v0')
        self.diffs = diffs

    def step(self, a):
        '''
        Wrapper around the OpenAI Gym environment :code:`step()` function.
        
        Inputs:
            | :code:`a` (:code:`int`): Action to take in Space Invaders environment.
        
        Returns:
            | (:code:`torch.Tensor`): Observation from the environment.
            | (:code:`float`): Reward signal from the environment.
            | (:code:`bool`): Indicates whether the simulation has finished.
            | (:code:`dict`): Current information about the environment.
        '''
        # Call gym's environment step function.
        self.obs, self.reward, done, info = self.env.step(a)
        self.preprocess()
        
        # Return converted observations and other information.
        return self.obs, self.reward, done, info

    
    def reset(self):
        '''
        Wrapper around the OpenAI Gym environment :code:`reset()` function.
        
        Returns:
            | (:code:`torch.Tensor`): Observation from the environment.
            obs (torch.Tensor): Observation from the environment.
        '''
        # Call gym's environment reset function.
        self.obs = self.env.reset()
        self.preprocess()
        

    def render(self):
        '''
        Wrapper around the OpenAI Gym environment :code:`render()` function.
        '''
        self.env.render()


    def close(self):
        '''
        Wrapper around the OpenAI Gym environment :code:`close()` function.
        '''
        self.env.close()


    def preprocess(self):
        '''
        Pre-Processing step for a state specific to Space Invaders.
        
        Inputs:
            obs(numpy.array): Observation from the environment.
        
        Returns:
            obs (torch.Tensor): Pre-processed observation.
        '''
        self.obs = subsample( gray_scale(self.obs), 84, 110)
        self.obs = self.obs[26:104, :]
        self.obs = binary_image(self.obs)
        self.obs = np.reshape(self.obs, (78, 84, 1))
        self.obs = torch.from_numpy(self.obs).view(1, -1).float()

        
class CartPole(Games):
    '''
    A wrapper around the :code:`CartPole-v0` OpenAI gym environment.
    '''
    def __init__(self, max_prob=0.5):
        '''
        Initializes the OpenAI Gym Space Invaders environment wrapper.
        
        Inputs:
            | :code:`max_prob` (:code:`float`): Specifies the maximum Bernoulli trial spiking probability.
        '''
        super().__init__()
        
        self.max_prob = max_prob
        self.env = gym.make('CartPole-v0')

    def step(self, a):
        '''
        Wrapper around the OpenAI Gym environment :code:`step()` function.
        
        Inputs:
            | :code:`a` (:code:`int`): Action to take in Space Invaders environment.
        
        Returns:
            | (:code:`torch.Tensor`): Observation from the environment.
            | (:code:`float`): Reward signal from the environment.
            | (:code:`bool`): Indicates whether the simulation has finished.
            | (:code:`dict`): Current information about the environment.
        '''
        # Call gym's environment step function.
        obs, reward, done, info = self.env.step(a)
        
        # Encoding into positive values.
        obs = np.array([obs[0] + 2.4,
                        -min(obs[1], 0),
                        max(obs[1], 0),
                        obs[2] + 41.8,
                        -min(obs[3], 0),
                        max(obs[3], 0)])
        
        # Convert to torch.Tensor, and then to Bernoulli-distributed spikes.
        obs = torch.from_numpy(obs).view(1, -1).float()
        obs = get_bernoulli(obs, max_prob=self.max_prob)
        
        # Return converted observations and other information.
        return next(obs).view(1, -1), reward, done, info

    def reset(self):
        '''
        Wrapper around the OpenAI Gym environment :code:`reset()` function.
        
        Returns:
            | (:code:`torch.Tensor`): Observation from the environment.
        '''
        # Call gym's environment reset function.
        obs = self.env.reset()
        
        # Encoding into positive values.
        obs = np.array([obs[0] + 2.4,
                        -min(obs[1], 0),
                        max(obs[1], 0),
                        obs[2] + 41.8,
                        -min(obs[3], 0),
                        max(obs[3], 0)])
        
        # Convert to torch.Tensor, and
        # convert to Bernoulli-distributed spikes.
        obs = torch.from_numpy(obs).view(1, -1).float()
        obs = get_bernoulli(obs, max_prob=self.max_prob)

        # Return converted observations.
        return next(obs)

    def preprocess(self):
        pass
    
    def render(self):
        '''
        Wrapper around the OpenAI Gym environment :code:`render()` function.
        '''
        self.env.render()

    def close(self):
        '''
        Wrapper around the OpenAI Gym environment :code:`close()` function.
        '''
        self.env.close()
