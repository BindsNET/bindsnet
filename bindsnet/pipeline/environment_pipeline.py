import time
from typing import Callable, Optional
import itertools

import matplotlib.pyplot as plt
import pandas as pd
import torch

from ..encoding import bernoulli
from ..environment import Environment
from ..network import Network
from ..network.monitors import Monitor
from ..network.nodes import AbstractInput

from .base_pipeline import BasePipeline


class EnvironmentPipeline(BasePipeline):
    # language=rst
    """
    Abstracts the interaction between network, environment (or dataset), input encoding, and environment feedback
    action.
    """

    def __init__(self, network: Network, environment: Environment, encoding: Callable = bernoulli,
                 action_function: Optional[Callable] = None, **kwargs):
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.
        :param environment: Arbitrary environment.
        :param action_function: Function to convert network outputs into environment inputs.

        Keyword arguments:

        :param int render_interval: Interval to render the environment.

        :param str output: String name of the layer from which to take output from.

        :param int reward_window: Moving average window for the reward plot.
        :param int reward_delay: How many iterations to delay delivery of reward.
        """
        super().__init__(network, **kwargs)

        self.env = environment
        self.action_function = action_function

        self.accumulated_reward = 0.0
        self.reward_list = []

        # Setting kwargs.
        self.output = kwargs.get('output', None)
        self.render_interval = kwargs.get('render_interval', None)
        self.reward_window = kwargs.get('reward_window', None)
        self.reward_delay = kwargs.get('reward_delay', None)
        self.num_episodes = kwargs.get('num_episodes', 100)

        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = torch.zeros(self.reward_delay)

        # Set up for multiple layers of input layers.
        self.inpts = [
            name for name, layer in network.layers.items() if isinstance(layer, AbstractInput)
        ]

        self.action = None
        self.obs = None
        self.reward = None
        self.done = None

        self.voltage_record = None
        self.threshold_value = None
        self.reward_plot = None

        self.first = True

    def init_fn(self):
        pass

    def train(self):
        for self.episode in range(self.num_episodes):
            self.reset_()

            for step in itertools.count():
                batch = self.env_step()

                self.step(batch)

                if batch['done']:
                    break
            print("Episode %d - accumulated reward %f" %
                    (self.episode, self.accumulated_reward))

    def env_step(self):
        # Render game.
        if self.render_interval is not None and self.step_count % self.render_interval == 0:
            self.env.render()

        # Choose action based on output neuron spiking.
        if self.action_function is not None:
            self.action = self.action_function(self, output=self.output)

        # Run a step of the environment.
        self.obs, reward, self.done, info = self.env.step(self.action)

        # Set reward in case of delay.
        if self.reward_delay is not None:
            self.rewards = torch.tensor([reward, *self.rewards[1:]]).float()
            self.reward = self.rewards[-1]
        else:
            self.reward = reward

        # Accumulate reward
        self.accumulated_reward += self.reward

        return {'obs': self.obs, 'reward': reward,
                'done': self.done, 'info': info}

    def step_(self, batch) -> None:
        # language=rst
        """
        Run an iteration of the network and log any needed data
        """

        inpts = {k: batch['obs'] for k in self.inpts}
        reward = batch['reward']

        # Run the network on the spike train-encoded inputs.
        self.network.run(inpts=inpts, time=batch['obs'].shape[0], reward=reward)

        if self.done:
            if self.network.reward_fn is not None:
                self.network.reward_fn.update(**kwargs)
            self.reward_list.append(self.accumulated_reward)

    def reset_(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_()
        self.accumulated_reward = 0.0

    def plots(self, batch, *args):
        return
