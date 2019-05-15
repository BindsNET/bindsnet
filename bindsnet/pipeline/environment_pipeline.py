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


class RLPipeline(BasePipeline):
    # language=rst
    """
    Abstracts the interaction between network, environment (or dataset), input encoding, and environment feedback
    action.
    """

    def __init__(self, network: Network, environment: Environment, encoding: Callable = bernoulli,
                 action_function: Optional[Callable] = None, enable_history: Optional[bool] = False,
                 **kwargs):
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.
        :param environment: Arbitrary environment.
        :param encoding: Function to encode observations into spike trains.
        :param action_function: Function to convert network outputs into environment inputs.
        :param enable_history: Enable history functionality.

        Keyword arguments:

        :param int time: Time input is presented for to the network.
        :param int history: Number of observations to keep track of.
        :param int delta: Step size to save observations in history.
        :param int render_interval: Interval to render the environment.

        :param str output: String name of the layer from which to take output from.

        :param int reward_window: Moving average window for the reward plot.
        :param int reward_delay: How many iterations to delay delivery of reward.
        """
        super().__init__(network, **kwargs)

        self.env = environment
        self.encoding = encoding
        self.action_function = action_function
        self.enable_history = enable_history

        self.history_index = 1
        self.accumulated_reward = 0.0
        self.reward_list = []

        # Setting kwargs.
        self.time = kwargs.get('time', 1)
        self.delta = kwargs.get('delta', 1)
        self.output = kwargs.get('output', None)
        self.history_length = kwargs.get('history_length', None)
        self.render_interval = kwargs.get('render_interval', None)
        self.reward_window = kwargs.get('reward_window', None)
        self.reward_delay = kwargs.get('reward_delay', None)
        self.num_episodes = kwargs.get('num_episodes', 100)

        self.dt = network.dt
        self.timestep = int(self.time / self.dt)

        if self.history_length is not None and self.delta is not None:
            self.history = {i: torch.Tensor() for i in range(1, self.history_length * self.delta + 1, self.delta)}
        else:
            self.history = {}

        if self.reward_delay is not None:
            assert self.reward_delay > 0
            self.rewards = torch.zeros(self.reward_delay)

        # Set up for multiple layers of input layers.
        self.encoded = {
            name: torch.Tensor() for name, layer in network.layers.items() if isinstance(layer, AbstractInput)
        }

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

        # Store frame of history and encode the inputs.
        if self.enable_history and len(self.history) > 0:
            self.update_history()
            self.update_index()

        # Encode the observation using given encoding function.
        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(self.obs, time=self.time, dt=self.network.dt)

        return {'obs': self.obs, 'reward': reward,
                'done': self.done, 'info': info,
                'encoded_obs': self.encoded}

    def step_(self, batch) -> None:
        # language=rst
        """
        Run an iteration of the network and log any needed data
        """

        encoded = batch['encoded_obs']
        reward = batch['reward']

        # Run the network on the spike train-encoded inputs.
        self.network.run(inpts=encoded, time=self.time, reward=reward)

        if self.done:
            if self.network.reward_fn is not None:
                self.network.reward_fn.update(**kwargs)
            self.step_count = 0
            self.reward_list.append(self.accumulated_reward)
            self.accumulated_reward = 0.0

    def update_history(self) -> None:
        # language=rst
        """
        Updates the observations inside history by performing subtraction from  most recent observation and the sum of
        previous observations. If there are not enough observations to take a difference from, simply store the
        observation without any differencing.
        """
        # Recording initial observations
        if self.step_count < len(self.history) * self.delta:
            # Store observation based on delta value
            if self.step_count % self.delta == 0:
                self.history[self.history_index] = self.obs
        else:
            # Take difference between stored frames and current frame
            temp = torch.clamp(self.obs - sum(self.history.values()), 0, 1)

            # Store observation based on delta value.
            if self.step_count % self.delta == 0:
                self.history[self.history_index] = self.obs

            assert (len(self.history) == self.history_length), 'History size is out of bounds'
            self.obs = temp

    def update_index(self) -> None:
        # language=rst
        """
        Updates the index to keep track of history. For example: history = 4, delta = 3 will produce self.history = {1,
        4, 7, 10} and self.history_index will be updated according to self.delta and will wrap around the history
        dictionary.
        """
        if self.step_count % self.delta == 0:
            if self.history_index != max(self.history.keys()):
                self.history_index += self.delta
            else:
                # Wrap around the history.
                self.history_index = (self.history_index % max(self.history.keys())) + 1

    def reset_(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network.reset_()
        self.step_count = 0
        self.accumulated_reward = 0.0
        self.history = {i: torch.Tensor() for i in self.history}

    def plots(self, batch, *args):
        return
