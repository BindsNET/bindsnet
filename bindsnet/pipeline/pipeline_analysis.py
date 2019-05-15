from abc import ABC, abstractmethod

import matplotlib.pyplot as plt
import pandas as pd
import torch
from ..analysis.plotting import plot_spikes, plot_voltages

class PipelineAnalyzer(ABC):
    """
    Responsible for pipeline analysis. Subclasses maintain state
    information related to plotting.
    """

    @abstractmethod
    def finalize_step(self):
        """
        Flush the output from the current step
        """
        pass

class MatplotlibAnalyzer(PipelineAnalyzer):
    """
    Renders output using matplotlib.

    Matplotlib requires objects to be kept around over the full lifetime
    of the plots--this is done through self.plots. Interactive session
    is needed so that we can continue processing and just update the
    plots.
    """

    def __init__(self):
        plt.ion()
        self.plots = {}

    def plot_obs(self, obs, tag='obs'):
        """
        Plot the processed observation after difference against history
        """
        if obs in self.plots:
            obs_ax, obs_im = self.plots[tag]
        else:
            obs_ax, obs_im = None, None

        if obs_im is None and obs_ax is None:
            fig, obs_ax = plt.subplots()
            obs_ax.set_title('Observation')
            obs_ax.set_xticks(())
            obs_ax.set_yticks(())
            obs_im = obs_ax.imshow(obs, cmap='gray')
            
            self.plots[obs] = obs_ax, obs_im
        else:
            obs_im.set_data(obs)

    def plot_reward(self, reward_list, reward_window: int=None, tag='reward') -> None:
        # language=rst
        """
        Plot the accumulated reward for each episode.
        """
        if obs in self.plots:
            obs_ax, obs_im = self.plots[tag]
        else:
            obs_ax, obs_im = None, None

        # Compute moving average
        if self.reward_window is not None:
            # Ensure window size > 0 and < size of reward list
            window = max(min(len(self.reward_list), self.reward_window), 0)

            # Fastest implementation of moving average
            reward_list_ = pd.Series(self.reward_list).rolling(window=window, min_periods=1).mean().values
        else:
            reward_list_ = self.reward_list[:]

        if reward_im is None and reward_ax is None:
            reward_im, reward_ax = plt.subplots()
            reward_ax.set_title('Accumulated reward')
            reward_ax.set_xlabel('Episode')
            reward_ax.set_ylabel('Reward')
            reward_plot, = self.reward_ax.plot(reward_list_)

            self.plots[tag] = reward_im, reward_ax
        else:
            reward_plot.set_data(range(self.episode), reward_list_)
            reward_ax.relim()
            reward_ax.autoscale_view()

    def plot_spike(self, spike_record, tag='spike'):
        if self.s_ims is None and self.s_axes is None:
            self.s_ims, self.s_axes = plot_spikes(self.spike_record)
        else:
            self.s_ims, self.s_axes = plot_spikes(self.spike_record, ims=self.s_ims, axes=self.s_axes)

    def plot_voltage(self, voltage_record, tag='voltage'):
        if self.v_ims is None and self.v_axes is None:
            self.v_ims, self.v_axes = plot_voltages(
                self.voltage_record, plot_type=self.plot_type, threshold=self.threshold_value
            )
        else:
            self.v_ims, self.v_axes = plot_voltages(
                self.voltage_record, ims=self.v_ims, axes=self.v_axes,
                plot_type=self.plot_type, threshold=self.threshold_value
            )

    def plot_data(self, spike_record, voltage_record, tag='data'):
        # Initialize plots
        self.plot_spikes(spike_record, tag+'_s')
        self.plot_voltage(voltage_record, tag+'_v')

    def finalize_step(self):
        plt.pause(1e-8)
        plt.show()
