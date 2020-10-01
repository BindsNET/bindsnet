from abc import ABC, abstractmethod
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from tensorboardX import SummaryWriter
from torchvision.utils import make_grid

from .plotting import plot_spikes, plot_voltages, plot_conv2d_weights
from ..utils import reshape_conv2d_weights


class PipelineAnalyzer(ABC):
    # language=rst
    """
    Responsible for pipeline analysis. Subclasses maintain state
    information related to plotting or logging.
    """

    @abstractmethod
    def finalize_step(self) -> None:
        # language=rst
        """
        Flush the output from the current step.
        """
        pass

    @abstractmethod
    def plot_obs(self, obs: torch.Tensor, tag: str = "obs", step: int = None) -> None:
        # language=rst
        """
        Pulls the observation from PyTorch and sets up for Matplotlib
        plotting.

        :param obs: A 2D array of floats depicting an input image.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        pass

    @abstractmethod
    def plot_reward(
        self,
        reward_list: list,
        reward_window: int = None,
        tag: str = "reward",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plot the accumulated reward for each episode.

        :param reward_list: The list of recent rewards to be plotted.
        :param reward_window: The length of the window to compute a moving average over.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        pass

    @abstractmethod
    def plot_spikes(
        self,
        spike_record: Dict[str, torch.Tensor],
        tag: str = "spike",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plots all spike records inside of ``spike_record``. Keeps unique
        plots for all unique tags that are given.

        :param spike_record: Dictionary of spikes to be rasterized.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        pass

    @abstractmethod
    def plot_voltages(
        self,
        voltage_record: Dict[str, torch.Tensor],
        thresholds: Optional[Dict[str, torch.Tensor]] = None,
        tag: str = "voltage",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plots all voltage records and given thresholds. Keeps unique
        plots for all unique tags that are given.

        :param voltage_record: Dictionary of voltages for neurons inside of networks
                               organized by the layer they correspond to.
        :param thresholds: Optional dictionary of threshold values for neurons.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        pass

    @abstractmethod
    def plot_conv2d_weights(
        self, weights: torch.Tensor, tag: str = "conv2d", step: int = None
    ) -> None:
        # language=rst
        """
        Plot a connection weight matrix of a ``Conv2dConnection``.

        :param weights: Weight matrix of ``Conv2dConnection`` object.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        pass


class MatplotlibAnalyzer(PipelineAnalyzer):
    # language=rst
    """
    Renders output using Matplotlib.

    Matplotlib requires objects to be kept around over the full lifetime
    of the plots; this is done through ``self.plots``. An interactive session
    is needed so that we can continue processing and just update the
    plots.
    """

    def __init__(self, **kwargs) -> None:
        # language=rst
        """
        Initializes the analyzer.

        Keyword arguments:

        :param str volts_type: Type of plotting for voltages (``"color"`` or ``"line"``).
        """
        self.volts_type = kwargs.get("volts_type", "color")
        plt.ion()
        self.plots = {}

    def plot_obs(self, obs: torch.Tensor, tag: str = "obs", step: int = None) -> None:
        # language=rst
        """
        Pulls the observation off of torch and sets up for Matplotlib
        plotting.

        :param obs: A 2D array of floats depicting an input image.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        obs = obs.detach().cpu().numpy()
        obs = np.transpose(obs, (1, 2, 0)).squeeze()

        if tag in self.plots:
            obs_ax, obs_im = self.plots[tag]
        else:
            obs_ax, obs_im = None, None

        if obs_im is None and obs_ax is None:
            fig, obs_ax = plt.subplots()
            obs_ax.set_title("Observation")
            obs_ax.set_xticks(())
            obs_ax.set_yticks(())
            obs_im = obs_ax.imshow(obs, cmap="gray")

            self.plots[tag] = obs_ax, obs_im
        else:
            obs_im.set_data(obs)

    def plot_reward(
        self,
        reward_list: list,
        reward_window: int = None,
        tag: str = "reward",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plot the accumulated reward for each episode.

        :param reward_list: The list of recent rewards to be plotted.
        :param reward_window: The length of the window to compute a moving average over.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        if tag in self.plots:
            reward_im, reward_ax, reward_plot = self.plots[tag]
        else:
            reward_im, reward_ax, reward_plot = None, None, None

        # Compute moving average.
        if reward_window is not None:
            # Ensure window size > 0 and < size of reward list.
            window = max(min(len(reward_list), reward_window), 0)

            # Fastest implementation of moving average.
            reward_list_ = (
                pd.Series(reward_list)
                .rolling(window=window, min_periods=1)
                .mean()
                .values
            )
        else:
            reward_list_ = reward_list[:]

        if reward_im is None and reward_ax is None:
            reward_im, reward_ax = plt.subplots()
            reward_ax.set_title("Accumulated reward")
            reward_ax.set_xlabel("Episode")
            reward_ax.set_ylabel("Reward")
            (reward_plot,) = reward_ax.plot(reward_list_)

            self.plots[tag] = reward_im, reward_ax, reward_plot
        else:
            reward_plot.set_data(range(len(reward_list_)), reward_list_)
            reward_ax.relim()
            reward_ax.autoscale_view()

    def plot_spikes(
        self,
        spike_record: Dict[str, torch.Tensor],
        tag: str = "spike",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plots all spike records inside of ``spike_record``. Keeps unique
        plots for all unique tags that are given.

        :param spike_record: Dictionary of spikes to be rasterized.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        if tag not in self.plots:
            self.plots[tag] = plot_spikes(spike_record)
        else:
            s_im, s_ax = self.plots[tag]
            self.plots[tag] = plot_spikes(spike_record, ims=s_im, axes=s_ax)

    def plot_voltages(
        self,
        voltage_record: Dict[str, torch.Tensor],
        thresholds: Optional[Dict[str, torch.Tensor]] = None,
        tag: str = "voltage",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plots all voltage records and given thresholds. Keeps unique
        plots for all unique tags that are given.

        :param voltage_record: Dictionary of voltages for neurons inside of networks
                               organized by the layer they correspond to.
        :param thresholds: Optional dictionary of threshold values for neurons.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        if tag not in self.plots:
            self.plots[tag] = plot_voltages(
                voltage_record, plot_type=self.volts_type, thresholds=thresholds
            )
        else:
            v_im, v_ax = self.plots[tag]
            self.plots[tag] = plot_voltages(
                voltage_record,
                ims=v_im,
                axes=v_ax,
                plot_type=self.volts_type,
                thresholds=thresholds,
            )

    def plot_conv2d_weights(
        self, weights: torch.Tensor, tag: str = "conv2d", step: int = None
    ) -> None:
        # language=rst
        """
        Plot a connection weight matrix of a ``Conv2dConnection``.

        :param weights: Weight matrix of ``Conv2dConnection`` object.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        wmin = weights.min().item()
        wmax = weights.max().item()

        if tag not in self.plots:
            self.plots[tag] = plot_conv2d_weights(weights, wmin, wmax)
        else:
            im = self.plots[tag]
            plot_conv2d_weights(weights, wmin, wmax, im=im)

    def finalize_step(self) -> None:
        # language=rst
        """
        Flush the output from the current step
        """
        plt.draw()
        plt.pause(1e-8)
        plt.show()


class TensorboardAnalyzer(PipelineAnalyzer):
    def __init__(self, summary_directory: str = "./logs"):
        # language=rst
        """
        Initializes the analyzer.

        :param summary_directory: Directory to save log files.
        """
        self.writer = SummaryWriter(summary_directory)

    def finalize_step(self) -> None:
        # language=rst
        """
        No-op for ``TensorboardAnalyzer``.
        """
        pass

    def plot_obs(self, obs: torch.Tensor, tag: str = "obs", step: int = None) -> None:
        # language=rst
        """
        Pulls the observation off of torch and sets up for Matplotlib
        plotting.

        :param obs: A 2D array of floats depicting an input image.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        obs_grid = make_grid(obs.float(), nrow=4, normalize=True)
        self.writer.add_image(tag, obs_grid, step)

    def plot_reward(
        self,
        reward_list: list,
        reward_window: int = None,
        tag: str = "reward",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plot the accumulated reward for each episode.

        :param reward_list: The list of recent rewards to be plotted.
        :param reward_window: The length of the window to compute a moving average over.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        self.writer.add_scalar(tag, reward_list[-1], step)

    def plot_spikes(
        self,
        spike_record: Dict[str, torch.Tensor],
        tag: str = "spike",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plots all spike records inside of ``spike_record``. Keeps unique
        plots for all unique tags that are given.

        :param spike_record: Dictionary of spikes to be rasterized.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        for k, spikes in spike_record.items():
            # shuffle spikes into 1x1x#NueronsxT
            spikes = spikes.view(1, 1, -1, spikes.shape[-1]).float()
            spike_grid_img = make_grid(spikes, nrow=1, pad_value=0.5)

            self.writer.add_image(tag + "_" + str(k), spike_grid_img, step)

    def plot_voltages(
        self,
        voltage_record: Dict[str, torch.Tensor],
        thresholds: Optional[Dict[str, torch.Tensor]] = None,
        tag: str = "voltage",
        step: int = None,
    ) -> None:
        # language=rst
        """
        Plots all voltage records and given thresholds. Keeps unique
        plots for all unique tags that are given.

        :param voltage_record: Dictionary of voltages for neurons inside of networks
                               organized by the layer they correspond to.
        :param thresholds: Optional dictionary of threshold values for neurons.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        for k, v in voltage_record.items():
            # Shuffle voltages into 1x1x#neuronsxT
            v = v.view(1, 1, -1, v.shape[-1])
            voltage_grid_img = make_grid(v, nrow=1, pad_value=0)

            self.writer.add_image(tag + "_" + str(k), voltage_grid_img, step)

    def plot_conv2d_weights(
        self, weights: torch.Tensor, tag: str = "conv2d", step: int = None
    ) -> None:
        # language=rst
        """
        Plot a connection weight matrix of a ``Conv2dConnection``.

        :param weights: Weight matrix of ``Conv2dConnection`` object.
        :param tag: A unique tag to associate the data with.
        :param step: The step of the pipeline.
        """
        reshaped = reshape_conv2d_weights(weights).unsqueeze(0)

        reshaped -= reshaped.min()
        reshaped /= reshaped.max()

        self.writer.add_image(tag, reshaped, step)
