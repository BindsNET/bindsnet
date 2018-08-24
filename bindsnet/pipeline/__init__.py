import time
import torch
import matplotlib.pyplot as plt

from typing import Callable, Optional

from ..network import Network
from ..encoding import bernoulli
from ..network.nodes import Input
from ..environment import Environment
from ..network.monitors import Monitor
from ..analysis.plotting import plot_spikes, plot_voltages

__all__ = [
    'Pipeline', 'action'
]

plt.ion()


class Pipeline:
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
        :param encoding: Function to encode observations into spike trains.
        :param action_function: Function to convert network outputs into environment inputs.

        Keyword arguments:
            
        :param int plot_interval: Interval to update plots.
        :param str save_dir: Directory to save network object to.
        :param int print_interval: Interval to print text output.
        :param int time: Time input is presented for to the network.
        :param int history: Number of observations to keep track of.
        :param int delta: Step size to save observations in history.
        :param bool render_interval: Interval to render the environment.
        :param int save_interval: How often to save the network to disk.
        :param str output: String name of the layer from which to take output from.
        """
        self.network = network
        self.env = environment
        self.encoding = encoding
        self.action_function = action_function

        self.iteration = 0
        self.history_index = 1
        self.s_ims, self.s_axes = None, None
        self.v_ims, self.v_axes = None, None
        self.obs_im, self.obs_ax = None, None

        # Setting kwargs.
        self.time = kwargs.get('time', 1)
        self.delta = kwargs.get('delta', 1)
        self.output = kwargs.get('output', None)
        self.save_dir = kwargs.get('save_dir', 'network.p')
        self.plot_interval = kwargs.get('plot_interval', None)
        self.save_interval = kwargs.get('save_interval', None)
        self.print_interval = kwargs.get('print_interval', None)
        self.history_length = kwargs.get('history_length', None)
        self.render_interval = kwargs.get('render_interval', None)

        if self.history_length is not None and self.delta is not None:
            self.history = {i: torch.Tensor() for i in range(1, self.history_length * self.delta + 1, self.delta)}
        else:
            self.history = {}

        if self.plot_interval is not None:
            for l in self.network.layers:
                self.network.add_monitor(Monitor(self.network.layers[l], 's', self.plot_interval * self.time),
                                         name=f'{l}_spikes')

                if 'v' in self.network.layers[l].__dict__:
                    self.network.add_monitor(Monitor(self.network.layers[l], 'v', self.plot_interval * self.time),
                                             name=f'{l}_voltages')

            self.spike_record = {l: torch.ByteTensor() for l in self.network.layers}
            self.set_spike_data()
            self.plot_data()

        # Set up for multiple layers of input layers.
        self.encoded = {key: torch.Tensor() for key, val in network.layers.items() if type(val) == Input}

        self.obs = None
        self.reward = None
        self.done = None

        self.voltage_record = None

        self.first = True
        self.clock = time.time()

    def set_spike_data(self) -> None:
        # language=rst
        """
        Get the spike data from all layers in the pipeline's network.
        """
        self.spike_record = {l: self.network.monitors[f'{l}_spikes'].get('s') for l in self.network.layers}

    def set_voltage_data(self) -> None:
        # language=rst
        """
        Get the voltage data from all applicable layers in the pipeline's network.
        """
        self.voltage_record = {}
        for l in self.network.layers:
            if 'v' in self.network.layers[l].__dict__:
                self.voltage_record[l] = self.network.monitors[f'{l}_voltages'].get('v')

    def step(self, **kwargs) -> None:
        # language=rst
        """
        Run an iteration of the pipeline.

        Keyword arguments:

        :param Dict[str, torch.Tensor] clamp: Mapping of layer names to T/F if neuron at time t should be "clamp" to a
                                              certain value specify in clamp_v. The ``Tensor``s of shape ``[time, n_input]``
        :param Dict[str, torch.Tensor] clamp_v: Mapping of layer names to certain value to clamps the True node specify
                                                by clamp. ``Tensor``s of shape ``[time, n_input]``
        """
        clamp = kwargs.get('clamp', {})
        clamp_v = kwargs.get('clamp_v', {})

        if self.print_interval is not None and self.iteration % self.print_interval == 0:
            print(f'Iteration: {self.iteration} (Time: {time.time() - self.clock:.4f})')
            self.clock = time.time()

        if self.save_interval is not None and self.iteration % self.save_interval == 0:
            print(f'Saving network to {self.save_dir}')
            self.network.save(self.save_dir)

        # Render game.
        if self.render_interval is not None and self.iteration % self.render_interval == 0:
            self.env.render()

        # Choose action based on output neuron spiking.
        if self.action_function is not None:
            a = self.action_function(self, output=self.output)
        else:
            a = None

        # Run a step of the environment.
        self.obs, self.reward, _, info = self.env.step(a)

        # Store frame of history and encode the inputs.
        if len(self.history) > 0:
            self.update_history()
            self.update_index()

        # Encode the observation using given encoding function.
        for inpt in self.encoded:
            self.encoded[inpt] = self.encoding(self.obs, time=self.time, max_prob=self.env.max_prob)

        # Run the network on the spike train-encoded inputs.
        self.network.run(inpts=self.encoded, time=self.time, reward=self.reward, clamp=clamp, clamp_v=clamp_v)

        # Plot relevant data.
        if self.plot_interval is not None and self.iteration % self.plot_interval == 0:
            self.plot_data()

            if self.iteration > len(self.history) * self.delta:
                self.plot_obs()

        self.iteration += 1

    def plot_obs(self) -> None:
        # language=rst
        """
        Plot the processed observation after difference against history
        """
        if self.obs_im is None and self.obs_ax is None:
            fig, self.obs_ax = plt.subplots()
            self.obs_ax.set_title('Observation')
            self.obs_ax.set_xticks(())
            self.obs_ax.set_yticks(())
            self.obs_im = self.obs_ax.imshow(self.env.reshape(), cmap='gray')
        else:
            self.obs_im.set_data(self.env.reshape())

    def plot_data(self) -> None:
        # language=rst
        """
        Plot desired variables.
        """
        # Set latest data
        self.set_spike_data()
        self.set_voltage_data()

        # Initialize plots
        if self.s_ims is None and self.s_axes is None and self.v_ims is None and self.v_axes is None:
            self.s_ims, self.s_axes = plot_spikes(self.spike_record)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record)
        else:
            # Update the plots dynamically
            self.s_ims, self.s_axes = plot_spikes(self.spike_record, ims=self.s_ims, axes=self.s_axes)
            self.v_ims, self.v_axes = plot_voltages(self.voltage_record, ims=self.v_ims, axes=self.v_axes)

        plt.pause(1e-8)
        plt.show()

    def update_history(self) -> None:
        # language=rst
        """
        Updates the observations inside history by performing subtraction from  most recent observation and the sum of
        previous observations. If there are not enough observations to take a difference from, simply store the
        observation without any differencing.
        """
        # Recording initial observations
        if self.iteration < len(self.history) * self.delta:
            # Store observation based on delta value
            if self.iteration % self.delta == 0:
                self.history[self.history_index] = self.obs
        else:
            # Take difference between stored frames and current frame
            temp = torch.clamp(self.obs - sum(self.history.values()), 0, 1)

            # Store observation based on delta value.
            if self.iteration % self.delta == 0:
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
        if self.iteration % self.delta == 0:
            if self.history_index != max(self.history.keys()):
                self.history_index += self.delta
            # Wrap around the history
            else:
                self.history_index = (self.history_index % max(self.history.keys())) + 1

    def reset_(self) -> None:
        # language=rst
        """
        Reset the pipeline.
        """
        self.env.reset()
        self.network._reset()
        self.iteration = 0
        self.history = {i: torch.Tensor() for i in self.history}
