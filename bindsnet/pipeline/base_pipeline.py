import torch

from typing import Optional
import time

from ..network import Network
from ..network.monitors import Monitor

class BasePipeline:
    """
    A generic pipeline that handles high level functionality
    """

    def __init__(self, network: Network, **kwargs):
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.

        Keyword arguments:

        :param int save_interval: How often to save the network to disk.
        :param str save_dir: Directory to save network object to.

        :param float plot_length: Relative time length of the plotted record data. Relative to parameter time.
        :param str plot_type: Type of plotting ('color' or 'line').
        :param int plot_interval: Interval to update plots.

        :param int print_interval: Interval to print text output.
        """
        self.network = network

        """
        Network saving handles caching of intermediate results
        """
        self.save_dir = kwargs.get('save_dir', 'network.pt')
        self.save_interval = kwargs.get('save_interval', None)

        """
        Handles plotting of all layer spikes and voltages. This
        constructs monitors at every level.
        """
        self.plot_interval = kwargs.get('plot_interval', None)
        self.plot_length = kwargs.get('plot_length', 10)

        self.print_interval = kwargs.get('print_interval', None)

        self.test_interval = kwargs.get('test_interval', None)

        if self.plot_interval is not None:
            for l in self.network.layers:
                self.network.add_monitor(Monitor(self.network.layers[l], 's', int(self.plot_length)),
                                         name=f'{l}_spikes')
                if 'v' in self.network.layers[l].__dict__:
                    self.network.add_monitor(Monitor(self.network.layers[l], 'v', int(self.plot_length)),
                                             name=f'{l}_voltages')

        self.step_count = 0

        self.init_fn()

        self.clock = time.time()

    def reset_(self) -> None:
        """
        Reset the pipeline.
        """

        self.network.reset_()
        self.step_count = 0

    def step(self, batch) -> None:
        """
        Single step of any pipeline. Requires these moving components
        """

        net_out = self.step_(batch)

        if self.print_interval is not None and self.step_count % self.print_interval == 0:
            print(f'Iteration: {self.step_count} (Time: {time.time() - self.clock:.4f})')
            self.clock = time.time()

        if self.plot_interval is not None and self.step_count % self.plot_interval == 0:
            self.plots(batch, net_out)

        if self.save_interval is not None and self.step_count % self.save_interval == 0:
            self.network.save(self.save_dir)

        if self.test_interval is not None and self.step_count % self.test_interval == 0:
            self.test()

        self.step_count += 1

        return net_out

    def get_spike_data(self) -> None:
        # language=rst
        """
        Get the spike data from all layers in the pipeline's network.
        """
        return {l: self.network.monitors[f'{l}_spikes'].get('s') for l in self.network.layers}

    def get_voltage_data(self) -> None:
        # language=rst
        """
        Get the voltage data and threshold value from all applicable layers in the pipeline's network.
        """
        voltage_record = {}
        threshold_value = {}
        for l in self.network.layers:
            if 'v' in self.network.layers[l].__dict__:
                voltage_record[l] = self.network.monitors[f'{l}_voltages'].get('v')
            if 'thresh' in self.network.layers[l].__dict__:
                threshold_value[l] = self.network.layers[l].thresh

        return voltage_record, threshold_value

    def step_(self, batch):
        raise NotImplementedError('You need to provide a step_ method')

    def train(self):
        raise NotImplementedError('You need to provide a train method')

    def test(self):
        raise NotImplementedError('You need to provide a test method')

    def init_fn(self):
        raise NotImplementedError('You need to provide an init_fn method')

    def plots(self, batch, *args):
        raise NotImplementedError('You need to provide a plots method')
