import torch
from torch._six import container_abcs, string_classes

from typing import Optional, Tuple, Dict, Any
import time

from ..network import Network
from ..network.monitors import Monitor


def recursive_to(item, device):
    """
    Recursively transfers everything contained in item to the target
    device.

    :param item: An individual tensor or container of tensors
    :param device: torch.device pointing to cuda or cpu

    :return: A version of item that has been sent to a device
    """

    if isinstance(item, torch.Tensor):
        return item.to(device)
    elif isinstance(item, string_classes):
        return item
    elif isinstance(item, container_abcs.Mapping):
        return {key: recursive_to(item[key], device) for key in item}
    elif isinstance(item, tuple) and hasattr(item, "_fields"):
        return type(item)(*(recursive_to(i, device) for i in item))
    elif isinstance(item, container_abcs.Sequence):
        return [recursive_to(i, device) for i in item]
    else:
        raise NotImplementedError("Target type not supported [%s]" % str(type(item)))


class BasePipeline:
    """
    A generic pipeline that handles high level functionality
    """

    def __init__(self, network: Network, **kwargs):
        """
        Initializes the pipeline.

        :param network: Arbitrary network object.
        will be managed by the BasePipeline class.

        Keyword arguments:

        :param int save_interval: How often to save the network to disk.
        :param str save_dir: Directory to save network object to.

        :param float plot_length: Relative time length of the plotted record data. Relative to parameter time.
        :param str plot_type: Type of plotting ('color' or 'line').
        :param int plot_interval: Interval to update plots.

        :param int print_interval: Interval to print text output.
        "param bool allow_gpu: Allows automatic transfer to the GPU
        """
        self.network = network

        """
        Network saving handles caching of intermediate results
        """
        self.save_dir = kwargs.get("save_dir", "network.pt")
        self.save_interval = kwargs.get("save_interval", None)

        """
        Handles plotting of all layer spikes and voltages. This
        constructs monitors at every level.
        """
        self.plot_interval = kwargs.get("plot_interval", None)
        self.plot_length = kwargs.get("plot_length", 10)

        if self.plot_interval is not None:
            for l in self.network.layers:
                self.network.add_monitor(
                    Monitor(self.network.layers[l], "s", int(self.plot_length)),
                    name=f"{l}_spikes",
                )
                if hasattr(self.network.layers[l], "v"):
                    self.network.add_monitor(
                        Monitor(self.network.layers[l], "v", int(self.plot_length)),
                        name=f"{l}_voltages",
                    )

        self.print_interval = kwargs.get("print_interval", None)

        self.test_interval = kwargs.get("test_interval", None)

        self.step_count = 0

        self.init_fn()

        self.clock = time.time()

        self.allow_gpu = kwargs.get("allow_gpu", True)

        if torch.cuda.is_available() and self.allow_gpu:
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.network.to(self.device)

    def reset_(self) -> None:
        """
        Reset the pipeline.
        """

        self.network.reset_()
        self.step_count = 0

    def step(self, batch) -> Any:
        """
        Single step of any pipeline at a high level.

        :param batch: A batch of inputs to be handed to the step_
                      function. This is an agreed upon standard in a
                      subclass of the BasePipeline.

        :return: The output from the subclass' step_ method which could
                 be anything. Passed to plotting to accomadate this.
        """

        batch = recursive_to(batch, self.device)

        net_out = self.step_(batch)

        if (
            self.print_interval is not None
            and self.step_count % self.print_interval == 0
        ):
            print(
                f"Iteration: {self.step_count} (Time: {time.time() - self.clock:.4f})"
            )
            self.clock = time.time()

        if self.plot_interval is not None and self.step_count % self.plot_interval == 0:
            self.plots(batch, net_out)

        if self.save_interval is not None and self.step_count % self.save_interval == 0:
            self.network.save(self.save_dir)

        if self.test_interval is not None and self.step_count % self.test_interval == 0:
            self.test()

        self.step_count += 1

        return net_out

    def get_spike_data(self) -> Dict[str, torch.Tensor]:
        # language=rst
        """
        Get the spike data from all layers in the pipeline's network.

        :return: A dictionary containing all spike monitors from the network.
        """
        return {
            l: self.network.monitors[f"{l}_spikes"].get("s")
            for l in self.network.layers
        }

    def get_voltage_data(
        self
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        # language=rst
        """
        Get the voltage data and threshold value from all applicable layers in the pipeline's network.

        :return: Two dictionaries containing the voltage data and
                 threshold values from the network.
        """
        voltage_record = {}
        threshold_value = {}
        for l in self.network.layers:
            if hasattr(self.network.layers[l], "v"):
                voltage_record[l] = self.network.monitors[f"{l}_voltages"].get("v")
            if hasattr(self.network.layers[l], "thresh"):
                threshold_value[l] = self.network.layers[l].thresh

        return voltage_record, threshold_value

    def step_(self, batch: Any) -> Any:
        """
        Perform a pass of the network given the input batch

        :param batch: The current batch. This could be anything as long
        as the subclass agrees upon the format in some way.

        :return: Any output that is need for recording purposes.
        """
        raise NotImplementedError("You need to provide a step_ method")

    def train(self) -> None:
        """
        A fully self contained training loop.
        """
        raise NotImplementedError("You need to provide a train method")

    def test(self) -> None:
        """
        A fully self contained test function.
        """
        raise NotImplementedError("You need to provide a test method")

    def init_fn(self) -> None:
        """
        Place holder function for subclass specific actions that need to
        happen during the constructor of the BasePipeline.
        """
        raise NotImplementedError("You need to provide an init_fn method")

    def plots(self, batch, step_output) -> None:
        """
        Create any plots and logs for a step given the input batch and
        step output.

        :param input_batch: The batch that was just passed into the network
        :param step_out: The output from the step_ function
        """
        raise NotImplementedError("You need to provide a plots method")
