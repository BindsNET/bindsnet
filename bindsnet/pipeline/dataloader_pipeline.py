from typing import Optional, Dict

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..network import Network
from .base_pipeline import BasePipeline
from ..analysis.pipeline_analysis import PipelineAnalyzer


class DataLoaderPipeline(BasePipeline):
    """
    A generic DataLoader pipeline that leverages the torch.utils.data
    setup. This still needs to be subclasses for specific
    implementations for functions given the dataset that will be used.
    An example can be seen in `TorchVisionDatasetPipeline`.
    """

    def __init__(
        self,
        network: Network,
        train_ds: Dataset,
        test_ds: Optional[Dataset] = None,
        **kwargs
    ):
        """
        Initializes the pipeline

        :param network: Arbitrary network object.
        :param train_ds: Arbitrary torch.utils.data.Dataset object.
        :param test_ds: Arbitrary torch.utils.data.Dataset object.
        will be managed by the BasePipeline class.

        Keyword arguments:
        """
        super().__init__(network, **kwargs)

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.num_epochs = kwargs.get("num_epochs", 10)

        self.batch_size = kwargs.get("batch_size", 1)
        self.num_workers = kwargs.get("num_workers", 0)
        self.pin_memory = kwargs.get("pin_memory", False)
        self.shuffle = kwargs.get("shuffle", True)

    def train(self) -> None:
        """
        Training loop that runs for the set number of epochs and creates
        a new DataLoader at each epoch.
        """

        for epoch in range(self.num_epochs):
            train_dataloader = DataLoader(
                self.train_ds,
                batch_size=self.batch_size,
                num_workers=self.num_workers,
                pin_memory=self.pin_memory,
                shuffle=self.shuffle,
            )

            for step, batch in enumerate(
                tqdm(
                    train_dataloader,
                    desc="Epoch %d/%d" % (epoch + 1, self.num_epochs),
                    total=len(self.train_ds) // self.batch_size,
                )
            ):
                net_out = self.step(batch)

    def test(self):
        raise NotImplementedError("You need to provide a test function")


class TorchVisionDatasetPipeline(DataLoaderPipeline):
    """
    An example implementation of DataLoaderPipeline that runs all of the
    datasets inside of `bindsnet.datasets` that inherit from an instance
    of a `torchvision.datasets`. These are documented in
    `bindsnet/datasets/README.md`. This specific class just runs an
    unsupervised network.
    """

    def __init__(
        self,
        network: Network,
        train_ds: Dataset,
        pipeline_analyzer: Optional[PipelineAnalyzer] = None,
        **kwargs
    ):
        """
        Initialize the pipeline

        :param network: Arbitrary network object
        :param train_ds: A `torchvision.datasets` wrapper dataset from `bindsnet.datasets`

        Keywork arguments

        :param str input_layer: Layer of the network to place input
        """

        super().__init__(network, train_ds, None, **kwargs)

        self.input_layer = kwargs.get("input_layer", "X")
        self.pipeline_analyzer = pipeline_analyzer

    def step_(self, batch: Dict[str, torch.Tensor]) -> None:
        """
        Perform a pass of the network given the input batch

        :param batch: A dictionary of the current batch. Includes image,
                      label and encoded versions.
        """

        self.network.reset_()
        inpts = {self.input_layer: batch["encoded_image"]}
        self.network.run(inpts, time=batch["encoded_image"].shape[1], input_time_dim=1)

        # Unsupervised training means that everything is stored inside
        # of the network object
        return None

    def init_fn(self) -> None:
        pass

    def plots(
        self, input_batch: Dict[str, torch.Tensor], step_out: None = None
    ) -> None:
        """
        Create any plots and logs for a step given the input batch and
        step output.

        :param input_batch: The batch that was just passed into the network
        :param step_out: The output from the step_ function
        """

        if self.pipeline_analyzer is not None:
            self.pipeline_analyzer.plot_obs(input_batch["encoded_image"][0,
                ...].sum(0), step=self.step_count)

            self.pipeline_analyzer.plot_spikes(self.get_spike_data(), step=self.step_count)

            vr, tv = self.get_voltage_data()
            self.pipeline_analyzer.plot_voltage(vr, tv, step=self.step_count)

            self.pipeline_analyzer.finalize_step()

    def test_step(self):
        pass
