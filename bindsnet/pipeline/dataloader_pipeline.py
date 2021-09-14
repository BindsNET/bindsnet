from typing import Dict, Optional

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from bindsnet.analysis.pipeline_analysis import PipelineAnalyzer
from bindsnet.datasets import DataLoader
from bindsnet.network import Network
from bindsnet.pipeline.base_pipeline import BasePipeline


class DataLoaderPipeline(BasePipeline):
    # language=rst
    """
    A generic ``DataLoader`` pipeline that leverages the ``torch.utils.data`` setup.
    This still needs to be subclassed for specific implementations for functions given
    the dataset that will be used. An example can be seen in
    ``TorchVisionDatasetPipeline``.
    """

    def __init__(
        self,
        network: Network,
        train_ds: Dataset,
        test_ds: Optional[Dataset] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary ``network`` object.
        :param train_ds: Arbitrary ``torch.utils.data.Dataset`` object.
        :param test_ds: Arbitrary ``torch.utils.data.Dataset`` object.
        """
        super().__init__(network, **kwargs)

        self.train_ds = train_ds
        self.test_ds = test_ds

        self.num_epochs = kwargs.get("num_epochs", 10)
        self.batch_size = kwargs.get("batch_size", 1)
        self.num_workers = kwargs.get("num_workers", 0)
        self.pin_memory = kwargs.get("pin_memory", True)
        self.shuffle = kwargs.get("shuffle", True)

    def train(self) -> None:
        # language=rst
        """
        Training loop that runs for the set number of epochs and creates a new
        ``DataLoader`` at each epoch.
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
                self.step(batch)

    def test(self) -> None:
        raise NotImplementedError("You need to provide a test function.")


class TorchVisionDatasetPipeline(DataLoaderPipeline):
    # language=rst
    """
    An example implementation of ``DataLoaderPipeline`` that runs all of the datasets
    inside of ``bindsnet.datasets`` that inherit from an instance of a
    ``torchvision.datasets``. These are documented in ``bindsnet/datasets/README.md``.
    This specific class just runs an unsupervised network.
    """

    def __init__(
        self,
        network: Network,
        train_ds: Dataset,
        pipeline_analyzer: Optional[PipelineAnalyzer] = None,
        **kwargs,
    ) -> None:
        # language=rst
        """
        Initializes the pipeline.

        :param network: Arbitrary ``network`` object.
        :param train_ds: A ``torchvision.datasets`` wrapper dataset from
            ``bindsnet.datasets``.

        Keyword arguments:

        :param str input_layer: Layer of the network that receives input.
        """
        super().__init__(network, train_ds, None, **kwargs)

        self.input_layer = kwargs.get("input_layer", "X")
        self.pipeline_analyzer = pipeline_analyzer

    def step_(self, batch: Dict[str, torch.Tensor], **kwargs) -> None:
        # language=rst
        """
        Perform a pass of the network given the input batch. Unsupervised training
        (implying everything is stored inside of the ``network`` object, therefore
        returns ``None``.

        :param batch: A dictionary of the current batch. Includes image, label and
            encoded versions.
        """
        self.network.reset_state_variables()
        inputs = {self.input_layer: batch["encoded_image"]}
        self.network.run(inputs, time=batch["encoded_image"].shape[0])

    def init_fn(self) -> None:
        pass

    def plots(self, batch: Dict[str, torch.Tensor], *args) -> None:
        # language=rst
        """
        Create any plots and logs for a step given the input batch.

        :param batch: A dictionary of the current batch. Includes image, label and
            encoded versions.
        """
        if self.pipeline_analyzer is not None:
            self.pipeline_analyzer.plot_obs(
                batch["encoded_image"][0, ...].sum(0), step=self.step_count
            )

            self.pipeline_analyzer.plot_spikes(
                self.get_spike_data(), step=self.step_count
            )

            vr, tv = self.get_voltage_data()
            self.pipeline_analyzer.plot_voltages(vr, tv, step=self.step_count)

            self.pipeline_analyzer.finalize_step()

    def test_step(self):
        pass
