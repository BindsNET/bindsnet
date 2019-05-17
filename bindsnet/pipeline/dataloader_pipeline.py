from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..network import Network
from .base_pipeline import BasePipeline
from .pipeline_analysis import PipelineAnalyzer, MatplotlibAnalyzer

class DataLoaderPipeline(BasePipeline):
    """
    A generic DataLoader pipeline that leverages the torch.utils.data
    setup.
    """

    def __init__(self, network: Network,
            train_ds: Dataset,
            test_ds: Optional[Dataset]=None,
            pipeline_analyzer: Optional[PipelineAnalyzer]=None,
            **kwargs):
        """
        Initializes the pipeline

        :param network: Arbitrary network object.
        :param train_ds: Arbitrary torch.utils.data.Dataset object.
        :param test_ds: Arbitrary torch.utils.data.Dataset object.

        Keyword arguments:
        """
        super().__init__(network, **kwargs)

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.num_epochs = kwargs.get('num_epochs', 10)

        self.batch_size = kwargs.get('batch_size', 1)
        self.num_workers = kwargs.get('num_workers', 0)
        self.pin_memory = kwargs.get('pin_memory', False)
        self.shuffle = kwargs.get('shuffle', True)

    def train(self):
        for epoch in range(self.num_epochs):
            train_dataloader = DataLoader(self.train_ds,
                    batch_size=self.batch_size,
                    num_workers=self.num_workers,
                    pin_memory=self.pin_memory,
                    shuffle=self.shuffle)

            for step, batch in enumerate(tqdm(train_dataloader,
                                              desc='Epoch %d/%d'%(epoch+1, self.num_epochs),
                                              total=len(self.train_ds)//self.batch_size)):
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
    def __init__(self, network: Network,
                 train_ds: Dataset, **kwargs):
        super().__init__(network, train_ds, **kwargs)

        self.analyzer = MatplotlibAnalyzer()

    def step_(self, batch):
        self.network.reset_()
        inpts = {"X": batch["encoded_image"]}
        self.network.run(inpts,
                time=batch['encoded_image'].shape[1],
                input_time_dim=1)

        # Unsupervised training means that everything is stored inside
        # of the network object
        return None

    def init_fn(self):
        pass

    def plots(self, input_batch, step_out):
        self.analyzer.plot_obs(input_batch["encoded_image"][0,...].sum(0))
        self.analyzer.plot_spikes(self.get_spike_data())
        self.analyzer.plot_voltage(*self.get_voltage_data())

        self.analyzer.finalize_step()

    def test_step(self):
        pass
