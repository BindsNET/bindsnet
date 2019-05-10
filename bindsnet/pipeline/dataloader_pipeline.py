from typing import Optional

import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from ..network import Network
from .base_pipeline import BasePipeline

class DataLoaderPipeline(BasePipeline):
    """
    A generic DataLoader pipeline that leverages the torch.utils.data
    setup.
    """

    def __init__(self, network: Network,
            train_ds: Dataset,
            test_ds: Optional[Dataset]=None,
            **kwargs):
        super().__init__(network, **kwargs)

        self.init_fn()

        self.train_ds = train_ds
        self.test_ds = test_ds
        self.num_epochs = kwargs.get('num_epochs', 10)

        self.test_interval = kwargs.get('test_interval', None)

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
                self._step(batch)

class TorchVisionDatasetPipeline(DataLoaderPipeline):
    """
    An example implementation of DataLoaderPipeline that runs all of the
    datasets inside of `bindsnet.datasets` that inherit from an instance
    of a `torchvision.datasets`. These are documented in
    `bindsnet/datasets/README.md`
    """
    def __init__(self, network: Network,
                 train_ds: Dataset, **kwargs):
        super().__init__(network, train_ds, **kwargs)

        assert network is not None
        assert train_ds is not None

    def init_fn(self):
        pass

    def _step(self, batch):
        self.network.reset_()
        inpts = {"X": batch["encoded_image"]}
        self.network.run(inpts, time=batch['encoded_image'].shape[0],
                input_time_dim=1)

    def plots(self, input_batch):
        return

    def test(self):
        pass
