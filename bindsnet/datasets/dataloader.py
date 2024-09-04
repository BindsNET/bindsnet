import torch

from bindsnet.datasets.collate import time_aware_collate


class DataLoader(torch.utils.data.DataLoader):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=time_aware_collate,
        pin_memory=False,
        drop_last=False,
        timeout=0,
        worker_init_fn=None,
    ):
        super().__init__(
            dataset,
            sampler=sampler,
            shuffle=shuffle,
            batch_size=batch_size,
            drop_last=drop_last,
            pin_memory=pin_memory,
            timeout=timeout,
            num_workers=num_workers,
            worker_init_fn=worker_init_fn,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
        )
