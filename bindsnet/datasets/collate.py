# language=rst
"""
This code is directly pulled from the pytorch version found at:

https://github.com/pytorch/pytorch/blob/master/torch/utils/data/_utils/collate.py

Modifications exist to have [time, batch, n_0, ... n_k] instead of batch in dimension 0.
"""

import collections

import torch
from torch._six import string_classes
from torch.utils.data._utils import collate as pytorch_collate


def safe_worker_check():
    # language=rst
    """
    Method to check to use shared memory.
    """
    try:
        return torch.utils.data.get_worker_info() is not None
    except:
        return pytorch_collate._use_shared_memory


def time_aware_collate(batch):
    # language=rst
    """
    Puts each data field into a tensor with dimensions ``[time, batch size, ...]``

    Interpretation of dimensions being input:
    -  0 dim (,) - (1, batch_size, 1)
    -  1 dim (time,) - (time, batch_size, 1)
    - >2 dim (time, n_0, ...) - (time, batch_size, n_0, ...)
    """
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        # catch 0 and 1 dimension cases and view as specified
        if elem.dim() == 0:
            batch = [x.view((1, 1)) for x in batch]
        elif elem.dim() == 1:
            batch = [x.view((x.shape[0], 1)) for x in batch]

        out = None
        if safe_worker_check():
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 1, out=out)
    elif (
        elem_type.__module__ == "numpy"
        and elem_type.__name__ != "str_"
        and elem_type.__name__ != "string_"
    ):
        elem = batch[0]
        if elem_type.__name__ == "ndarray":
            # array of string classes and object
            if (
                pytorch_collate.np_str_obj_array_pattern.search(elem.dtype.str)
                is not None
            ):
                raise TypeError(
                    pytorch_collate.default_collate_err_msg_format.format(elem.dtype)
                )

            return time_aware_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.Mapping):
        return {key: time_aware_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(time_aware_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.Sequence):
        transposed = zip(*batch)
        return [time_aware_collate(samples) for samples in transposed]

    raise TypeError(pytorch_collate.default_collate_err_msg_format.format(elem_type))
