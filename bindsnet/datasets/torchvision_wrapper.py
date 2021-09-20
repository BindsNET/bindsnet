from typing import Dict, Optional

import torch
import torchvision

from bindsnet.encoding import Encoder, NullEncoder


def create_torchvision_dataset_wrapper(ds_type):
    # language=rst
    """
    Creates wrapper classes for datasets that output ``(image, label)`` from
    ``__getitem__``. This applies to all of the datasets inside of ``torchvision``.
    """
    if type(ds_type) == str:
        ds_type = getattr(torchvision.datasets, ds_type)

    class TorchvisionDatasetWrapper(ds_type):
        __doc__ = (
            """BindsNET torchvision dataset wrapper for:

        The core difference is the output of __getitem__ is no longer
        (image, label) rather a dictionary containing the image, label,
        and their encoded versions if encoders were provided.

            \n\n"""
            + str(ds_type)
            if ds_type.__doc__ is None
            else ds_type.__doc__
        )

        def __init__(
            self,
            image_encoder: Optional[Encoder] = None,
            label_encoder: Optional[Encoder] = None,
            *args,
            **kwargs,
        ):
            # language=rst
            """
            Constructor for the BindsNET torchvision dataset wrapper.
            For details on the dataset you're interested in visit

            https://pytorch.org/docs/stable/torchvision/datasets.html

            :param image_encoder: Spike encoder for use on the image
            :param label_encoder: Spike encoder for use on the label
            :param *args: Arguments for the original dataset
            :param **kwargs: Keyword arguments for the original dataset
            """
            super().__init__(*args, **kwargs)

            self.args = args
            self.kwargs = kwargs

            # Allow the passthrough of None, but change to NullEncoder
            if image_encoder is None:
                image_encoder = NullEncoder()

            if label_encoder is None:
                label_encoder = NullEncoder()

            self.image_encoder = image_encoder
            self.label_encoder = label_encoder

        def __getitem__(self, ind: int) -> Dict[str, torch.Tensor]:
            # language=rst
            """
            Utilizes the ``torchvision.dataset`` parent class to grab the data, then
            encodes using the supplied encoders.

            :param int ind: Index to grab data at.
            :return: The relevant data and encoded data from the requested index.
            """
            image, label = super().__getitem__(ind)

            output = {
                "image": image,
                "label": label,
                "encoded_image": self.image_encoder(image),
                "encoded_label": self.label_encoder(label),
            }

            return output

        def __len__(self):
            return super().__len__()

    return TorchvisionDatasetWrapper
