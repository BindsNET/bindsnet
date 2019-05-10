from typing import Optional
import warnings

import torch
import torchvision

from .spike_encoders import Encoder

def torchvision_dataset_wrapper_creator(ds_type):
    """ Creates wrapper classes for datasets that output (image, label)
    from __getitem__. This is all of the datasets inside of torchvision.
    """

    if type(ds_type) == str:
        ds_type = getattr(torchvision.datasets, ds_type)

    class torchvision_dataset_wrapper(ds_type):
        __doc__ = """BindsNET torchvision dataset wrapper for:

        The core difference is the output of __getitem__ is no longer
        (image, label) rather a dictionary containing the image, label,
        and their encoded versions if encoders were provided.

            \n\n"""\
                + str(ds_type) if ds_type.__doc__ is None else ds_type.__doc__

        def __init__(self,
                     image_encoder: Optional[Encoder],
                     label_encoder: Optional[Encoder],
                     *args, **kwargs):
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

            self.image_encoder = image_encoder
            self.label_encoder = label_encoder

        def __getitem__(self, ind):
            image, label = super().__getitem__(ind)

            output = {"image": image, "label": label}

            if self.image_encoder is not None:
                output["encoded_image"] = self.image_encoder(image)

            if self.label_encoder is not None:
                output["encoded_label"] = self.label_encoder(label)

            return output

        def __len__(self):
            return super().__len__()

    return torchvision_dataset_wrapper
