import math
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn.modules.utils import _pair


def im2col_indices(
    x: Tensor,
    kernel_height: int,
    kernel_width: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
) -> Tensor:
    # language=rst
    """
    im2col is a special case of unfold which is implemented inside of Pytorch.

    :param x: Input image tensor to be reshaped to column-wise format.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Input tensor reshaped to column-wise format.
    """
    return F.unfold(x, (kernel_height, kernel_width), padding=padding, stride=stride)


def col2im_indices(
    cols: Tensor,
    x_shape: Tuple[int, int, int, int],
    kernel_height: int,
    kernel_width: int,
    padding: Tuple[int, int] = (0, 0),
    stride: Tuple[int, int] = (1, 1),
) -> Tensor:
    # language=rst
    """
    col2im is a special case of fold which is implemented inside of Pytorch.

    :param cols: Image tensor in column-wise format.
    :param x_shape: Shape of original image tensor.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Image tensor in original image shape.
    """
    return F.fold(
        cols, x_shape, (kernel_height, kernel_width), padding=padding, stride=stride
    )


def get_square_weights(
    weights: Tensor, n_sqrt: int, side: Union[int, Tuple[int, int]]
) -> Tensor:
    # language=rst
    """
    Return a grid of a number of filters ``sqrt ** 2`` with side lengths ``side``.

    :param weights: Two-dimensional tensor of weights for two-dimensional data.
    :param n_sqrt: Square root of no. of filters.
    :param side: Side length(s) of filter.
    :return: Reshaped weights to square matrix of filters.
    """
    if isinstance(side, int):
        side = (side, side)

    square_weights = torch.zeros(side[0] * n_sqrt, side[1] * n_sqrt)
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < weights.size(1):
                break

            x = i * side[0]
            y = (j % n_sqrt) * side[1]
            filter_ = weights[:, n].contiguous().view(*side)
            square_weights[x : x + side[0], y : y + side[1]] = filter_

    return square_weights


def get_square_assignments(assignments: Tensor, n_sqrt: int) -> Tensor:
    # language=rst
    """
    Return a grid of assignments.

    :param assignments: Vector of integers corresponding to class labels.
    :param n_sqrt: Square root of no. of assignments.
    :return: Reshaped square matrix of assignments.
    """
    square_assignments = torch.mul(torch.ones(n_sqrt, n_sqrt), -1.0)
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < assignments.size(0):
                break

            square_assignments[
                i : (i + 1), (j % n_sqrt) : ((j % n_sqrt) + 1)
            ] = assignments[n]

    return square_assignments


def reshape_locally_connected_weights(
    w: Tensor,
    n_filters: int,
    kernel_size: Union[int, Tuple[int, int]],
    conv_size: Union[int, Tuple[int, int]],
    locations: Tensor,
    input_sqrt: Union[int, Tuple[int, int]],
) -> Tensor:
    # language=rst
    """
    Get the weights from a locally connected layer and reshape them to be two-dimensional and square.

    :param w: Weights from a locally connected layer.
    :param n_filters: No. of neuron filters.
    :param kernel_size: Side length(s) of convolutional kernel.
    :param conv_size: Side length(s) of convolution population.
    :param locations: Binary mask indicating receptive fields of convolution population neurons.
    :param input_sqrt: Sides length(s) of input neurons.
    :return: Locally connected weights reshaped as a collection of spatially ordered square grids.
    """
    kernel_size = _pair(kernel_size)
    conv_size = _pair(conv_size)
    input_sqrt = _pair(input_sqrt)

    k1, k2 = kernel_size
    c1, c2 = conv_size
    i1, i2 = input_sqrt
    c1sqrt, c2sqrt = int(math.ceil(math.sqrt(c1))), int(math.ceil(math.sqrt(c2)))
    fs = int(math.ceil(math.sqrt(n_filters)))

    w_ = torch.zeros((n_filters * k1, k2 * c1 * c2))

    for n1 in range(c1):
        for n2 in range(c2):
            for feature in range(n_filters):
                n = n1 * c2 + n2
                filter_ = w[
                    locations[:, n],
                    feature * (c1 * c2) + (n // c2sqrt) * c2sqrt + (n % c2sqrt),
                ].view(k1, k2)
                w_[feature * k1 : (feature + 1) * k1, n * k2 : (n + 1) * k2] = filter_

    if c1 == 1 and c2 == 1:
        square = torch.zeros((i1 * fs, i2 * fs))

        for n in range(n_filters):
            square[
                (n // fs) * i1 : ((n // fs) + 1) * i2,
                (n % fs) * i2 : ((n % fs) + 1) * i2,
            ] = w_[n * i1 : (n + 1) * i2]

        return square
    else:
        square = torch.zeros((k1 * fs * c1, k2 * fs * c2))

        for n1 in range(c1):
            for n2 in range(c2):
                for f1 in range(fs):
                    for f2 in range(fs):
                        if f1 * fs + f2 < n_filters:
                            square[
                                k1 * (n1 * fs + f1) : k1 * (n1 * fs + f1 + 1),
                                k2 * (n2 * fs + f2) : k2 * (n2 * fs + f2 + 1),
                            ] = w_[
                                (f1 * fs + f2) * k1 : (f1 * fs + f2 + 1) * k1,
                                (n1 * c2 + n2) * k2 : (n1 * c2 + n2 + 1) * k2,
                            ]

        return square


def reshape_conv2d_weights(weights: torch.Tensor) -> torch.Tensor:
    # language=rst
    """
    Flattens a connection weight matrix of a Conv2dConnection

    :param weights: Weight matrix of Conv2dConnection object.
    :param wmin: Minimum allowed weight value.
    :param wmax: Maximum allowed weight value.
    """
    sqrt1 = int(np.ceil(np.sqrt(weights.size(0))))
    sqrt2 = int(np.ceil(np.sqrt(weights.size(1))))
    height, width = weights.size(2), weights.size(3)
    reshaped = torch.zeros(
        sqrt1 * sqrt2 * weights.size(2), sqrt1 * sqrt2 * weights.size(3)
    )

    for i in range(sqrt1):
        for j in range(sqrt1):
            for k in range(sqrt2):
                for l in range(sqrt2):
                    if i * sqrt1 + j < weights.size(0) and k * sqrt2 + l < weights.size(
                        1
                    ):
                        fltr = weights[i * sqrt1 + j, k * sqrt2 + l].view(height, width)
                        reshaped[
                            i * height
                            + k * height * sqrt1 : (i + 1) * height
                            + k * height * sqrt1,
                            (j % sqrt1) * width
                            + (l % sqrt2) * width * sqrt1 : ((j % sqrt1) + 1) * width
                            + (l % sqrt2) * width * sqrt1,
                        ] = fltr

    return reshaped
