import math
import torch
import numpy as np

from torch import Tensor
from numpy import ndarray
from typing import Tuple, Union
from torch.nn.modules.utils import _pair


def get_im2col_indices(x_shape: Tuple[int, int, int, int], kernel_height: int,
                       kernel_width: int, padding: Tuple[int, int]=(0, 0),
                       stride: Tuple[int, int]=(1, 1)) -> Tuple[ndarray, ndarray, ndarray]:
    # language=rst
    """
    Figure out what the size of the output should be. Taken from `this repository
    <https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py>`_.

    :param x_shape: Shape of the input tensor.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Indices for converted image tensor to column-wise format.
    """
    _, c, h, w = x_shape

    assert (h + 2 * padding[0] - kernel_height) % stride[0] == 0
    assert (w + 2 * padding[1] - kernel_height) % stride[1] == 0

    out_height = int((h + 2 * padding[0] - kernel_height) / stride[0] + 1)
    out_width = int((w + 2 * padding[1] - kernel_width) / stride[1] + 1)

    i0 = np.repeat(np.arange(kernel_height), kernel_width)
    i0 = np.tile(i0, c)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(kernel_width), kernel_height * c)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(c), kernel_height * kernel_width).reshape(-1, 1)

    return k, i, j


def im2col_indices(x: Tensor, kernel_height: int, kernel_width: int, padding: Tuple[int, int]=(0, 0),
                   stride: Tuple[int, int]=(1, 1)) -> Tensor:
    # language=rst
    """
    An implementation of im2col based on some fancy indexing. Taken from `this repository
    <https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py>`_.

    :param x: Input image tensor to be reshaped to column-wise format.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Input tensor reshaped to column-wise format.
    """
    # Zero-pad the input
    p = padding
    x_padded = np.pad(x, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')

    k, i, j = get_im2col_indices(x.shape, kernel_height, kernel_width, padding, stride)

    cols = x_padded[:, k, i, j]
    c = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(kernel_height * kernel_width * c, -1)

    return Tensor(cols)


def col2im_indices(cols: Tensor, x_shape: Tuple[int, int, int, int], kernel_height: int, kernel_width: int,
                   padding: Tuple[int, int]=(0, 0), stride: Tuple[int, int]=(1, 1)) -> Tensor:
    # language=rst
    """
    An implementation of col2im based on fancy indexing and np.add.at. Taken from `this repository
    <https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py>`_.

    :param cols: Image tensor in column-wise format.
    :param x_shape: Shape of original image tensor.
    :param kernel_height: Height of the convolutional kernel in pixels.
    :param kernel_width: Width of the convolutional kernel in pixels.
    :param padding: Amount of zero padding on the input image.
    :param stride: Amount to stride over image by per convolution.
    :return: Image tensor in original image shape.
    """
    n, c, h, w = x_shape
    h_padded, w_padded = h + 2 * padding[0], w + 2 * padding[1]
    x_padded = np.zeros((n, c, h_padded, w_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, kernel_height, kernel_width, padding, stride)
    cols_reshaped = cols.reshape(c * kernel_height * kernel_width, -1, n)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)

    if padding == (0, 0):
        return Tensor(x_padded)

    return x_padded[:, :, padding[0]:-padding[0], padding[1]:-padding[1]]


def get_square_weights(weights: Tensor, n_sqrt: int, side: Union[int, Tuple[int, int]]) -> Tensor:
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

    square_weights = torch.zeros_like(torch.Tensor(side[0] * n_sqrt, side[1] * n_sqrt))
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < weights.size(1):
                break

            x = i * side[0]
            y = (j % n_sqrt) * side[1]
            filter_ = weights[:, n].contiguous().view(*side)
            square_weights[x: x + side[0], y: y + side[1]] = filter_

    return square_weights


def get_square_assignments(assignments: Tensor, n_sqrt: int) -> Tensor:
    # language=rst
    """
    Return a grid of assignments.

    :param assignments: Vector of integers corresponding to class labels.
    :param n_sqrt: Square root of no. of assignments.
    :return: Reshaped square matrix of assignments.
    """
    square_assignments = -1 * torch.ones_like(torch.Tensor(n_sqrt, n_sqrt))
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            n = i * n_sqrt + j

            if not n < assignments.size(0):
                break

            square_assignments[i: (i + 1), (j % n_sqrt): ((j % n_sqrt) + 1)] = assignments[n]

    return square_assignments


def reshape_locally_connected_weights(w: Tensor, n_filters: int, kernel_size: Union[int, Tuple[int, int]],
                                      conv_size: Union[int, Tuple[int, int]], locations: Tensor,
                                      input_sqrt: Union[int, Tuple[int, int]]) -> Tensor:
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
                filter_ = w[locations[:, n], feature * (c1 * c2) + (n // c2sqrt) * c2sqrt + (n % c2sqrt)].view(k1, k2)
                w_[feature * k1: (feature + 1) * k1, n * k2: (n + 1) * k2] = filter_

    if c1 == 1 and c2 == 1:
        square = torch.zeros((i1 * fs, i2 * fs))

        for n in range(n_filters):
            square[(n // fs) * i1: ((n // fs) + 1) * i2, (n % fs) * i2: ((n % fs) + 1) * i2] = w_[n * i1: (n + 1) * i2]

        return square
    else:
        square = torch.zeros((k1 * fs * c1, k2 * fs * c2))

        for n1 in range(c1):
            for n2 in range(c2):
                for f1 in range(fs):
                    for f2 in range(fs):
                        if f1 * fs + f2 < n_filters:
                            square[k1 * (n1 * fs + f1): k1 * (n1 * fs + f1 + 1),
                                   k2 * (n2 * fs + f2): k2 * (n2 * fs + f2 + 1)] = \
                                   w_[(f1 * fs + f2) * k1: (f1 * fs + f2 + 1) * k1,
                                      (n1 * c2 + n2) * k2: (n1 * c2 + n2 + 1) * k2]

        return square
