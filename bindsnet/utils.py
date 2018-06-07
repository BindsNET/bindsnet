import torch
import numpy as np


def get_im2col_indices(x_shape, field_height, field_width, padding=1, stride=1):
    '''
    First figure out what the size of the output should be. Taken from `this repository <https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py>`_.
    '''
    N, C, H, W = x_shape
    
    assert (H + 2 * padding[0] - field_height) % stride[0] == 0
    assert (W + 2 * padding[1] - field_height) % stride[1] == 0
    
    out_height = int((H + 2 * padding[0] - field_height) / stride[0] + 1)
    out_width = int((W + 2 * padding[1] - field_width) / stride[1] + 1)
    
    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, C)
    i1 = stride[0] * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * C)
    j1 = stride[1] * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)

    k = np.repeat(np.arange(C), field_height * field_width).reshape(-1, 1)
    
    return (k, i, j)

def im2col_indices(x, field_height, field_width, padding=1, stride=1):
    '''
    An implementation of im2col based on some fancy indexing. Taken from `this repository <https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py>`_.
    '''
    # Zero-pad the input
    p = padding
    
    x_padded = np.pad(x, ((0, 0), (0, 0), (p[0], p[0]), (p[1], p[1])), mode='constant')

    k, i, j = get_im2col_indices(x.shape, field_height, field_width, padding, stride)

    cols = x_padded[:, k, i, j]
    C = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * C, -1)
    
    return torch.Tensor(cols)

def col2im_indices(cols, x_shape, field_height=3, field_width=3, padding=1, stride=1):
    '''
    An implementation of col2im based on fancy indexing and np.add.at. Taken from `this repository <https://github.com/huyouare/CS231n/blob/master/assignment2/cs231n/im2col.py>`_.
    '''
    N, C, H, W = x_shape
    H_padded, W_padded = H + 2 * padding, W + 2 * padding
    x_padded = np.zeros((N, C, H_padded, W_padded), dtype=cols.dtype)
    k, i, j = get_im2col_indices(x_shape, field_height, field_width, padding,
                               stride)
    cols_reshaped = cols.reshape(C * field_height * field_width, -1, N)
    cols_reshaped = cols_reshaped.transpose(2, 0, 1)
    np.add.at(x_padded, (slice(None), k, i, j), cols_reshaped)
    
    if padding == 0:
        return x_padded
    
    return x_padded[:, :, padding:-padding, padding:-padding]

def get_square_weights(weights, n_sqrt, side):
    '''
    Return a grid of a number of filters :code:`sqrt ** 2` with side lengths :code:`side`.
    '''
    square_weights = torch.zeros_like(torch.Tensor(side * n_sqrt, side * n_sqrt))
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            if not i * n_sqrt + j < weights.size(1):
                break
            
            fltr = weights[:, i * n_sqrt + j].contiguous().view(side, side)
            square_weights[i * side : (i + 1) * side, (j % n_sqrt) * side : ((j % n_sqrt) + 1) * side] = fltr
    
    return square_weights

def get_square_assignments(assignments, n_sqrt):
    '''
    Return a grid of assignments.
    '''
    square_assignments = -1 * torch.ones_like(torch.Tensor(n_sqrt, n_sqrt))
    for i in range(n_sqrt):
        for j in range(n_sqrt):
            if not i * n_sqrt + j < assignments.size(0):
                break
            
            assignment = assignments[i * n_sqrt + j]
            square_assignments[i : (i + 1), (j % n_sqrt) : ((j % n_sqrt) + 1)] = assignments[i * n_sqrt + j]
    
    return square_assignments