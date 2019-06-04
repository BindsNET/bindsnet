from typing import Optional, Union, Iterable, Iterator

import torch
import numpy as np


def single(
    datum: torch.Tensor, time: int, dt: float = 1.0, sparsity: float = 0.3, **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Generates timing based single-spike encoding. Spike occurs earlier if the
    intensity of the input feature is higher. Features whose value is lower than
    threshold is remain silent.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of the input and output.
    :param dt: Simulation time step.
    :param sparsity: Sparsity of the input representation. 0 for no spike and 1 for all spike.
    :return: Tensor of shape ``[time, n_1, ..., n_k]``.
    """
    time = int(time / dt)
    shape = list(datum.shape)
    datum = np.copy(datum)
    quantile = np.quantile(datum, 1 - sparsity)
    s = np.zeros([time, *shape])
    s[0] = np.where(datum > quantile, np.ones(shape), np.zeros(shape))
    return torch.Tensor(s).byte()


def repeat(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    :param datum: Repeats a tensor along a new dimension in the 0th position for ``int(time / dt)`` timesteps.
    :param time: Tensor of shape ``[n_1, ..., n_k]``.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of repeated data along the 0th dimension.
    """
    time = int(time / dt)
    return datum.repeat([time, *torch.ones(len(datum.shape))])


def bernoulli(
    datum: torch.Tensor, time: Optional[int] = None, dt: float = 1.0, **kwargs
) -> torch.Tensor:
    # language=rst
    """
    :param datum: Generates Bernoulli-distributed spike trains based on input intensity. Inputs must be non-negative.
                  Spikes correspond to successful Bernoulli trials, with success probability equal to (normalized in
                  [0, 1]) input value.
    :param time: Tensor of shape ``[n_1, ..., n_k]``.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get("max_prob", 1.0)

    assert 0 <= max_prob <= 1, "Maximum firing probability must be in range [0, 1]"
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)

    if time is not None:
        time = int(time / dt)

    # Normalize inputs and rescale (spike probability proportional to normalized intensity).
    if datum.max() > 1.0:
        datum /= datum.max()

    # Make spike data from Bernoulli sampling.
    if time is None:
        spikes = torch.bernoulli(max_prob * datum)
        spikes = spikes.view(*shape)
    else:
        spikes = torch.bernoulli(max_prob * datum.repeat([time, 1]))
        spikes = spikes.view(time, *shape)

    return spikes.byte()


def poisson(datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be non-negative, and give the
    firing rate in Hz. Inter-spike intervals (ISIs) for non-negative data incremented by one to avoid zero intervals
    while maintaining ISI distributions.

    For example, an input of intensity :code:`x` will have an average firing rate of :code:`x`Hz.

    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    # Get shape and size of data.
    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    # Compute firing rates in seconds as function of data intensity,
    # accounting for simulation time step.
    rate = torch.zeros(size)
    rate[datum != 0] = 1 / datum[datum != 0] * (1000 / dt)

    # Create Poisson distribution and sample inter-spike intervals
    # (incrementing by 1 to avoid zero intervals).
    dist = torch.distributions.Poisson(rate=rate)
    intervals = dist.sample(sample_shape=torch.Size([time]))
    intervals[:, datum != 0] += (intervals[:, datum != 0] == 0).float()

    # Calculate spike times by cumulatively summing over time dimension.
    times = torch.cumsum(intervals, dim=0).long()
    times[times >= time] = 0

    # Create tensor of spikes.
    spikes = torch.zeros(time, size).byte()
    spikes[times, torch.arange(size)] = 1
    spikes[0] = 0

    return spikes.view(time, *shape)


def rank_order(
    datum: torch.Tensor, time: int, dt: float = 1.0, **kwargs
) -> torch.Tensor:
    # language=rst
    """
    Encodes data via a rank order coding-like representation. One spike per neuron, temporally ordered by decreasing
    intensity. Inputs must be non-negative.

    :param datum: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    assert (datum >= 0).all(), "Inputs must be non-negative"

    shape, size = datum.shape, datum.numel()
    datum = datum.view(-1)
    time = int(time / dt)

    # Create spike times in order of decreasing intensity.
    datum /= datum.max()
    times = torch.zeros(size)
    times[datum != 0] = 1 / datum[datum != 0]
    times *= time / times.max()  # Extended through simulation time.
    times = torch.ceil(times).long()

    # Create spike times tensor.
    spikes = torch.zeros(time, size).byte()
    for i in range(size):
        if 0 < times[i] < time:
            spikes[times[i] - 1, i] = 1

    return spikes.reshape(time, *shape)
