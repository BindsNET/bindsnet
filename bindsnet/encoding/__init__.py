import torch
import numpy as np

from typing import Optional, Union, Iterable, Iterator


def bernoulli(datum: torch.Tensor, time: Optional[int] = None, **kwargs) -> torch.Tensor:
    # language=rst
    """

    :param datum: Generates Bernoulli-distributed spike trains based on input intensity. Inputs must be non-negative.
                  Spikes correspond to successful Bernoulli trials, with success probability equal to (normalized in
                  [0, 1]) input value.
    :param time: Tensor of shape ``[n_1, ..., n_k]``.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get('max_prob', 1.0)
    assert 0 <= max_prob <= 1, 'Maximum firing probability must be in range [0, 1]'

    datum = np.copy(datum)
    shape, size = datum.shape, datum.size
    datum = datum.ravel()

    # Normalize inputs and rescale (spike probability proportional to normalized intensity).
    if datum.max() > 1.0:
        datum /= datum.max()

    datum *= max_prob

    # Make spike data from Bernoulli sampling.
    if time is None:
        s = np.random.binomial(1, datum, [size])
        s = s.reshape([*shape])
    else:
        s = np.random.binomial(1, datum, [time, size])
        s = s.reshape([time, *shape])

    return torch.Tensor(s).byte()


def bernoulli_loader(data: Union[torch.Tensor, Iterable[torch.Tensor]], time: Optional[int] = None,
                     **kwargs) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.bernoulli`` to iteratively encode a sequence of data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get('max_prob', 1.0)

    for i in range(len(data)):
        yield bernoulli(data[i], time, max_prob=max_prob)  # Encode datum as Bernoulli spike trains.


def poisson(datum: torch.Tensor, time: int, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Generates Poisson-distributed spike trains based on input intensity. Inputs must be non-negative.
    :param datum: Tensor of shape ``[n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    datum = np.copy(datum)
    shape, size = datum.shape, datum.size
    datum = datum.ravel()

    # Invert inputs (firing rate inverse of inter-arrival time).
    datum[datum != 0] = 1 / datum[datum != 0] * 1000

    # Make spike data from Poisson sampling.
    s_times = np.random.poisson(datum, [time, size])
    s_times = np.cumsum(s_times, axis=0)
    s_times[s_times >= time] = 0

    # Create spike trains from spike times.
    s = np.zeros([time, size])
    for i in range(time):
        s[s_times[i], np.arange(size)] = 1

    s[0, :] = 0
    s = s.reshape([time, *shape])

    return torch.Tensor(s).byte()


def poisson_loader(data: Union[torch.Tensor, Iterable[torch.Tensor]], time: int, **kwargs) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.poisson`` to iteratively encode a sequence of data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    for i in range(len(data)):
        yield poisson(data[i], time)  # Encode datum as Poisson spike trains.


def rank_order(datum: torch.Tensor, time: int, **kwargs) -> torch.Tensor:
    # language=rst
    """
    Encodes data via a rank order coding-like representation. One spike per neuron, temporally ordered by decreasing
    intensity. Inputs must be non-negative.

    :param datum: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :return: Tensor of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    datum = np.copy(datum)
    shape, size = datum.shape, datum.size
    datum = datum.ravel()

    # Compute single spike times in order of decreasing intensity.
    datum /= datum.max()
    s_times = np.zeros(size)
    s_times[datum != 0] = 1 / datum[datum != 0]
    s_times *= time / s_times.max()
    s_times = np.ceil(s_times).astype(np.int)
    s_times[s_times > time] = time

    # Create spike times tensor.
    s = np.zeros([time, size])
    for i in range(size):
        if s_times[i] != 0:
            s[s_times[i] - 1, i] = 1

    s = s.reshape([time, *shape])

    return torch.Tensor(s).byte()


def rank_order_loader(data: Union[torch.Tensor, Iterable[torch.Tensor]], time: int, **kwargs) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.rank_order`` to iteratively encode a sequence of data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    for i in range(len(data)):
        yield rank_order(data[i], time)  # Encode datum as rank order-encoded spike trains.
