from typing import Iterable, Iterator, Optional, Union

import torch

from bindsnet.encoding.encodings import bernoulli, poisson, rank_order


def bernoulli_loader(
    data: Union[torch.Tensor, Iterable[torch.Tensor]],
    time: Optional[int] = None,
    dt: float = 1.0,
    **kwargs,
) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.bernoulli`` to iteratively encode a sequence of
    data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of Bernoulli spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of Bernoulli-distributed spikes.

    Keyword arguments:

    :param float max_prob: Maximum probability of spike per Bernoulli trial.
    """
    # Setting kwargs.
    max_prob = kwargs.get("dt", 1.0)

    for i in range(len(data)):
        # Encode datum as Bernoulli spike trains.
        yield bernoulli(datum=data[i], time=time, dt=dt, max_prob=max_prob)


def poisson_loader(
    data: Union[torch.Tensor, Iterable[torch.Tensor]],
    time: int,
    dt: float = 1.0,
    **kwargs,
) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.poisson`` to iteratively encode a sequence of
    data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of Poisson spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of Poisson-distributed spikes.
    """
    for i in range(len(data)):
        # Encode datum as Poisson spike trains.
        yield poisson(datum=data[i], time=time, dt=dt)


def rank_order_loader(
    data: Union[torch.Tensor, Iterable[torch.Tensor]],
    time: int,
    dt: float = 1.0,
    **kwargs,
) -> Iterator[torch.Tensor]:
    # language=rst
    """
    Lazily invokes ``bindsnet.encoding.rank_order`` to iteratively encode a sequence of
    data.

    :param data: Tensor of shape ``[n_samples, n_1, ..., n_k]``.
    :param time: Length of rank order-encoded spike train per input variable.
    :param dt: Simulation time step.
    :return: Tensors of shape ``[time, n_1, ..., n_k]`` of rank order-encoded spikes.
    """
    for i in range(len(data)):
        # Encode datum as rank order-encoded spike trains.
        yield rank_order(datum=data[i], time=time, dt=dt)
