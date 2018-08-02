import torch
import numpy as np

from . import Pipeline


def select_multinomial(pipeline: Pipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action probabilistically based on spiking activity from a network layer.

    :param pipeline: Pipeline with environment that has an integer action space.
    :return: Action sampled from multinomial over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError('select_multinomial() requires an "output" layer argument.')

    output = pipeline.network.layers[output]
    action_space = pipeline.env.action_space

    assert output.n % action_space.n == 0, 'Output layer size not equal to size of action space.'

    pop_size = int(output.n / action_space.n)
    spikes = output.s
    _sum = spikes.sum().float()

    # Choose action based on population's spiking.
    if _sum == 0:
        action = np.random.choice(pipeline.env.action_space.n)
    else:
        pop_spikes = torch.Tensor([spikes[(i * pop_size):(i * pop_size) + pop_size].sum() for i in range(output.n)])
        action = torch.multinomial((pop_spikes.float() / _sum).view(-1), 1)[0]

    return action


def select_softmax(pipeline: Pipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action using softmax function based on spiking from a network layer.

    :param pipeline: Pipeline with environment that has an integer action space.
    :return: Action sampled from softmax over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError('select_softmax() requires an "output" layer argument.')

    assert pipeline.network.layers[output].n == pipeline.env.action_space.n, \
        'Output layer size not equal to size of action space.'

    # Sum of previous iterations' spikes (Not yet implemented)
    spikes = pipeline.network.layers[output].s
    _sum = torch.sum(torch.exp(spikes.float()))

    # Choose action based on readout neuron spiking
    if _sum == 0:
        action = np.random.choice(pipeline.env.action_space.n)
    else:
        action = torch.multinomial((torch.exp(spikes.float()) / _sum).view(-1), 1)[0]

    return action


def select_random(pipeline: Pipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action randomly from the action space.

    :param pipeline: Pipeline with environment that has an integer action space.
    :return: Action randomly sampled over size of pipeline's action space.
    """
    # Choose action randomly from the action space.
    return np.random.choice(pipeline.env.action_space.n)
