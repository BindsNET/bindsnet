import torch

from bindsnet.pipeline.environment_pipeline import EnvironmentPipeline


def select_multinomial(pipeline: EnvironmentPipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action probabilistically based on spiking activity from a network layer.

    :param pipeline: EnvironmentPipeline with environment that has an integer action
        space.
    :return: Action sampled from multinomial over activity of similarly-sized output
        layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs["output"]
    except KeyError:
        raise KeyError('select_multinomial() requires an "output" layer argument.')

    output = pipeline.network.layers[output]
    action_space = pipeline.env.action_space

    assert output.n % action_space.n == 0, (
        f"Output layer size of {output.n} is not divisible by action space size of"
        f" {action_space.n}."
    )

    pop_size = int(output.n / action_space.n)
    spikes = output.s
    _sum = spikes.sum().float()

    # Choose action based on population's spiking.
    if _sum == 0:
        action = torch.randint(low=0, high=pipeline.env.action_space.n, size=(1,))[0]
    else:
        pop_spikes = torch.tensor(
            [
                spikes[(i * pop_size) : (i * pop_size) + pop_size].sum()
                for i in range(action_space.n)
            ],
            device=spikes.device,
        )
        action = torch.multinomial((pop_spikes.float() / _sum).view(-1), 1)[0].item()

    return action


def select_softmax(pipeline: EnvironmentPipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action using softmax function based on spiking from a network layer.

    :param pipeline: EnvironmentPipeline with environment that has an integer action
        space and :code:`spike_record` set.
    :return: Action sampled from softmax over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs["output"]
    except KeyError:
        raise KeyError('select_softmax() requires an "output" layer argument.')

    assert (
        pipeline.network.layers[output].n == pipeline.env.action_space.n
    ), "Output layer size is not equal to the size of the action space."

    assert hasattr(
        pipeline, "spike_record"
    ), "EnvironmentPipeline is missing the attribute: spike_record."

    spikes = torch.sum(pipeline.spike_record[output], dim=0)
    probabilities = torch.softmax(spikes, dim=0)
    return torch.multinomial(probabilities, num_samples=1).item()


def select_highest(pipeline: EnvironmentPipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action with have the highst spikes. In case of equal spiking select randomly

    :param pipeline: EnvironmentPipeline with environment that has an integer action
        space and :code:`spike_record` set.
    :return: Action sampled from softmax over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs["output"]
    except KeyError:
        raise KeyError('select_softmax() requires an "output" layer argument.')

    assert (
        pipeline.network.layers[output].n == pipeline.env.action_space.n
    ), "Output layer size is not equal to the size of the action space."

    assert hasattr(
        pipeline, "spike_record"
    ), "EnvironmentPipeline is missing the attribute: spike_record."

    spikes = torch.sum(pipeline.spike_record[output], dim=0).squeeze()
    action = torch.where(spikes == spikes.max())[0]
    if torch.sum(spikes) == 0:
        # choose random between fire(1) start(0).
        action[0] = torch.randint(low=0, high=1, size=(1,))[0]
    # elif action.shape[0] > 1:
    #     # shuffle the action array and place the random index at the first location.
    #     p = torch.randint(low=0, high=action.shape[0], size=(1,))[0]
    #     action[0] = action[p]

    return action[0].item()


def select_first_spike(pipeline: EnvironmentPipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action with have the highst spikes. In case of equal spiking select randomly

    :param pipeline: EnvironmentPipeline with environment that has an integer action
        space and :code:`spike_record` set.
    :return: Action sampled from softmax over activity of similarly-sized output layer.

    Keyword arguments:

    :param str output: Name of output layer whose activity to base action selection on.
    """
    try:
        output = kwargs["output"]
    except KeyError:
        raise KeyError('select_softmax() requires an "output" layer argument.')

    assert (
        pipeline.network.layers[output].n == pipeline.env.action_space.n
    ), "Output layer size is not equal to the size of the action space."

    assert hasattr(
        pipeline, "spike_record"
    ), "EnvironmentPipeline is missing the attribute: spike_record."

    spikes = pipeline.spike_record[output].squeeze().squeeze().nonzero()
    if spikes.shape[0] == 0:
        # choose random between fire(1) start(0).
        action = torch.randint(low=0, high=1, size=(1,))[0]
    else:
        action = spikes[0, 1]

    return action


def select_random(pipeline: EnvironmentPipeline, **kwargs) -> int:
    # language=rst
    """
    Selects an action randomly from the action space.

    :param pipeline: EnvironmentPipeline with environment that has an integer action
        space.
    :return: Action randomly sampled over size of pipeline's action space.
    """
    # Choose action randomly from the action space.
    # return np.random.choice(pipeline.env.action_space.n)
    return torch.randint(low=0, high=pipeline.env.action_space.n, size=(1,))[0]
