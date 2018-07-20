import torch
import numpy as np


def select_multinomial(pipeline, **kwargs):
    '''
    Selects an action probabilistically based on spiking activity from a network layer.
    
    Inputs:
    
        | :code:`pipeline` (:code:`bindsnet.pipeline.Pipeline`): Pipeline
        with environment that has an integer action space.
    
    Returns:
    
        | (:code:`int`): Integer indicating an action from the action space.
    '''
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError('select_multinomial() requires an "output" layer argument.')
    
    assert pipeline.network.layers[output].n % pipeline.env.action_space.n == 0, \
           'Output layer size not equal to size of action space.'
    
    pop_size = int(pipeline.network.layers[output].n / pipeline.env.action_space.n)
    
    spikes = pipeline.network.layers[output].s
    _sum = spikes.sum().float()
    
    # Choose action based on population's spiking.
    if _sum == 0:
        action = np.random.choice(range(pipeline.env.action_space.n))
    else:
        pop_spikes = torch.Tensor([spikes[(i * pop_size):(i * pop_size) + pop_size].sum() \
                                          for i in range(pipeline.network.layers[output].n)]).float()
        action = torch.multinomial((pop_spikes / _sum).view(-1), 1)[0]
    
    return action

def select_softmax(pipeline, **kwargs):
    '''
    Selects an action using softmax function based on spiking from a network layer.
    
    Inputs:
    
        | :code:`pipeline` (:code:`bindsnet.pipeline.Pipeline`): Pipeline
        with environment that accepts feedback in the form of actions.
    
    Returns:
    
        | (:code:`int`): Number indicating the desired action from the action space.
    '''
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
        action = np.random.choice(range(pipeline.env.action_space.n))
    else:
        action = torch.multinomial(( torch.exp(spikes.float()) / _sum ).view(-1), 1)[0]
    
    return action
    
def select_random(pipeline, **kwargs):
    '''
    Selects an action randomly from the action space.
    
    Inputs:
    
        | :code:`pipeline` (:code:`bindsnet.pipeline.Pipeline`): Pipeline
        with environment that accepts feedback in the form of actions.
    
    Returns:
    
        | (:code:`int`): Number indicating the randomly selected action from the action space.
    '''
    # Choose action randomly from the action space.
    return np.random.choice(range(pipeline.env.action_space.n))
