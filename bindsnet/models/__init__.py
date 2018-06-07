import torch

from ..network  import *
from ..learning import *


class TwoLayerNetwork(Network):
    '''
    Implements an :code:`Input` instance connected to a :code:`LIFNodes` instance with a fully-connected :code:`Connection`.
    '''
    def __init__(self, n_inpt, n_neurons=100, dt=1.0, nu_pre=1e-4, nu_post=1e-2, wmin=0, wmax=1, norm=78.4):
        '''
        Inputs:
        
            | :code:`n_input` (:code:`int`): Number of input neurons. Matches the 1D size of the input data.
            | :code:`n_neurons` (:code:`int`): Number of excitatory, inhibitory neurons.
            | :code:`dt` (:code:`float`): Simulation time step.
            | :code:`norm` (:code:`float`): Input to excitatory layer connection weights norm.
        '''
        super().__init__(dt=dt)
        
        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.dt = dt
        
        self.add_layer(Input(n=self.n_inpt,
                             traces=True,
                             trace_tc=5e-2),
                       name='X')
        
        self.add_layer(LIFNodes(n=self.n_neurons,
                                traces=True,
                                rest=-65.0,
                                reset=-65.0,
                                thresh=-52.0,
                                refrac=5,
                                decay=1e-2,
                                trace_tc=5e-2),
                       name='Y')
        
        self.add_connection(Connection(source=self.layers['X'],
                                       target=self.layers['Y'],
                                       w=0.3 * torch.rand(self.n_inpt, self.n_neurons),
                                       update_rule=post_pre,
                                       nu_pre=nu_pre,
                                       nu_post=nu_post,
                                       wmin=wmin,
                                       wmax=wmax,
                                       norm=norm),
                            source='X',
                            target='Y')
        

class DiehlAndCook2015(Network):
    '''
    Implements the spiking neural network architecture from `(Diehl & Cook 2015) <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    '''
    def __init__(self, n_inpt, n_neurons=100, exc=22.5, inh=17.5, dt=1.0, nu_pre=1e-4,
                 nu_post=1e-2, wmin=0, wmax=1, norm=78.4, theta_plus=0.05, theta_decay=1e-7,
                 X_Ae_decay=None, Ae_Ai_decay=None, Ai_Ae_decay=None):
        '''
        Inputs:
        
            | :code:`n_input` (:code:`int`): Number of input neurons. Matches the 1D size of the input data.
            | :code:`n_neurons` (:code:`int`): Number of excitatory, inhibitory neurons.
            | :code:`exc` (:code:`float`): Strength of synapse weights from excitatory to inhibitory layer.
            | :code:`inh` (:code:`float`): Strength of synapse weights from inhibitory to excitatory layer.
            | :code:`dt` (:code:`float`): Simulation time step.
            | :code:`norm` (:code:`float`): Input to excitatory layer connection weights norm.
        '''
        super().__init__(dt=dt)
        
        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
        
        self.add_layer(Input(n=self.n_inpt,
                             traces=True,
                             trace_tc=5e-2),
                       name='X')
        
        self.add_layer(DiehlAndCookNodes(n=self.n_neurons,
                                         traces=True,
                                         rest=-65.0,
                                         reset=-60.0,
                                         thresh=-52.0,
                                         refrac=5,
                                         decay=1e-2,
                                         trace_tc=5e-2,
                                         theta_plus=theta_plus,
                                         theta_decay=theta_decay),
                       name='Ae')
        
        self.add_layer(LIFNodes(n=self.n_neurons,
                                traces=True,
                                rest=-60.0,
                                reset=-45.0,
                                thresh=-40.0,
                                decay=1e-1,
                                refrac=2,
                                trace_tc=5e-2),
                       name='Ai')
        
        self.add_connection(Connection(source=self.layers['X'],
                                       target=self.layers['Ae'],
                                       w=0.3 * torch.rand(self.n_inpt, self.n_neurons),
                                       update_rule=post_pre,
                                       nu_pre=nu_pre,
                                       nu_post=nu_post,
                                       wmin=wmin,
                                       wmax=wmax,
                                       norm=norm,
                                       decay=X_Ae_decay),
                            source='X',
                            target='Ae')
        
        self.add_connection(Connection(source=self.layers['Ae'],
                                       target=self.layers['Ai'],
                                       w=self.exc * torch.diag(torch.ones(self.n_neurons)),
                                       wmin=0,
                                       wmax=self.exc,
                                       decay=Ae_Ai_decay),
                            source='Ae',
                            target='Ai')
        
        self.add_connection(Connection(source=self.layers['Ai'],
                                       target=self.layers['Ae'],
                                       w=-self.inh * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons))),
                                       wmin=-self.inh,
                                       wmax=0,
                                       decay=Ai_Ae_decay),
                            source='Ai',
                            target='Ae')
