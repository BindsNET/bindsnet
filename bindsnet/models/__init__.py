from typing import Optional

import torch

from ..network import Network
from ..learning import PostPre
from ..network.topology import Connection
from ..network.nodes import *


class TwoLayerNetwork(Network):
    # language=rst
    """
    Implements an ``Input`` instance connected to a ``LIFNodes`` instance with a fully-connected ``Connection``.
    """
    def __init__(self, n_inpt: int, n_neurons: int = 100, dt: float = 1.0, nu_pre: float = 1e-4, nu_post: float = 1e-2,
                 wmin: float = 0.0, wmax: float = 1.0, norm: float = 78.4) -> None:
        # language=rst
        """
        Constructor for class ``TwoLayerNetwork``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in the ``LIFNodes`` population.
        :param dt: Simulation time step.
        :param nu_pre: Pre-synaptic learning rate.
        :param nu_post: Post-synaptic learning rate.
        :param wmin: Minimum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param wmax: Maximum allowed weight on ``Input`` to ``LIFNodes`` synapses.
        :param norm: ``Input`` to ``LIFNodes`` layer connection weights normalization constant.
        """
        super().__init__(dt=dt)
        
        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.dt = dt
        
        self.add_layer(Input(n=self.n_inpt, traces=True, trace_tc=5e-2), name='X')
        self.add_layer(LIFNodes(n=self.n_neurons, traces=True, rest=-65.0, reset=-65.0, thresh=-52.0, refrac=5,
                                decay=1e-2, trace_tc=5e-2), name='Y')

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        self.add_connection(Connection(source=self.layers['X'], target=self.layers['Y'], w=w, update_rule=PostPre,
                                       nu=(nu_pre, nu_post), wmin=wmin, wmax=wmax, norm=norm),
                            source='X', target='Y')
        

class DiehlAndCook2015(Network):
    # language=rst
    """
    Implements the spiking neural network architecture from `(Diehl & Cook 2015)
    <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_.
    """
    def __init__(self, n_inpt: int, n_neurons: int = 100, exc: float = 22.5, inh: float = 17.5, dt: float = 1.0,
                 nu_pre: float = 1e-4, nu_post: float = 1e-2, wmin: float = 0.0, wmax: float = 1.0, norm: float = 78.4,
                 theta_plus: float = 0.05, theta_decay: float = 1e-7, X_Ae_decay: Optional[float] = None,
                 Ae_Ai_decay: Optional[float] = None, Ai_Ae_decay: Optional[float] = None) -> None:
        # language=rst
        """
        Constructor for class ``DiehlAndCook2015``.

        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of excitatory, inhibitory neurons.
        :param exc: Strength of synapse weights from excitatory to inhibitory layer.
        :param inh: Strength of synapse weights from inhibitory to excitatory layer.
        :param dt: Simulation time step.
        :param nu_pre: Pre-synaptic learning rate.
        :param nu_post: Post-synaptic learning rate.
        :param wmin: Minimum allowed weight on input to excitatory synapses.
        :param wmax: Maximum allowed weight on input to excitatory synapses.
        :param norm: Input to excitatory layer connection weights normalization constant.
        :param theta_plus: On-spike increment of ``DiehlAndCookNodes`` membrane threshold potential.
        :param theta_decay: Time constant of ``DiehlAndCookNodes`` threshold potential decay.
        :param X_Ae_decay: Decay of activation of connection from input to excitatory neurons.
        :param Ae_Ai_decay: Decay of activation of connection from excitatory to inhibitory neurons.
        :param Ai_Ae_decay: Decay of activation of connection from inhibitory to excitatory neurons.
        """
        super().__init__(dt=dt)
        
        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.exc = exc
        self.inh = inh
        self.dt = dt
        
        self.add_layer(Input(n=self.n_inpt, traces=True, trace_tc=5e-2), name='X')
        self.add_layer(DiehlAndCookNodes(n=self.n_neurons, traces=True, rest=-65.0, reset=-60.0, thresh=-52.0, refrac=5,
                                         decay=1e-2, trace_tc=5e-2, theta_plus=theta_plus, theta_decay=theta_decay),
                       name='Ae')
        
        self.add_layer(LIFNodes(n=self.n_neurons, traces=False, rest=-60.0, reset=-45.0, thresh=-40.0, decay=1e-1,
                                refrac=2, trace_tc=5e-2),
                       name='Ai')

        w = 0.3 * torch.rand(self.n_inpt, self.n_neurons)
        self.add_connection(Connection(source=self.layers['X'], target=self.layers['Ae'], w=w, update_rule=PostPre,
                                       nu=(nu_pre, nu_post), wmin=wmin, wmax=wmax, norm=norm, decay=X_Ae_decay),
                            source='X', target='Ae')

        w = self.exc * torch.diag(torch.ones(self.n_neurons))
        self.add_connection(Connection(source=self.layers['Ae'], target=self.layers['Ai'], w=w, wmin=0, wmax=self.exc,
                                       decay=Ae_Ai_decay),
                            source='Ae', target='Ai')

        w = -self.inh * (torch.ones(self.n_neurons, self.n_neurons) - torch.diag(torch.ones(self.n_neurons)))
        self.add_connection(Connection(source=self.layers['Ai'], target=self.layers['Ae'], w=w, wmin=-self.inh, wmax=0,
                                       decay=Ai_Ae_decay),
                            source='Ai', target='Ae')


class CANs(Network):

    def __init__(self, n_inpt: int, n_neurons: int = 1000, n_CANs: int = 4, dt: float = 1.0,
                 running_time:int = 500, nu_pre: float = 1e-4, nu_post: float = 1e-2) -> None:
        # language=rst
        """
        :param n_inpt: Number of input neurons. Matches the 1D size of the input data.
        :param n_neurons: Number of neurons in each CAN unit.
        :param n_CANs: Number of CAN units.
        :param dt: Simulation time step.
        :param running_time: Minimum iteration to run for memory allocations.
        :param nu_pre: Pre-synaptic learning rate.
        :param nu_post: Post-synaptic learning rate.

        """
        super().__init__(dt=dt)

        self.n_inpt = n_inpt
        self.n_neurons = n_neurons
        self.n_CANs = n_CANs
        self.running_time = running_time

        In = Input(n=self.n_inpt)
        self.add_layer(layer=In, name='Input Layer')

        for i in range(self.n_CANs):
            # add CAN column
            CAN = IzhikevichNodes(n=self.n_neurons, excitatory=0.8)
            # CAN = nodes.AdaptiveLIFNodes(n=Nu_Neurons_in_CAN)

            # create random weights for internal connection of CAN
            w = torch.rand(self.n_neurons, self.n_neurons)
            # make the weights of inhibitory neurons negative
            wi = torch.zeros(self.n_inpt, self.n_neurons)
            # wi[i, :] = torch.rand(Nu_Neurons_in_CAN)

            temp = CAN.excitatory == True
            if torch.sum(temp) > 0:
                w = torch.where(temp, 0.5 * w, -w)
                # create weights between input to output.
                # Excitatory neurons get 5 time the input and Inhibitory 1 time the input
                wi[i, :] = torch.where(temp, 5. * torch.ones(self.n_neurons), 1. * torch.ones(self.n_neurons))

            # connect the internal connection in CAN
            C = Connection(source=CAN, target=CAN, w=w)
            # connect the CAN to the input layer
            Ci = Connection(source=In, target=CAN, w=wi)

            temp_CAN_name = 'CAN-' + str(i)
            self.add_layer(layer=CAN, name=temp_CAN_name)
            self.add_connection(connection=C, source=temp_CAN_name, target=temp_CAN_name)
            self.add_connection(connection=Ci, source='Input Layer', target=temp_CAN_name)
