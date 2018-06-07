import torch
import pickle as p

from .nodes    import *
from .topology import *
from .monitors import *


def load_network(fname):
    '''
    Loads serialized network object from disk.
    
    Inputs:
    
        | :code:`fname` (:code:`str`): Path to serialized network object on disk.
    '''
    try:
        with open(fname, 'rb') as f:
            return p.load(open(fname, 'rb'))
    except FileNotFoundError:
        print('Network not found on disk.')


class Network:
    '''
    Most important object of the :code:`bindsnet` package. Responsible for the simulation and interaction of nodes and connections.
    
    **Example:**
    
    .. code-block:: python
    
        import torch
        import matplotlib.pyplot as plt
        
        from bindsnet         import encoding
        from bindsnet.network import Network, nodes, topology, monitors
        
        network = Network(dt=1.0)  # Instantiates network.
        
        X = nodes.Input(100)  # Input layer.
        Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
        C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.
        
        # Spike monitor objects.
        M1 = monitors.Monitor(obj=X, state_vars=['s'])
        M2 = monitors.Monitor(obj=Y, state_vars=['s'])
        
        # Add everything to the network object.
        network.add_layer(layer=X, name='X')
        network.add_layer(layer=Y, name='Y')
        network.add_connection(connection=C, source='X', target='Y')
        network.add_monitor(monitor=M1, name='X')
        network.add_monitor(monitor=M2, name='Y')
        
        # Create Poisson-distributed spike train inputs.
        data = 15 * torch.rand(1, 100)  # Generate random Poisson rates for 100 input neurons.
        trains = encoding.get_poisson(data=data, time=5000)  # Encode input as 5000ms Poisson spike trains.
        
        # Simulate network on generated spike trains.
        for train in trains:
            inpts = {'X' : train}  # Create inputs mapping.
            network.run(inpts=inpts, time=5000)  # Run network simulation.
            
        # Plot spikes of input and output layers.
        spikes = {'X' : M1.get('s'), 'Y' : M2.get('s')}
        
        fig, axes = plt.subplots(2, 1, figsize=(12, 7))
        for i, layer in enumerate(spikes):
            axes[i].matshow(spikes[layer], cmap='binary')
            axes[i].set_title('%s spikes' % layer)
            axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
            axes[i].set_xticks(()); axes[i].set_yticks(())
            axes[i].set_aspect('auto')
        
        plt.tight_layout(); plt.show()
    '''
    def __init__(self, dt=1.0):
        '''
        Initializes network object. 
        
        Inputs:
        
            | :code:`dt` (:code:`float`): Simulation timestep. All other 
                objects' time constants are relative to this value.
        '''
        self.dt = dt
        self.layers = {}
        self.connections = {}
        self.monitors = {}

    def add_layer(self, layer, name):
        '''
        Adds a layer of nodes to the network.
        
        Inputs:
        
            | :code:`layer` (:code:`bindsnet.nodes.Nodes`): A subclass of the :code:`Nodes` object.
            | :code:`name` (:code:`str`): Logical name of layer.
        '''
        self.layers[name] = layer

    def add_connection(self, connection, source, target):
        '''
        Adds a connection between layers of nodes to the network.
        
        Inputs:
        
            | :code:`connection` (:code:`bindsnet.topology.Connection`): An instance of class :code:`Connection`.
            | :code:`source` (:code:`str`): Logical name of the connection's source layer.
            | :code:`target` (:code:`str`): Logical name of the connection's target layer.
        '''
        self.connections[(source, target)] = connection

    def add_monitor(self, monitor, name):
        '''
        Adds a monitor on a network object to the network.
        
        Inputs:
        
            | :code:`monitor` (:code:`bindsnet.Monitor`): An instance of class :code:`Monitor`.
            | :code:`name` (:code:`str`): Logical name of monitor object.
        '''
        self.monitors[name] = monitor

    def save(self, fname):
        '''
        Serializes the network object to disk.
        
        Inputs:
        
            | :code:`fname` (:code:`str`): Path to store serialized network object on disk.
        
        **Example:**
        
        .. code-block:: python
        
            import torch                                         
            import matplotlib.pyplot as plt

            from pathlib          import Path
            from bindsnet.network import *
            from bindsnet.network import topology

            # Build simple network.
            network = Network(dt=1.0)

            X = nodes.Input(100)  # Input layer.
            Y = nodes.LIFNodes(100)  # Layer of LIF neurons.
            C = topology.Connection(source=X, target=Y, w=torch.rand(X.n, Y.n))  # Connection from X to Y.

            # Add everything to the network object.
            network.add_layer(layer=X, name='X')
            network.add_layer(layer=Y, name='Y')
            network.add_connection(connection=C, source='X', target='Y')

            # Save the network to disk.
            network.save(str(Path.home()) + '/network.p')
        '''
        p.dump(self, open(fname, 'wb'))

    def get_inputs(self):
        '''
        Fetches outputs from network layers for input to downstream layers.
        
        Returns:
        
            | (:code:`dict[torch.Tensor or torch.cuda.Tensor]`): Inputs to all layers for the current iteration.
        '''
        inpts = {}
        
        # Loop over network connections.
        for key in self.connections:
            # Fetch source and target populations.
            source = self.connections[key].source
            target = self.connections[key].target
            
            if not key[1] in inpts:
                inpts[key[1]] = torch.zeros(target.shape)

            # Add to input: source's spikes multiplied by connection weights.
            inpts[key[1]] += self.connections[key].compute(source.s)
            
        return inpts

    def run(self, inpts, time, **kwargs):
        '''
        Simulation network for given inputs and time.
        
        Inputs:
        
            | :code:`inpts` (:code:`dict`): Dictionary of :code:`Tensor`s of shape :code:`[time, n_input]`.
            | :code:`time` (:code:`int`): Simulation time.

            Keyword arguments:

                | :code:`clamps` (:code:`dict`): Mapping of layer names to neurons which to "clamp" to spiking.
                | :code:`reward` (:code:`float`): Scalar value used in reward-modulated learning.
        
        **Example:**
    
        .. code-block:: python
        
            import torch
            import matplotlib.pyplot as plt
            
            from bindsnet.network import *
            from bindsnet.network.monitors import Monitor
            
            # Build simple network.
            network = Network()
            network.add_layer(Input(500), name='I')
            network.add_monitor(Monitor(network.layers['I'], state_vars=['s']), 'I')
            
            # Generate spikes by running Bernoulli trials on Uniform(0, 0.5) samples.
            spikes = torch.bernoulli(0.5 * torch.rand(500, 500))
            
            # Run network simulation.
            network.run(inpts={'I' : spikes}, time=500)
            
            # Look at input spiking activity.
            spikes = network.monitors['I'].get('s')
            plt.matshow(spikes, cmap='binary')
            plt.xticks(()); plt.yticks(());
            plt.xlabel('Time'); plt.ylabel('Neuron index')
            plt.title('Input spiking')
            plt.show()
        '''
        # Parse keyword arguments.
        clamps = kwargs.get('clamp', {})
        reward = kwargs.get('reward', None)
        
        # Effective number of timesteps
        timesteps = int(time / self.dt)

        # Get input to all layers.
        inpts.update(self.get_inputs())
        
        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):
            # Update each layer of nodes.
            for l in self.layers:
                if type(self.layers[l]) is Input:
                    self.layers[l].step(inpts[l][t, :], self.dt)
                else:
                    self.layers[l].step(inpts[l], self.dt)
                
                # Force neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    self.layers[l].s[clamp] = 1

            # Run synapse updates.
            for c in self.connections:
                self.connections[c].update(reward=reward)

            # Get input to all layers.
            inpts.update(self.get_inputs())

            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()
        
        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()

    def _reset(self):
        '''
        Reset state variables of objects in network.
        '''
        for layer in self.layers:
            self.layers[layer]._reset()

        for connection in self.connections:
            self.connections[connection]._reset()

        for monitor in self.monitors:
            self.monitors[monitor]._reset()
