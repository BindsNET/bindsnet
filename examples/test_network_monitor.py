import os
import sys
import torch
import numpy  as np
import pickle as p
import matplotlib.pyplot as plt

from time        import time, sleep

sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'network')))
sys.path.append(os.path.abspath(os.path.join('..', 'bindsnet', 'datasets')))

from network     import Network, NetworkMonitor
from connections import Connection
from nodes       import Input, LIFNodes

# Build a network.
network = Network()

n_inpt = 100
n_layer = 400

inpt = Input(n_inpt)
layer = LIFNodes(n_layer)
inpt_layer_connection = Connection(inpt, layer, w=torch.rand(n_inpt, n_layer))
layer_layer_connection = Connection(layer, layer, w=torch.rand(n_layer, n_layer))

network.add_layer(inpt, 'X')
network.add_layer(layer, 'A')
network.add_connection(inpt_layer_connection, source='X', target='A')
network.add_connection(layer_layer_connection, source='A', target='A')

# Add network-level monitor (specifying simulation length
# allows for faster monitoring via pre-allocation of memory).
t = 350
monitor = NetworkMonitor(network, state_vars=['v', 's', 'w'], time=t)
network.add_monitor(monitor, 'network')

# Run the simulation.
t1 = time()
network.run({'X' : torch.rand(t, n_inpt)}, t)
t2 = time()
print('Run 1 time:', t2 - t1)

# Reset network monitor.
monitor._reset()

# Re-run the simulation.
t1 = time()
network.run({'X' : torch.rand(t, n_inpt)}, t)
t2 = time()
print('Run 2 time:', t2 - t1)

# Write the simulation data to disk.
path = os.path.join('..', 'results', 'sim_results')
t1 = time(); monitor.save(path, fmt='npz'); t2 = time()
print('Write time:', t2 - t1)

# Read the simulation data back from disk.
t1 = time(); sim_data = np.load(path + '.npz'); t2 = time()

data = {}
for key in sim_data.keys():
	data[key] = sim_data[key]
	
print('Read time:', t2 - t1)
