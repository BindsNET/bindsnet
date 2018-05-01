import os
import sys
import torch
import matplotlib.pyplot as plt

from time                      import time

from bindsnet.network          import Network
from bindsnet.network.topology import Connection
from bindsnet.network.nodes    import Input, LIFNodes

n = 5000

network = Network()
input_ = Input(n)
layer = LIFNodes(n)
conn = Connection(input_, layer, w=torch.diag(torch.ones(n)))

network.add_layer(input_, name='X')
network.add_layer(layer, name='A')
network.add_connection(conn, source='X', target='A')

t = 10000
inpts = {'X' : 10 * torch.rand(t, n)}

t1 = time(); spikes = network.run(inpts, t); t2 = time()
print('Time:', t2 - t1)

fig, ax = plt.subplots(figsize=(16, 9))

ax.matshow(spikes['A'].t(), cmap='binary')
ax.set_xticks(()); ax.set_yticks(())
ax.set_aspect('auto')

plt.show()
