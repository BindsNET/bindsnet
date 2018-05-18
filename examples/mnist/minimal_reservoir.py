import torch
from bindsnet.datasets         import MNIST
from bindsnet.network          import Network
from bindsnet.pipeline         import Pipeline
from bindsnet.network.topology import Connection
from bindsnet.encoding         import poisson_loader
from bindsnet.network.nodes    import Input, LIFNodes
from bindsnet.environment      import DatasetEnvironment

# Build a simple two-layer, input-output network.
network = Network(dt=1.0)
inpt = Input(784, shape=(28, 28)); network.add_layer(inpt, name='I')
output = LIFNodes(500, thresh=-52 + 2 * torch.randn(500)); network.add_layer(output, name='O')
connection = Connection(source=inpt,
						target=output,
						w=torch.torch.randn(inpt.n, output.n)); network.add_connection(connection, source='I', target='O')

# Get MNIST training images and labels.
images, labels = MNIST(path='../../data/MNIST').get_train()

# Create lazily iterating Poisson-distributed data loader.
loader = zip(poisson_loader(images, time=250), iter(labels))

# Train a linear readout.
for datum, label in loader:
	network.run(inpts={'I' : datum}, time=250)
	network._reset()
