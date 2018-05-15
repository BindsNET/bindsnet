from torch                import Tensor
from numpy.random         import choice
from bindsnet.datasets    import MNIST
from bindsnet.encoding    import poisson
from bindsnet.pipeline    import Pipeline
from bindsnet.models      import DiehlAndCook2015
from bindsnet.environment import DatasetEnvironment


# Build network.
network = DiehlAndCook2015(n_inpt=784,
						   n_neurons=100,
						   dt=1.0,
						   exc=22.5,
						   inh=17.5,
						   nu_pre=0,
						   nu_post=1e-2,
						   norm=78.4)

# Specify dataset wrapper environment.
environment = DatasetEnvironment(dataset=MNIST(path='../../data/MNIST'),
								 train=True,
								 intensity=0.25)

# Build pipeline from components.
pipeline = Pipeline(network=network,
					environment=environment,
					encoding=poisson,
					time=50,
				    plot_interval=1)

# Train the network.
labels = environment.labels
for i in range(60000):
	# Choose an output neuron to clamp to spiking behavior.
	c = choice(10, size=1, replace=False)
	clamp = {'Ae' : 10 * labels[i].long() + Tensor(c).long()}
	
	# Run a step of the pipeline with clamped neuron.
	pipeline.step(clamp=clamp)
	network._reset()