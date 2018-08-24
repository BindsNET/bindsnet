from torch import Tensor
from numpy.random import choice
from bindsnet.datasets import CIFAR10
from bindsnet.encoding import poisson
from bindsnet.pipeline import Pipeline
from bindsnet.models import DiehlAndCook2015
from bindsnet.environment import DatasetEnvironment

# Build network.
network = DiehlAndCook2015(n_inpt=32*32*3, n_neurons=100, dt=1.0, exc=22.5,
                           inh=17.5, nu_pre=0, nu_post=1e-2, norm=78.4)

# Specify dataset wrapper environment.
environment = DatasetEnvironment(dataset=CIFAR10(path='../../data/CIFAR10'),
                                 train=True)

# Build pipeline from components.
pipeline = Pipeline(network=network, environment=environment,
                    encoding=poisson, time=50, plot_interval=1)

# Train the network.
labels = environment.labels
for i in range(60000):
    # Choose an output neuron to clamp to spiking behavior.
    c = choice(10, size=1, replace=False)
    c = 10 * labels[i].long() + Tensor(c).long()
    clamp = torch.zeros(pipeline.time, network.n_neurons, dtype=torch.uint8)
    clamp[:, c] = 1
    clamp_v = torch.zeros(pipeline.time, network.n_neurons, dtype=torch.float)
    clamp_v[:,c] =  network.layers['Ae'].thresh + network.layers['Ae'].theta[c] + 10

    # Run a step of the pipeline with clamped neuron.
    pipeline.step(clamp={'Ae':clamp},clamp_v={'Ae':clamp_v})
    network.reset_()
