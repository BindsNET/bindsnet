from bindsnet.datasets import MNIST
from bindsnet.encoding import poisson
from bindsnet.pipeline import Pipeline
from bindsnet.models import DiehlAndCook2015
from bindsnet.environment import DatasetEnvironment

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015(n_inpt=784, n_neurons=400, exc=22.5,
                           inh=17.5, dt=1.0, norm=78.4)

# Specify dataset wrapper environment.
environment = DatasetEnvironment(dataset=MNIST(path='../../data/MNIST'),
                                 train=True, download=True, intensity=0.25)

# Build pipeline from components.
pipeline = Pipeline(network=network, environment=environment,
                    encoding=poisson, time=350, plot_interval=1)

# Train the network.
for i in range(60000):    
    pipeline.step()
    network.reset_()
