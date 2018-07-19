from bindsnet.datasets    import CIFAR10
from bindsnet.encoding    import poisson
from bindsnet.pipeline    import Pipeline
from bindsnet.models      import DiehlAndCook2015
from bindsnet.environment import DatasetEnvironment

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015(n_inpt=32*32*3,
                           n_neurons=400,
                           exc=22.5,
                           inh=17.5,
                           dt=1.0,
                           norm=78.4)

# Specify dataset wrapper environment.
environment = DatasetEnvironment(dataset=CIFAR10(path='../../data/CIFAR10', download=True),
                                 train=True,
                                 intensity=0.25)

# Build pipeline from components.
pipeline = Pipeline(network=network,
                    environment=environment,
                    encoding=poisson,
                    time=350,
                    plot_interval=1)

# Train the network.
for i in range(50000):
    pipeline.step()
    network.reset_()
