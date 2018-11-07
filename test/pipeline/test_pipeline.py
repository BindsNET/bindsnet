import os

from bindsnet import ROOT_DIR
from bindsnet.datasets import MNIST
from bindsnet.pipeline import Pipeline
from bindsnet.models import DiehlAndCook2015
from bindsnet.encoding import poisson
from bindsnet.environment import DatasetEnvironment


class TestPipeline:

    def test_mnist_pipeline(self):
        print(os.path.join(ROOT_DIR, 'data', 'MNIST'))

        network = DiehlAndCook2015(n_inpt=784, n_neurons=400, exc=22.5, inh=17.5, dt=1.0, norm=78.4)
        environment = DatasetEnvironment(dataset=MNIST(path=os.path.join(ROOT_DIR, 'data', 'MNIST'), download=True), train=True, intensity=0.25)
        pipeline = Pipeline(network=network, environment=environment, encoding=poisson, time=350)

        assert pipeline.network == network
        assert pipeline.env == environment
        assert pipeline.encoding == poisson
        assert pipeline.time == 350
        assert pipeline.history_length is None
