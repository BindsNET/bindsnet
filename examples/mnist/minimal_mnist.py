from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from bindsnet.pipeline import TorchVisionDatasetPipeline
from bindsnet.models import DiehlAndCook2015
from bindsnet.analysis.pipeline_analysis import TensorboardAnalyzer
from torchvision import transforms

# Build Diehl & Cook 2015 network.
network = DiehlAndCook2015(
    n_inpt=784,
    n_neurons=400,
    exc=22.5,
    inh=17.5,
    dt=1.0,
    norm=78.4,
    inpt_shape=(1, 28, 28),
)

# Specify dataset
mnist = MNIST(
    PoissonEncoder(time=50, dt=1.0),
    None,
    root="../../data/MNIST",
    download=True,
    train=True,
    transform=transforms.Compose(
        [transforms.ToTensor(), transforms.Lambda(lambda x: x * 128.0)]
    ),
)

# Plotting configuration.
plot_config = {
    "data_step": 1,
    "data_length": 10,
    "reward_eps": 1,
    "reward_window": 10,
    "volts_type": "line",
}

# Build pipeline from components.
pipeline = TorchVisionDatasetPipeline(
    network,
    mnist,
    TensorboardAnalyzer("logs/minimal_mnist"),
    plot_config=plot_config,
    batch_size=64,
)

pipeline.train()
