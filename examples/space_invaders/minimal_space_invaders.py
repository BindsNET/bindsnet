import torch

from bindsnet.network import Network
from bindsnet.pipeline import EnvironmentPipeline
from bindsnet.learning import MSTDPET
from bindsnet.encoding import BernoulliEncoder
from bindsnet.network.topology import Connection
from bindsnet.environment import GymEnvironment
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.pipeline.action import select_multinomial

# Build network.
network = Network(dt=1.0)

# Layers of neurons.
inpt = Input(n=78 * 84, shape=[1, 1, 78, 84], traces=True)
middle = LIFNodes(n=225, traces=True, thresh=-52.0 + torch.randn(225))
out = LIFNodes(n=60, refrac=0, traces=True, thresh=-40.0)

# Connections between layers.
inpt_middle = Connection(source=inpt, target=middle, wmax=1e-2)
middle_out = Connection(
    source=middle,
    target=out,
    wmax=0.5,
    update_rule=MSTDPET,
    nu=2e-2,
    norm=0.15 * middle.n,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="X")
network.add_layer(middle, name="Y")
network.add_layer(out, name="Z")
network.add_connection(inpt_middle, source="X", target="Y")
network.add_connection(middle_out, source="Y", target="Z")

# Load SpaceInvaders environment.
environment = GymEnvironment(
    "SpaceInvaders-v0",
    BernoulliEncoder(time=int(network.dt), dt=network.dt),
    history_length=2,
    delta=4,
)
environment.reset()

# Plotting configuration.
plot_config = {
    "data_step": 1,
    "data_length": 10,
    "reward_eps": 1,
    "reward_window": 10,
    "volts_type": "line",
}

# Build pipeline from specified components.
pipeline = EnvironmentPipeline(
    network,
    environment,
    time=network.dt,
    action_function=select_multinomial,
    output="Z",
    plot_config=plot_config,
    render_interval=5,
)

# Run environment simulation and network training.
pipeline.train()
