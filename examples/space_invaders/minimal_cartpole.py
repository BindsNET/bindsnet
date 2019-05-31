from functools import partial
from itertools import product
from typing import List

import torch

from bindsnet.encoding import Encoder
from bindsnet.environment import GymEnvironment
from bindsnet.learning import Rmax
from bindsnet.learning.reward import MovingAvgRPE
from bindsnet.network import Network
from bindsnet.network.nodes import Input, SRM0Nodes
from bindsnet.network.topology import Connection
from bindsnet.pipeline import EnvironmentPipeline


def cartesian_prod(*tensors: torch.Tensor, dim: int = 0, newdim: int = 0) -> torch.Tensor:
    total_dims = len(tensors[0].shape)

    if newdim < 0:
        newdim += total_dims
    if dim < 0:
        dim += total_dims

    tensors = [torch.unbind(t, dim) for t in tensors]
    tensors = [torch.stack(t, (newdim - 1) if newdim > dim else newdim) for t in product(*tensors)]

    return torch.stack(tensors, (dim + 1) if newdim <= dim else dim)


def place_cell_centers(state_bounds: List[List[float]], n: int) -> torch.Tensor:
    centers = torch.zeros((len(state_bounds), n))

    for i, b in enumerate(state_bounds):
        centers[i, :] = torch.linspace(*b, n)

    unpacked = torch.unbind(centers)

    return cartesian_prod(*unpacked)


def place_cells(centers: torch.Tensor, width: torch.Tensor, state: torch.Tensor, time: int,
                dt: float = 1.0, max_rate: float = 100.0, correction: float = 20.0, **kwargs) -> torch.Tensor:
    # Number of steps.
    n = centers.shape[1]
    time = int(time / dt)
    firing_rates = torch.ones((time, n))

    # Compute firing rates.
    distance = (centers - state[:, None]) ** 2
    firing_rates *= max_rate * torch.exp(-(distance / (correction * width[:, None] ** 2)).sum(0))

    # Convert to spikes.
    spikes = torch.rand(time, n) < (firing_rates * (dt / 1000))  # We need dt in seconds.

    return spikes


def select_binary(pipeline: EnvironmentPipeline, **kwargs) -> int:
    # We need to know which layer is output.
    try:
        output = kwargs['output']
    except KeyError:
        raise KeyError("select_single() requires an 'output' layer argument.")

    # Get spikes.
    spikes = pipeline.network.layers[output].s

    # Only one neuron!
    assert pipeline.network.layers[output].n == 1, "Only one output neuron is allowed."

    if spikes.sum() == 1:
        action = 1
    else:
        action = 0

    return action


# Get info about environment from https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py.
# Useful bounds based on termination requirements/common sense.
n_obs = 4
n_action = 2
obs_bounds = [[-2.4, 2.4], [-10.0, 10.0], [-0.4189, 0.4189], [-20.0, 20.0]]  # X, V_cart, theta, V_pole.

# Create partial function.
n_place = 11
centers = place_cell_centers(obs_bounds, n_place)  # Centers of place cells.
width = torch.tensor([0.48, 2.0, 0.083, 4.0])  # Width of place cells per state dimension.
place_cells_partial = partial(place_cells, centers, width, max_rate=100, correction=2)


class PlaceCellEncoder(Encoder):
    def __init__(self, time: int, dt: float = 1.0, **kwargs):
        super().__init__(time, dt=dt, **kwargs)
        self.enc = place_cells_partial


# Build network.
network = Network(dt=10.0, reward_fn=MovingAvgRPE)

# Get environment.
environment = GymEnvironment(
    "CartPole-v1",
    PlaceCellEncoder(time=int(network.dt), dt=network.dt),
)
environment.reset()

# Layers of neurons.
inpt = Input(n=n_place * n_obs, traces=True, traces_additive=True)
outpt = SRM0Nodes(n=1, traces=True, reset=-70.0, rest=-70.0, thresh=-60.0, d_thresh=1.0, tc_decay=20.0, tc_trace=20.0,
                  traces_additive=True, refrac=0)

# Connections between layers.
conn = Connection(
    source=inpt,
    target=outpt,
    wmin=-1.0,
    wmax=1.0,
    update_rule=Rmax,
    nu=5e-5,
)

# Add all layers and connections to the network.
network.add_layer(inpt, name="X")
network.add_layer(outpt, name="Y")
network.add_connection(conn, source="X", target="Y")

# Build pipeline from specified components.
pipeline = EnvironmentPipeline(
    network,
    environment,
    action_function=select_binary,
    output="Y",
    num_episodes=10,
    plot_interval=1,
    plot_length=10,
    plot_type="line",
    reward_window=100,
)

# Run environment simulation and network training.
pipeline.train(ema_window=10.0)
