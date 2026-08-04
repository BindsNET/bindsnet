"""
ExampleNetwork -- a large, sparse, recurrent ``MulticompartmentConnection`` (MCC)
stress workload.

Topology: ``Input(I) -> EXC_LIF <-> INH_LIF``. Four single-``Weight`` MCC.
The ``I -> EXC`` connection also carries an ``MSTDP`` learning rule.

Run it directly to stress the simulator:

    python examples/stress_test/example_network.py --device cuda --exc 20000 --time 50
    python examples/stress_test/example_network.py --device cpu  --exc 2000  --time 20
"""

import argparse
import os
import sys
import time

# Make ``bindsnet`` importable when this file is run as a standalone script.
sys.path.insert(
    0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import torch

from bindsnet.learning.MCC_learning import MSTDP
from bindsnet.network.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import MulticompartmentConnection
from bindsnet.network.topology_features import Weight


class ExampleNetwork(Network):
    def __init__(
        self,
        device="cpu",
        in_size=100,
        exc_size=20_000,
        inh_size=2_000,
        batch_size=1,
        i_to_exc_connectivity=0.15,
        i_to_inh_connectivity=0.05,
        inh_to_exc_connectivity=0.05,
        exc_to_inh_connectivity=0.05,
    ):
        super().__init__()
        self.device = device
        self.in_size = in_size
        self.exc_size = exc_size
        self.inh_size = inh_size
        self.batch_size = batch_size
        self.i_to_exc_connectivity = i_to_exc_connectivity
        self.i_to_inh_connectivity = i_to_inh_connectivity
        self.inh_to_exc_connectivity = inh_to_exc_connectivity
        self.exc_to_inh_connectivity = exc_to_inh_connectivity
        self.build()

    def _sparse_weight(self, rows, cols, connectivity, sign=1.0):
        w = sign * torch.rand(rows, cols, device=self.device)
        keep = torch.rand(rows, cols, device=self.device) > (1 - connectivity)
        return w * keep

    def build(self):
        device = self.device
        self.add_layer(layer=Input(self.in_size), name="I")
        self.add_layer(layer=LIFNodes(self.exc_size), name="EXC_LIF")
        self.add_layer(layer=LIFNodes(self.inh_size), name="INH_LIF")
        self.add_connection(
            connection=MulticompartmentConnection(
                source=self.layers["I"],
                target=self.layers["EXC_LIF"],
                device=device,
                pipeline=[
                    Weight(
                        name="I_to_EXC_weight",
                        value=self._sparse_weight(
                            self.in_size, self.exc_size, self.i_to_exc_connectivity
                        ),
                        learning_rule=MSTDP,
                        range=(0, 1),
                    )
                ],
            ),
            source="I",
            target="EXC_LIF",
        )
        self.add_connection(
            connection=MulticompartmentConnection(
                source=self.layers["I"],
                target=self.layers["INH_LIF"],
                device=device,
                pipeline=[
                    Weight(
                        name="I_to_INH_weight",
                        value=self._sparse_weight(
                            self.in_size, self.inh_size, self.i_to_inh_connectivity
                        ),
                    )
                ],
            ),
            source="I",
            target="INH_LIF",
        )
        self.add_connection(
            connection=MulticompartmentConnection(
                source=self.layers["INH_LIF"],
                target=self.layers["EXC_LIF"],
                device=device,
                pipeline=[
                    Weight(
                        name="INH_to_EXC_weight",
                        value=self._sparse_weight(
                            self.inh_size,
                            self.exc_size,
                            self.inh_to_exc_connectivity,
                            sign=-1.0,
                        ),
                    )
                ],
            ),
            source="INH_LIF",
            target="EXC_LIF",
        )
        self.add_connection(
            connection=MulticompartmentConnection(
                source=self.layers["EXC_LIF"],
                target=self.layers["INH_LIF"],
                device=device,
                pipeline=[
                    Weight(
                        name="EXC_to_INH_weight",
                        value=self._sparse_weight(
                            self.exc_size, self.inh_size, self.exc_to_inh_connectivity
                        ),
                    )
                ],
            ),
            source="EXC_LIF",
            target="INH_LIF",
        )
        self.to(device)

    def make_input(self, runtime):
        # Poisson-ish random spike train into the input layer.
        return {
            "I": torch.rand(runtime, self.batch_size, self.in_size, device=self.device)
            > 0.90
        }


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Stress-test the ExampleNetwork.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--in-size", type=int, default=100)
    p.add_argument("--exc", type=int, default=20_000)
    p.add_argument("--inh", type=int, default=2_000)
    p.add_argument("--time", type=int, default=50)
    args = p.parse_args()

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("[note] CUDA unavailable; falling back to CPU.")
        device = "cpu"

    net = ExampleNetwork(
        device=device, in_size=args.in_size, exc_size=args.exc, inh_size=args.inh
    )
    net.train(False)  # forward-only stress (no learning)
    inputs = net.make_input(args.time)

    net.run(inputs=inputs, time=args.time)  # warmup
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    net.reset_state_variables()
    t0 = time.perf_counter()
    net.run(inputs=inputs, time=args.time)
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) * 1e3
    print(
        f"ExampleNetwork [{device}] in={args.in_size} exc={args.exc} inh={args.inh} "
        f"time={args.time}: {ms:.1f} ms total ({ms / args.time:.3f} ms/step)"
    )
