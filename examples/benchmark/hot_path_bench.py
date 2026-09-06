"""
Timing / peak-memory benchmark for the per-timestep hot paths (learning rules,
neuron updates). Prints one line per workload.

    python examples/benchmark/hot_path_bench.py            # CPU
    python examples/benchmark/hot_path_bench.py --gpu      # CUDA
"""

import argparse
import time

import torch

from bindsnet.encoding import poisson
from bindsnet.learning import MSTDP, PostPre
from bindsnet.models import DiehlAndCook2015
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection, LocalConnection2D

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", action="store_true")
parser.add_argument("--reps", type=int, default=3)
parser.add_argument("--time", type=int, default=250)
args = parser.parse_args()
dev = "cuda" if args.gpu and torch.cuda.is_available() else "cpu"
T = args.time


def timeit(net, inputs, **kw):
    if dev != "cpu":
        net.to(dev)
        inputs = {k: v.to(dev) for k, v in inputs.items()}
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    net.run(inputs=inputs, time=T, **kw)  # warm-up
    net.reset_state_variables()
    if dev != "cpu":
        torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(args.reps):
        net.run(inputs=inputs, time=T, **kw)
        net.reset_state_variables()
    if dev != "cpu":
        torch.cuda.synchronize()
    wall = (time.perf_counter() - t) / args.reps
    mem = torch.cuda.max_memory_allocated() / 2**20 if dev != "cpu" else float("nan")
    return wall, mem


rows = []

for b in (1, 16):
    torch.manual_seed(0)
    net = DiehlAndCook2015(
        n_inpt=784, n_neurons=400, batch_size=b, inpt_shape=(1, 28, 28), device=dev
    )
    inp = poisson(torch.rand(b, 1, 28, 28) * 128, time=T)
    rows.append((f"DiehlAndCook2015 784->400 batch={b}",) + timeit(net, {"X": inp}))

for b in (1, 16):
    torch.manual_seed(0)
    net = Network(dt=1.0, batch_size=b)
    net.add_layer(Input(n=784, traces=True), "in")
    net.add_layer(LIFNodes(n=1000, traces=True), "out")
    net.add_connection(
        Connection(
            net.layers["in"],
            net.layers["out"],
            nu=(1e-4, 1e-2),
            update_rule=PostPre,
            wmin=0,
            wmax=1,
            norm=78.4,
        ),
        "in",
        "out",
    )
    inp = poisson(torch.rand(b, 784) * 128, time=T)
    rows.append((f"Connection+PostPre 784->1000 batch={b}",) + timeit(net, {"in": inp}))

torch.manual_seed(0)
net = Network(dt=1.0)
net.add_layer(Input(shape=[1, 28, 28], traces=True), "in")
net.add_layer(LIFNodes(shape=[16, 6, 6], traces=True), "out")
net.add_connection(
    LocalConnection2D(
        net.layers["in"],
        net.layers["out"],
        kernel_size=8,
        stride=4,
        n_filters=16,
        nu=(1e-4, 1e-2),
        update_rule=PostPre,
        wmin=0,
        wmax=1,
    ),
    "in",
    "out",
)
inp = poisson(torch.rand(1, 1, 28, 28) * 128, time=T)
rows.append(("LocalConnection2D+PostPre 28x28 k8 s4 f16",) + timeit(net, {"in": inp}))

torch.manual_seed(0)
net = Network(dt=1.0)
net.add_layer(Input(n=784), "in")
net.add_layer(LIFNodes(n=500), "out")
net.add_connection(
    Connection(
        net.layers["in"], net.layers["out"], nu=1e-3, update_rule=MSTDP, wmin=0, wmax=1
    ),
    "in",
    "out",
)
inp = poisson(torch.rand(1, 784) * 128, time=T)
rows.append(("Connection+MSTDP 784->500",) + timeit(net, {"in": inp}, reward=1.0))

print(f"device={dev}  timesteps={T}  reps={args.reps}")
for name, wall, mem in rows:
    print(f"{name:<45s} {wall:8.3f} s/run   peak {mem:8.1f} MiB")
