"""BindsNET-only version of the scaling benchmark described in the README.

n Poisson input neurons with rates drawn from U(0, 100) Hz, connected all-to-all
to n LIF neurons with weights drawn from N(0, 1), simulated for 1000 ms at
dt = 1.0 ms, for n = 250 to 10,000 in steps of 250. Needs no other simulator.

    python examples/benchmark/bindsnet_scaling.py --csv bindsnet_scaling.csv

Prints, per n and device, the wall time of ``network.run`` alone ("sim") and of
building the network plus encoding plus running ("total").
"""

import argparse
from time import perf_counter

import torch

from bindsnet.encoding import poisson
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection


def run_once(n, time, dt, device):
    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    sync()
    t0 = perf_counter()
    network = Network(dt=dt)
    network.add_layer(Input(n=n), name="X")
    network.add_layer(LIFNodes(n=n), name="Y")
    network.add_connection(
        Connection(network.layers["X"], network.layers["Y"], w=torch.randn(n, n)),
        source="X",
        target="Y",
    )
    network.to(device)
    rates = torch.rand(n) * 100.0  # Hz
    inputs = {"X": poisson(datum=rates, time=time, dt=dt, device=device)}
    sync()
    t1 = perf_counter()
    network.run(inputs=inputs, time=time)
    sync()
    t2 = perf_counter()
    return t2 - t0, t2 - t1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=250)
    parser.add_argument("--stop", type=int, default=10000)
    parser.add_argument("--step", type=int, default=250)
    parser.add_argument("--time", type=int, default=1000, help="ms")
    parser.add_argument("--dt", type=float, default=1.0, help="ms")
    parser.add_argument("--devices", nargs="+", default=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    devices = [d for d in args.devices if d != "cuda" or torch.cuda.is_available()]

    # Warm up each device so the first size does not pay one-off start-up costs.
    for device in devices:
        run_once(args.start, 10, args.dt, device)

    rows = []
    print(f"{'n':>6} {'device':>6} {'sim_s':>9} {'total_s':>9}")
    for n in range(args.start, args.stop + 1, args.step):
        for device in devices:
            total, sim = run_once(n, args.time, args.dt, device)
            rows.append((n, device, sim, total))
            print(f"{n:>6} {device:>6} {sim:>9.4f} {total:>9.4f}", flush=True)

    if args.csv:
        with open(args.csv, "w") as f:
            f.write("n,device,sim_s,total_s\n")
            for row in rows:
                f.write("%d,%s,%.6f,%.6f\n" % row)


if __name__ == "__main__":
    main()
