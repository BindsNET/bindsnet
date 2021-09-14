import argparse
import os
from time import time as t

import brian2genn
import matplotlib.pyplot as plt
import nengo
import numpy as np
import pandas as pd
import torch
from brian2 import *
from experiments import ROOT_DIR
from experiments.benchmark import plot_benchmark
from nest import *

from bindsnet.encoding import poisson
from bindsnet.network import Network
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

plots_path = os.path.join(ROOT_DIR, "figures")
benchmark_path = os.path.join(ROOT_DIR, "benchmark")
if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)

# "Warm up" the GPU.
torch.set_default_tensor_type("torch.cuda.FloatTensor")
x = torch.rand(1000)
del x

# BRIAN2 clock
defaultclock = 1.0 * ms


def BindsNET_cpu(n_neurons, time):
    t0 = t()

    torch.set_default_tensor_type("torch.FloatTensor")

    t1 = t()

    network = Network()
    network.add_layer(Input(n=n_neurons), name="X")
    network.add_layer(LIFNodes(n=n_neurons), name="Y")
    network.add_connection(
        Connection(source=network.layers["X"], target=network.layers["Y"]),
        source="X",
        target="Y",
    )

    data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}
    network.run(inputs=data, time=time)

    return t() - t0, t() - t1


def BindsNET_gpu(n_neurons, time):
    if torch.cuda.is_available():
        t0 = t()

        torch.set_default_tensor_type("torch.cuda.FloatTensor")

        t1 = t()

        network = Network()
        network.add_layer(Input(n=n_neurons), name="X")
        network.add_layer(LIFNodes(n=n_neurons), name="Y")
        network.add_connection(
            Connection(source=network.layers["X"], target=network.layers["Y"]),
            source="X",
            target="Y",
        )

        data = {"X": poisson(datum=torch.rand(n_neurons), time=time)}
        network.run(inputs=data, time=time)

        return t() - t0, t() - t1


def BRIAN2(n_neurons, time):
    t0 = t()

    set_device("runtime", build_on_run=False)
    defaultclock = 1.0 * ms
    device.build()

    t1 = t()

    eqs_neurons = """
        dv/dt = (ge * (-60 * mV) + (-74 * mV) - v) / (10 * ms) : volt
        dge/dt = -ge / (5 * ms) : 1
    """

    input = PoissonGroup(n_neurons, rates=15 * Hz)
    neurons = NeuronGroup(
        n_neurons,
        eqs_neurons,
        threshold="v > (-54 * mV)",
        reset="v = -60 * mV",
        method="exact",
    )
    S = Synapses(input, neurons, """w: 1""")
    S.connect()
    S.w = "rand() * 0.01"

    run(time * ms)

    return t() - t0, t() - t1


def BRIAN2GENN(n_neurons, time):
    t0 = t()

    set_device("genn", build_on_run=False)
    defaultclock = 1.0 * ms
    device.build()

    t1 = t()

    eqs_neurons = """
        dv/dt = (ge * (-60 * mV) + (-74 * mV) - v) / (10 * ms) : volt
        dge/dt = -ge / (5 * ms) : 1
    """

    input = PoissonGroup(n_neurons, rates=15 * Hz)
    neurons = NeuronGroup(
        n_neurons,
        eqs_neurons,
        threshold="v > (-54 * mV)",
        reset="v = -60 * mV",
        method="exact",
    )
    S = Synapses(input, neurons, """w: 1""")
    S.connect()
    S.w = "rand() * 0.01"

    run(time * ms)

    device.reinit()
    device.activate()

    return t() - t0, t() - t1


def PyNEST(n_neurons, time):
    t0 = t()

    ResetKernel()
    SetKernelStatus({"local_num_threads": 8, "resolution": 1.0})

    t1 = t()

    r_ex = 60.0  # [Hz] rate of exc. neurons

    neuron = Create("iaf_psc_alpha", n_neurons)
    noise = Create("poisson_generator", n_neurons)

    SetStatus(noise, [{"rate": r_ex}])
    Connect(noise, neuron)

    Simulate(time)

    return t() - t0, t() - t1


def Nengo(n_neurons, time):
    t0 = t()
    t1 = t()

    model = nengo.Network()
    with model:
        X = nengo.Ensemble(n_neurons, dimensions=1, neuron_type=nengo.LIF())
        Y = nengo.Ensemble(n_neurons, dimensions=2, neuron_type=nengo.LIF())
        nengo.Connection(X, Y, transform=np.random.rand(n_neurons, n_neurons))

    with nengo.Simulator(model) as sim:
        sim.run(time / 1000)

    return t() - t0, t() - t1


def main(start=100, stop=1000, step=100, time=1000, interval=100, plot=False):
    f = os.path.join(benchmark_path, f"benchmark_{start}_{stop}_{step}_{time}.csv")
    if os.path.isfile(f):
        os.remove(f)

    times = {
        "BindsNET_cpu": [],
        "BindsNET_gpu": [],
        "BRIAN2": [],
        "BRIAN2GENN": [],
        "BRIAN2GENN comp.": [],
        "PyNEST": [],  # , 'Nengo': []
    }

    for n_neurons in range(start, stop + step, step):
        print(f"\nRunning benchmark with {n_neurons} neurons.")
        for framework in times.keys():
            if n_neurons > 2500 and framework == "PyNEST":
                times[framework].append(np.nan)
                continue

            if framework == "BRIAN2GENN comp.":
                continue

            print(f"- {framework}:", end=" ")

            fn = globals()[framework]
            total, sim = fn(n_neurons=n_neurons, time=time)
            times[framework].append(sim)

            if framework == "BRIAN2GENN":
                times["BRIAN2GENN comp."].append(total - sim)

            print(f"(total: {total:.4f}; sim: {sim:.4f})")

    print(times)

    df = pd.DataFrame.from_dict(times)
    df.index = list(range(start, stop + step, step))

    print()
    print(df)
    print()

    df.to_csv(f)

    plot_benchmark.main(
        start=start, stop=stop, step=step, time=time, interval=interval, plot=plot
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=100)
    parser.add_argument("--plot", dest="plot", action="store_true")
    parser.set_defaults(plot=False)
    args = parser.parse_args()

    main(
        start=args.start,
        stop=args.stop,
        step=args.step,
        time=args.time,
        interval=args.interval,
        plot=args.plot,
    )
