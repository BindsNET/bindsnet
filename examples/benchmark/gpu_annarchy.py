from __future__ import print_function

import argparse
import os
from time import time as t

import ANNarchy
import numpy as np
import pandas as pd

plots_path = os.path.join("..", "..", "figures")
benchmark_path = os.path.join("..", "..", "benchmark")
if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)


def ANNarchy_gpu(n_neurons, time):
    t0 = t()

    ANNarchy.setup(paradigm="cuda", dt=1.0)
    ANNarchy.clear()

    IF = ANNarchy.Neuron(
        parameters="""
            tau_m = 10.0
            tau_e = 5.0
            vt = -54.0
            vr = -60.0
            El = -74.0
            Ee = 0.0
        """,
        equations="""
            tau_m * dv/dt = El - v + g_exc *  (Ee - vr) : init = -60.0
            tau_e * dg_exc/dt = - g_exc
        """,
        spike="""
            v > vt
        """,
        reset="""
            v = vr
        """,
    )

    Input = ANNarchy.PoissonPopulation(name="Input", geometry=n_neurons, rates=50.0)
    Output = ANNarchy.Population(name="Output", geometry=n_neurons, neuron=IF)
    proj = ANNarchy.Projection(pre=Input, post=Output, target="exc", synapse=None)
    proj.connect_all_to_all(weights=ANNarchy.Uniform(0.0, 1.0))

    ANNarchy.compile()

    t1 = t()

    ANNarchy.simulate(duration=time)

    return t() - t0, t() - t1


def main(start=100, stop=1000, step=100, time=1000, interval=100, plot=False):
    times = {"ANNarchy_gpu": [], "ANNarchy_gpu (w/ comp.)": []}

    f = os.path.join(
        benchmark_path, "benchmark_{start}_{stop}_{step}_{time}.csv".format(**locals())
    )
    if not os.path.isfile(f):
        raise Exception("{0} not found.".format(f))

    for n_neurons in range(start, stop + step, step):
        print("\nRunning benchmark with {0} neurons.".format(n_neurons))
        for framework in times.keys():
            if "comp" in framework:
                continue

            print("- {0}:".format(framework), end=" ")

            fn = globals()[framework]
            total, sim = fn(n_neurons=n_neurons, time=time)
            times[framework].append(sim)
            times[framework + " (w/ comp.)"].append(total)

            print("(total, sim: {0:.4f}, {1:.4f})".format(total, sim))

    df = pd.read_csv(f, index_col=0)

    for framework in times.keys():
        print(pd.Series(times[framework]))
        df[framework] = times[framework]

    print()
    print(df)
    print()

    df.to_csv(f)


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

    print(args)

    main(
        start=args.start,
        stop=args.stop,
        step=args.step,
        time=args.time,
        interval=args.interval,
        plot=args.plot,
    )
