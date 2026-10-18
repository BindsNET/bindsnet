"""Scaling benchmark of Hazan et al. (2018), README "Benchmarking" section.

Every simulator gets the same network: n Poisson input neurons with rates drawn
from U(0, 100) Hz, connected all-to-all to n leaky integrate-and-fire neurons
with weights drawn from N(0, 1) (in mV, added to the membrane potential on each
input spike), simulated for 1000 ms at dt = 1.0 ms, for n = 250 to 10,000 in
steps of 250. The LIF parameters are BindsNET's ``LIFNodes`` defaults: rest and
reset -65 mV, threshold -52 mV, membrane time constant 100 ms, refractory 5 ms.

    python examples/benchmark/benchmark.py --simulators bindsnet_cpu bindsnet_gpu brian2 nest
    python examples/benchmark/plot_benchmark.py benchmark.csv

Each simulator is imported only when requested; one that is not installed is
skipped with a message. BindsNET needs nothing extra
(``bindsnet_scaling.py`` holds its timing code). Brian2, brian2genn, NEST and
ANNarchy versions of this network were written in 2026 to match the README
description and have not been run by the maintainers; the code that produced the
2018 figure (``docs/BindsNET benchmark.png``) is not in this repository, and the
earlier version of this file used different inputs for each simulator (see
REPRODUCING.md).
"""

import argparse
import os
import subprocess
import sys
import tempfile
from time import perf_counter

import numpy as np
from bindsnet_scaling import run_once as bindsnet_run_once

REST, RESET, THRESH, TAU_M, REFRAC = -65.0, -65.0, -52.0, 100.0, 5.0  # mV, ms


def bindsnet_cpu(n, time, dt, rng):
    return bindsnet_run_once(n, time, dt, "cpu")


def bindsnet_gpu(n, time, dt, rng):
    import torch

    if not torch.cuda.is_available():
        raise ImportError("CUDA is not available")
    return bindsnet_run_once(n, time, dt, "cuda")


def _brian2(n, time, dt, rng, genn):
    import brian2 as b2

    t0 = perf_counter()
    if genn:
        import brian2genn  # noqa: F401  (registers the "genn" device)

        build_dir = tempfile.TemporaryDirectory(prefix="brian2genn_")
        b2.set_device("genn", directory=build_dir.name)
    b2.defaultclock.dt = dt * b2.ms
    inputs = b2.PoissonGroup(n, rates=rng.uniform(0.0, 100.0, n) * b2.Hz)
    lif = b2.NeuronGroup(
        n,
        f"dv/dt = ({REST}*mV - v) / ({TAU_M}*ms) : volt (unless refractory)",
        threshold=f"v >= {THRESH}*mV",
        reset=f"v = {RESET}*mV",
        refractory=REFRAC * b2.ms,
        method="exact",
    )
    lif.v = REST * b2.mV
    syn = b2.Synapses(inputs, lif, "w : 1", on_pre="v_post += w*mV")
    syn.connect()
    syn.w = rng.standard_normal(n * n)
    net = b2.Network(inputs, lif, syn)
    t1 = perf_counter()
    net.run(time * b2.ms, namespace={})
    t2 = perf_counter()
    if genn:
        b2.device.reinit()
        b2.device.activate()
        b2.set_device("runtime")
        build_dir.cleanup()
    return t2 - t0, t2 - t1


def brian2(n, time, dt, rng):
    return _brian2(n, time, dt, rng, genn=False)


def brian2genn(n, time, dt, rng):
    return _brian2(n, time, dt, rng, genn=True)


def nest(n, time, dt, rng):
    import nest as nest_

    t0 = perf_counter()
    nest_.ResetKernel()
    nest_.set_verbosity("M_ERROR")
    nest_.resolution = dt
    nest_.local_num_threads = os.cpu_count()
    generators = nest_.Create(
        "poisson_generator", n, params={"rate": rng.uniform(0.0, 100.0, n).tolist()}
    )
    # Parrot neurons repeat each generator's train, so every LIF neuron sees the
    # same n input trains (a poisson_generator sends independent trains per target).
    inputs = nest_.Create("parrot_neuron", n)
    nest_.Connect(generators, inputs, "one_to_one")
    lif = nest_.Create(
        "iaf_psc_delta",
        n,
        params={
            "E_L": REST,
            "V_reset": RESET,
            "V_th": THRESH,
            "V_m": REST,
            "tau_m": TAU_M,
            "t_ref": REFRAC,
        },
    )
    nest_.Connect(
        inputs,
        lif,
        "all_to_all",
        syn_spec={"weight": nest_.random.normal(0.0, 1.0), "delay": dt},
    )
    t1 = perf_counter()
    nest_.Simulate(float(time))
    t2 = perf_counter()
    return t2 - t0, t2 - t1


def _annarchy(n, time, dt, rng, paradigm):
    import ANNarchy as ann

    t0 = perf_counter()
    ann.setup(paradigm=paradigm)
    net = ann.Network(dt=dt)
    neuron = ann.Neuron(
        parameters=f"tau_m = {TAU_M}\nEl = {REST}\nvt = {THRESH}\nvr = {RESET}",
        equations=f"tau_m * dv/dt = El - v : init = {REST}",
        spike="v >= vt",
        reset="v = vr",
        refractory=REFRAC,
    )
    inputs = net.create(ann.PoissonPopulation(n, rates=0.0))
    inputs.rates = rng.uniform(0.0, 100.0, n)
    lif = net.create(geometry=n, neuron=neuron)
    proj = net.connect(
        inputs, lif, target="exc", synapse=ann.Synapse(pre_spike="v_post += w")
    )
    proj.all_to_all(weights=ann.Normal(0.0, 1.0))
    with tempfile.TemporaryDirectory(prefix="annarchy_") as build_dir:
        net.compile(directory=build_dir, silent=True)
        t1 = perf_counter()
        net.simulate(float(time))
        t2 = perf_counter()
    return t2 - t0, t2 - t1


def annarchy_cpu(n, time, dt, rng):
    return _annarchy(n, time, dt, rng, "openmp")


def annarchy_gpu(n, time, dt, rng):
    return _annarchy(n, time, dt, rng, "cuda")


SIMULATORS = {
    f.__name__: f
    for f in (
        bindsnet_cpu,
        bindsnet_gpu,
        brian2,
        brian2genn,
        nest,
        annarchy_cpu,
        annarchy_gpu,
    )
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=250)
    parser.add_argument("--stop", type=int, default=10000)
    parser.add_argument("--step", type=int, default=250)
    parser.add_argument("--time", type=int, default=1000, help="ms")
    parser.add_argument("--dt", type=float, default=1.0, help="ms")
    parser.add_argument(
        "--simulators", nargs="+", default=list(SIMULATORS), choices=list(SIMULATORS)
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--csv", type=str, default="benchmark.csv")
    parser.add_argument("--append", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.append:
        with open(args.csv, "w") as f:
            f.write("n,simulator,sim_s,total_s\n")
        # One process per simulator: NEST 3.10 and ANNarchy 5.0 corrupt each
        # other's memory when loaded into the same process.
        common = [
            f"--{k}={getattr(args, k)}"
            for k in ("start", "stop", "step", "time", "dt", "seed", "csv")
        ]
        for name in args.simulators:
            subprocess.run(
                [sys.executable, __file__, *common, "--simulators", name, "--append"]
            )
        return

    import torch

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    available = []
    for name in args.simulators:
        try:
            SIMULATORS[name](args.start, 10, args.dt, rng)  # import check + warm-up
            available.append(name)
        except Exception as e:  # not installed, no CUDA, or failed to build
            print(f"skipping {name}: {e}")

    with open(args.csv, "a") as f:
        for n in range(args.start, args.stop + 1, args.step):
            for name in available:
                total, sim = SIMULATORS[name](n, args.time, args.dt, rng)
                f.write(f"{n},{name},{sim:.6f},{total:.6f}\n")
                f.flush()
                print(f"{n:>6} {name:>13} sim {sim:9.4f} s  total {total:9.4f} s")


if __name__ == "__main__":
    main()
