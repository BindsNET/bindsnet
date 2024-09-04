import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd
from experiments import ROOT_DIR

benchmark_path = os.path.join(ROOT_DIR, "benchmark")
figure_path = os.path.join(ROOT_DIR, "figures")

if not os.path.isdir(benchmark_path):
    os.makedirs(benchmark_path)


def main(start=100, stop=1000, step=100, time=1000, interval=100, plot=False):
    name = f"benchmark_{start}_{stop}_{step}_{time}"
    f = os.path.join(benchmark_path, name + ".csv")
    df = pd.read_csv(f, index_col=0)

    plt.plot(df["BindsNET_cpu"], label="BindsNET (CPU)", linestyle="-", color="b")
    plt.plot(df["BindsNET_gpu"], label="BindsNET (GPU)", linestyle="-", color="g")
    plt.plot(df["BRIAN2"], label="BRIAN2", linestyle="--", color="r")
    plt.plot(df["BRIAN2GENN"], label="brian2genn", linestyle="--", color="c")
    plt.plot(df["BRIAN2GENN comp."], label="brian2genn comp.", linestyle=":", color="c")
    plt.plot(df["PyNEST"], label="PyNEST", linestyle="--", color="y")
    plt.plot(df["ANNarchy_cpu"], label="ANNarchy (CPU)", linestyle="--", color="m")
    plt.plot(df["ANNarchy_gpu"], label="ANNarchy (GPU)", linestyle="--", color="k")
    plt.plot(
        df["ANNarchy_gpu comp."], label="ANNarchy (GPU) comp.", linestyle=":", color="k"
    )

    # for c in df.columns:
    #     if 'BindsNET' in c:
    #         plt.plot(df[c], label=c, linestyle='-')
    #     else:
    #         plt.plot(df[c], label=c, linestyle='--')

    plt.title("Benchmark comparison of SNN simulation libraries")
    plt.xticks(range(0, stop + interval, interval))
    plt.xlabel("Number of input / output neurons")
    plt.ylabel("Simulation time (seconds)")
    plt.legend(loc=1, prop={"size": 5})
    plt.yscale("log")

    plt.savefig(os.path.join(figure_path, name + ".png"))

    if plot:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, default=100)
    parser.add_argument("--stop", type=int, default=1000)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--time", type=int, default=1000)
    parser.add_argument("--interval", type=int, default=1000)
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
