"""Plots simulation time against n from the CSV that benchmark.py writes.

python examples/benchmark/plot_benchmark.py benchmark.csv
"""

import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", type=str)
    parser.add_argument("--plot", dest="plot", action="store_true")
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    for name, group in df.groupby("simulator", sort=False):
        style = "-" if name.startswith("bindsnet") else "--"
        plt.plot(group["n"], group["sim_s"], label=name, linestyle=style)

    plt.title("Benchmark comparison of SNN simulation libraries")
    plt.xlabel("Number of input / output neurons")
    plt.ylabel("Simulation time (seconds)")
    plt.legend(loc="upper left", prop={"size": 7})
    plt.yscale("log")
    plt.savefig(os.path.splitext(args.csv)[0] + ".png")

    if args.plot:
        plt.show()


if __name__ == "__main__":
    main()
