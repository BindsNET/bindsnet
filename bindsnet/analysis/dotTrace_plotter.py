import glob
import sys

import matplotlib.pyplot as plt
import numpy as np

# Define grid dimensions globally
ROWS = 28
COLS = 28


def plotGrids(gridData):
    if gridData.shape[0] % ROWS != 0 or gridData.shape[1] != COLS:
        raise ("Incompatible grid dimensionality: check data and assumed dimensions.")

    grids = gridData.shape[0] // ROWS

    print("Reshaping into", grids, "grids of shape (", ROWS, ",", COLS, ")")
    gridData = gridData.reshape((grids, ROWS, COLS))

    plotAnotherRange = True

    while plotAnotherRange:
        start = -1
        end = 1
        print("Select the range of iterations to generate grid plots from.")
        print("0 means plot all iterations.")
        while (start < 0 or grids - 1 < start) or (end < 1 or grids < end):
            start = int(input("Start: "))

            # If start is set to zero, plot everything.
            if start == 0:
                continue

            end = int(input("End: "))

        if start == 0:
            print("\nPlotting whole shebang!")
        else:
            print("\nPlotting range from iteration", start, "to", end)

        # Plotting time!
        plt.figure()
        plt.ion()
        plt.imshow(gridData[start], cmap="hot", interpolation="nearest")
        plt.colorbar()
        plt.pause(0.001)  # Pause so that that GUI can do its thing.
        for g in gridData[start + 1 : end]:
            plt.imshow(g, cmap="hot", interpolation="nearest")
            plt.pause(0.001)  # Pause so that that GUI can do its thing.

        plotAnotherRange = str.lower(input("Plot another range? (y/n): ")) == "y"


def plotRewards(rewData, fname):
    cumRewards = np.cumsum(rewData)
    tsteps = np.array(range(len(cumRewards)))

    # Plotting time!
    plt.figure()
    plt.plot(tsteps, cumRewards)
    plt.xlabel("Timesteps")
    plt.ylabel("Cumulative Reward")
    plt.title("Cumulative Reward by Iteration")
    plt.savefig(fname[0:-4] + ".png", dpi=200)
    plt.pause(0.001)  # Pause so that that GUI can do its thing.


def plotPerformance(perfData, fname):

    # Set bins to a tenth of the episodes, rounded up.
    binIdx = np.array(range(len(perfData))) // 10
    bins = np.bincount(binIdx, perfData).astype("uint32")

    # Plotting time!
    plt.figure()
    plt.bar(np.unique(binIdx), bins, color="seagreen")
    plt.xlabel("Episode Bins")
    plt.ylabel("Number of Intercepts")
    plt.title("Interception Performance Across Episodes")
    plt.savefig(fname[0:-4] + ".png", dpi=200)
    plt.pause(0.001)  # Pause so that that GUI can do its thing.


def main():
    """
    File types:

    0) grid         - the 2D matrix observation
    1) reward       - list of rewards per iteration
    2) performance  - list of performance values
    """
    fileType = 0  # default to grid

    # By default, we'll search the examples directory, but tweak as needed.
    files = glob.glob("../../examples/*/out/*csv")

    if len(files) == 0:
        print("Could not find any csv files. Exiting...")
        sys.exit()

    plotAnotherFile = True

    while plotAnotherFile:
        print("Select the file to generate grid plots from.")
        for i, f in enumerate(files):
            print(str(i), "-", f)

        # Select the intended file.
        sel = -1
        while sel < 0 or len(files) < sel:
            sel = int(input("\nFile selection: "))

        fileToPlot = files[sel]

        # Check file type
        if 0 < fileToPlot.find("grid"):
            print("\nFound 'grid' in name: assuming a grid file type.")
            fileType = 0
        elif 0 < fileToPlot.find("rew"):
            print("\nFound 'rew' in name: assuming a reward file type.")
            fileType = 1

        elif 0 < fileToPlot.find("perf"):
            print("\nFound 'perf' in name: assuming a performance file type.")
            fileType = 2
        else:
            print("\nUnknown file type. Which type are we plotting?")
            print("\n0) grid\n1) reward\n2) performance")
            fileType = -1
            while fileType < 0 or 2 < fileType:
                fileType = int(input("\nFile type: "))

        print("\nPlotting: ", fileToPlot)
        data = np.genfromtxt(fileToPlot, delimiter=",")

        # Plot by file type
        if fileType == 0:
            plotGrids(data)
        elif fileType == 1:
            plotRewards(data, fileToPlot)
        elif fileType == 2:
            plotPerformance(data, fileToPlot)
        else:
            print("ERROR: Unknown file type")

        plotAnotherFile = str.lower(input("Plot another file? (y/n): ")) == "y"


if __name__ == "__main__":
    main()
