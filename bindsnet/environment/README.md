## Dot Simulator

### Overview

This simulator lets us generate dots and make them move in a configurable 2D space, providing a visual to a neural network for training in experiments.

Specifically, this generates a grid for each timestep, where a specified number of points have values of 1 with fading tails ("decay"), designating the current positions and movements of their corresponding dots. All other points are set to 0. From timestep to timestep, the dots either remain where they are or move one space.

The 2D observation of the current state is provided every step, as well as the reward, completion flag, and sucessful interception flag. It may be helpful to scale the grid values when encoding them as spike trains.

The intended objective is to train a network to use its "network dot" to trace or intercept a moving "target" dot. But this simulator is designed to easily adapt to multiple kinds of experiments.


### Dot Movement

By default, there is a single "target" dot that moves in a random direction every timestep (or it can stay still, which can be disabled), and as it moves, it leaves a tunable "decay" in the form of a fading tail. The simulator supports four directions of movement by default (up/down/left/right) by default, as well as remaining still, but the diag parameters allows diagonal movement for more complexity. The rate of the target's randomized movement can also be modified (ie. random direction every timestep or only change direction so often).

The simulator supports multiple bounds-handling schemes. By default, dots will simply not move past the edges. Alternatively, the bound_hand parameter can be set to 'bounce', for a geometric reflection off the edges, or 'trans' which will have a mirrored result: a geometric translation to the opposite side of the grid.

To add further complexity, additional targets can be added as desired via the dots parameter, and the herrs parameter can be set to generate multiple "red herrings" as distraction dots. The speed of the dots' movements can also be set; it is 1 by default.

<p align="middle">
<img src="https://github.com/kamue1a/bindsnet/blob/dot_sim/docs/DotTraceSample.png" alt="DotTraceSample"  width="503" height="403">
</p>
>The grid visuals provided by the render function will double the value of the network dot; this is a visual aid only, invisible to the network.


### Reward Functions

This simulator supports multiple reward functions (aka. fitness functions):
- Euclidean		(fit_func='euc'):  the default option, this function computes the Euclidean (aka. Pythagorean) distance between the network dot and the target dot.
- Displacement	(fit_func='disp'): this option computes the x,y displacement of the network dot with respect to the target dot, returning an x,y tuple. Currently, BindsNET only supports single reward values. To use this one, either be creative or update the network code...
- Range Rings	(fit_func='rng'):  this option uses the Euclidean distance and groups it into range rings. The radial distance of the range rings can be set by the ring_size parameter.
- Directional	(fit_func='dir'):  the directional option checks to see if the network's decision moved its dot closer, laterally, or further away from the target dot's prior position (ie. before applying movement this timestep) and returns a +1, 0, or -1 accordingly.

Additionally, upon a successful intercept, the network will receive +10 if the bullseye parameter is active, and its dot will be teleported to another random location if the teleport parameter is active.

>In the event multiple target dots are generated, the fitness functions only compute rewards with respect to the first target dot.


### Additional Features

The environment can take a seed for random number generation in python, numpy, and Pytorch; otherwise, it will generate and save a new seed based on the current system time.

As this simulator was developed in Anaconda Spyder on Windows, it can be run from Windows or Linux. Since environments handle plotting differently, and experiments can sometimes be terminated prematurely, this environment supports the recording of grid observations in text files and post-op plotting. Live rendering can also be disabled via the mute parameter, and a text-based alternative using pandas dataframe formatting can be enabled via the pandas parameter.

Filenames and file paths can be specified for recording grid observations. By default, the filenames will be "grid" followed by "s#_$.csv" where # is the random seed used and $ is the current file number. addFileSuffix(suffix) adds the provided suffix (typically used for "train" or "test") to the filename, and changeFileSuffix(sFrom, sTo) will find sFrom in the filename and replace it with sTo.

To ensure that files do not become too large to either be saved or be practically useful, cycleOutFiles(newInt) can be used to cycle the current save file, incrementing the file number suffix, or resetting it if newInt is set to a positive number.

Post-op plotting is supported by dotTrace_plotter.py in the analysis directory. By default, this tool searches the examples directory for csvs in "out" directories, but that path can be easily changed. It supports plotting ranges of grid observations, reward plots, and performance plots. See below for an example of recording reward and performance data for plotting purposes.


### Example
See dot_tracing.py for an example in using the Dot Simulator for training an SNN in BindsNET.

dot_tracing trains a basic RNN network on the dot simulator and demonstrates how to record reward and performance data (if desired) and plot spiking activity via monitors.


