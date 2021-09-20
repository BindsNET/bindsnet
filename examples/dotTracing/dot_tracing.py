import argparse
import time

import numpy as np
import torch

from bindsnet.analysis.plotting import plot_spikes  # plot_performance

# from bindsnet.encoding import bernoulli
from bindsnet.encoding import poisson
from bindsnet.environment.dot_simulator import DotSimulator

# from bindsnet.pipeline import EnvironmentPipeline
# from bindsnet.learning import MSTDP
from bindsnet.learning import MSTDPET, PostPre
from bindsnet.network import Network

# from bindsnet.pipeline.action import select_softmax
# from bindsnet.network.nodes import AbstractInput
from bindsnet.network.monitors import Monitor
from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection

# Handle arguments for dot tracing params.
parser = argparse.ArgumentParser()
parser.add_argument("--steps", type=int, default=100)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--dim", type=int, default=28)
parser.add_argument("--granularity", type=int, default=100)
parser.add_argument("--neurons", type=int, default=100)
parser.add_argument("--dt", type=int, default=1.0)
parser.add_argument("--trn_eps", type=int, default=1000)
parser.add_argument("--tst_eps", type=int, default=100)
parser.add_argument("--decay", type=int, default=4)
parser.add_argument("--herrs", type=int, default=0)
parser.add_argument("--diag", type=bool, default=False)
parser.add_argument("--randr", type=float, default=0.15)
parser.add_argument("--boundh", type=str, default="bounce")
parser.add_argument("--fit_func", type=str, default="dir")
parser.add_argument("--allow_stay", type=bool, default=False)
parser.add_argument("--pandas", type=bool, default=False)
parser.add_argument("--mute", type=bool, default=False)
parser.add_argument("--write", type=bool, default=True)
parser.add_argument("--fcycle", type=int, default=100)
parser.add_argument("--gpu", type=bool, default=True)

args = parser.parse_args()

steps = args.steps  # timesteps in which the dot is moving
dim = args.dim  # 28x28 square
granularity = args.granularity  # granularity (or precision) of spike trains
neurons = args.neurons  # Number of neurons in hidden layer
dt = args.dt  # delta time of network
trn_eps = args.trn_eps  # training episodes
tst_eps = args.tst_eps  # testing episodes
decay = args.decay  # length of decaing tail behind a dot
herrs = args.herrs  # distraction dots
diag = args.diag  # allows diagonal movement
randr = args.randr  # determines rate of randomization of movement
boundh = args.boundh  # bounds handling mode
fit_func = args.fit_func  # fitness function
allow_stay = args.allow_stay  # disable option for targets to remain in place.
pandas = args.pandas  # true = pandas DF printout; false = heatmap
mute = args.mute  # prohibit graphical rendering
write = args.write  # write observed grids to file.
fcycle = args.fcycle  # number of episodes per save file
gpu = args.gpu  # Utilize cuda

if diag:
    moveChoices = 9
else:
    moveChoices = 5

""" Set some globals """
# Set processor type
if torch.cuda.is_available() and gpu:
    DEVICE = torch.device("cuda")
else:
    DEVICE = torch.device("cpu")

# Set neural network layer names
LAYER1 = "Input"
LAYER2 = "Hidden"
LAYER3 = "Output"

# file path for recording grid observations, rewards, and performance.
OUT_FILE_PATH = "out/"


def genFileName(ftype, suffix=""):
    """
    Generates output file names for rewards and performance
    """
    # Grab system time and trim off extra large parts of the number.
    sysTime = time.time()
    sysTime = int(1e10 * (sysTime - 1e6 * (sysTime // 1e6)))

    # Create filename if one isn't provided.
    return OUT_FILE_PATH + ftype + "_s" + str(sysTime) + "_" + suffix + ".csv"


def runSimulator(net, env, spikes, episodes, gran=100, rfname="", pfname=""):

    steps = env.timesteps
    dt = net.dt
    spike_ims, spike_axes = None, None

    # For each episode...
    for ep in range(episodes):
        # Reset variables for new episode.
        total_reward = 0
        rewards = np.zeros(steps)
        intercepts = 0
        step = 0
        net.reset_state_variables()
        env.reset()
        done = False
        env.render()
        clock = time.time()

        # Initialize action tensor, network output monitor, and spike train record.
        action = torch.randint(low=0, high=env.action_space.n, size=(1,))[0]
        spike_record = torch.zeros(
            (steps, int(gran / dt), env.action_space.n), device=DEVICE
        )
        # perf_ax = None

        # Run through episode.
        while not done:

            step += 1
            obs, reward, done, intercept = env.step(action)
            obs = torch.Tensor(obs).to(DEVICE)
            reward = reward.to(DEVICE)

            # Determine the action probabilities
            probabilities = torch.softmax(
                torch.sum(spike_record[step - 1 % steps], dim=0), dim=0
            )
            action = torch.multinomial(probabilities, num_samples=1).item()

            # Place the observations into the inputs.
            obs = obs.unsqueeze(0)
            inputs = {LAYER1: poisson(obs * 5e2, gran, dt, device=DEVICE)}
            if DEVICE == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # Run the network on the spike train-encoded inputs.
            net.run(inputs=inputs, time=gran, reward=reward)
            spike_record[step % steps] = spikes[LAYER3].get("s").squeeze()
            rewards[step - 1] = reward.item()

            # record successful intercept
            if intercept:
                intercepts += 1

            if done:
                # Update network with cumulative reward
                if net.reward_fn is not None:
                    net.reward_fn.update(accumulated_reward=total_reward, steps=step)

                # Save rewards thus far to file
                if rfname != "":
                    f = open(rfname, "ab")
                    np.savetxt(f, rewards, delimiter=",", fmt="%.6f")
                    f.close()

            spikes_ = {layer: spikes[layer].get("s").view(gran, -1) for layer in spikes}
            spike_ims, spike_axes = plot_spikes(spikes_, ims=spike_ims, axes=spike_axes)
            # perf_ax = plot_performance(reward, x_scale=10, ax=perf_ax)

            env.render()
            total_reward += reward

            if step % 10 == 0:
                print(
                    f"Iteration: {step} (Time: {time.time() - clock:.4f}); reward: {reward}"
                )
                clock = time.time()

        print(f"Episode {ep} total reward:{total_reward}")
        # Save intcercepts thus far to file
        if pfname != "":
            f = open(pfname, "a+")
            if 0 < ep:
                f.write("," + str(intercepts))
            else:
                f.write(str(intercepts))
            f.close()

        # Cycle output files every 10000 iterations
        if ep % fcycle == 0:
            env.cycleOutFiles()


def main():

    # Build network.
    network = Network(dt=dt)

    # Input Layer
    inpt = Input(n=dim * dim, shape=[1, 1, 1, dim, dim], traces=True)

    # Hidden Layer
    middle = LIFNodes(n=neurons, traces=True)

    # Ouput Layer
    out = LIFNodes(n=moveChoices, refrac=0, traces=True)

    # Connections from input layer to hidden layer
    inpt_middle = Connection(source=inpt, target=middle, wmin=0, wmax=1)

    # Connections from hidden layer to output layer
    middle_out = Connection(
        source=middle,
        target=out,
        wmin=0,  # minimum weight value
        wmax=1,  # maximum weight value
        update_rule=MSTDPET,  # learning rule
        nu=1e-1,  # learning rate
        norm=0.5 * middle.n,  # normalization
    )

    # Recurrent connection, retaining data within the hidden layer
    recurrent = Connection(
        source=middle,
        target=middle,
        wmin=0,  # minimum weight value
        wmax=1,  # maximum weight value
        update_rule=PostPre,  # learning rule
        nu=1e-1,  # learning rate
        norm=5e-3 * middle.n,  # normalization
    )

    # Add all layers and connections to the network.
    network.add_layer(inpt, name=LAYER1)
    network.add_layer(middle, name=LAYER2)
    network.add_layer(out, name=LAYER3)
    network.add_connection(inpt_middle, source=LAYER1, target=LAYER2)
    network.add_connection(middle_out, source=LAYER2, target=LAYER3)
    network.add_connection(recurrent, source=LAYER2, target=LAYER2)
    network.to(DEVICE)

    # Add monitors
    # network.add_monitor(Monitor(network.layers["Hidden"], ["s"], time=granularity), "Hidden")
    # network.add_monitor(Monitor(network.layers["Output"], ["s"], time=granularity), "Output")
    spikes = {}
    for layer in set(network.layers):
        spikes[layer] = Monitor(
            network.layers[layer],
            state_vars=["s"],
            time=int(granularity / dt),
            device=DEVICE,
        )
        network.add_monitor(spikes[layer], name=layer)

    # Load the Dot Simultation environment.
    environment = DotSimulator(
        steps,
        decay=decay,
        herrs=herrs,
        diag=diag,
        randr=randr,
        write=write,
        mute=mute,
        bound_hand=boundh,
        fit_func=fit_func,
        allow_stay=allow_stay,
        pandas=pandas,
        fpath=OUT_FILE_PATH,
    )
    environment.reset()

    print("Training: ")
    rewFile = genFileName("rew", "train")
    perfFile = genFileName("perf", "train")
    environment.addFileSuffix("train")
    runSimulator(
        network,
        environment,
        spikes,
        episodes=trn_eps,
        gran=granularity,
        rfname=rewFile,
        pfname=perfFile,
    )

    # Freeze learning
    network.learning = False

    print("Testing: ")
    rewFile = genFileName("rew", "test")
    perfFile = genFileName("perf", "test")
    environment.changeFileSuffix("train", "test")
    runSimulator(
        network,
        environment,
        spikes,
        episodes=tst_eps,
        gran=granularity,
        rfname=rewFile,
        pfname=perfFile,
    )


if __name__ == "__main__":
    main()
