import random
from time import time

import numpy as np
import torch

# Number of cues to be used in the experiment.
NUM_CUES = 4


class CueRewardSimulator:
    """
    This simulator provides basic cues and rewards according to the
    network's choice, as described in the Backpropamine paper:
        https://openreview.net/pdf?id=r1lrAiA5Ym

    :param epdur: int: duration (timesteps) of an episode; default = 200
    :param cuebits: int: max number of bits to hold a cue (max value = 2^n for n bits)
    :param seed: real: random seed
    :param zprob: real: probability of zero vectors in each trial.
    """

    def __init__(self, **kwargs) -> None:
        self.ep_duration = kwargs.get("epdur", 200)  # episode duration in timesteps
        self.cuebits = kwargs.get("cuebits", 20)
        self.seed = int(kwargs.get("seed", time()))
        self.zeroprob = kwargs.get("zprob", 0.6)

        # zero array consists of the binary cue vector + four other fields.
        self.zeroArray = np.zeros(self.cuebits + 4, dtype="int32")

        assert (
            0.0 <= self.zeroprob and self.zeroprob <= 1.0
        ), "zprob must be valid probability"

    def make(self, name):
        self.reset()  # Simply reset according to grid definition.

    def step(self, action):
        """
        Every trial, we randomly select two of the four cues to provide to the
        network. Every timestep within that trial we either randomly display
        only zeros, or we alternate between the two cues in the pair.

        At the end of a trial, we provide the response cue, for which the network
        must respond 1 if the target was one of the provided cues or 0 if it was
        not. The next timestep, we evaluate the response, giving a reward of 1
        for correct and -1 for incorrect.

        :param action: network's decision if the target cue was displayed.
        :return obs: observation of vector with binary cue and the following fields:
                - time since start of episode
                - one-hot-encoded value for a response of 0 in previous timestep
                - one-hot-encoded value for a response of 1 in previous timestep
                - reward of previous timestep
        :return reward: 1 for correct response; -1 for incorrect response.
        :return done: indicates termination of simulation
        :return info: dictionary including values for debugging purposes.
        """
        self.tstep += 1  # increment episode timestep
        self.trialTime -= 1  # decrement current trial timestep

        # Populate base fields of observation.
        self.obs = self.zeroArray  # default to empty array
        self.obs[-4] = self.tstep  # time since start of episode.
        self.obs[-3] = int(self.response == 0)  # response = 0 for previous timestep
        self.obs[-2] = int(self.response == 1)  # response = 1 for previous timestep
        self.obs[-1] = self.reward[0]  # reward of previous timestep

        self.response = action  # Remember previous response
        self.reward[0] = 0  # default current reward to 0

        # If starting a new trial
        if self.trialTime <= 0:
            # Set new trial length, based on mean number of trials per episode = 15
            self.trialTime = random.randint(10, 20) // self.ep_duration

            # Randomly select the pair of cues to be shown to the network.
            self.pairmask = np.array(range(NUM_CUES))[
                np.argsort(np.random.uniform(0, 1, 4)) < 2
            ]

            # Determine if target is one of these current cues displayed.
            self.targ_disp = int(np.any(self.pairmask == self.target))

            self.cue_pair_ind = 0  # Reset cue pair indicator

        # Deterministic special cases for last two trial timesteps.
        if self.trialTime <= 2:
            # If it's the second to last trial timestep, cue response.
            # Response cue is another binary cue vector but with a value of 1.
            if self.trialTime == 2:
                self.obs[0] = 1

            # If it's the last trial timestep, provide empty input, check
            # check the answer to the response cue, and compute the reward.
            else:
                # reward = 1 for correct and -1 for incorrect.
                self.reward[0] = int(action == self.targ_disp) * 2 - 1

        # Else, roll the dice for a zero vector.
        elif np.random.uniform(0, 1) > self.zeroprob:
            # If we're not providing a zero vector, present one of
            # the current cue pair and switch turns for next time.
            self.obs[:-4] = self.cues[self.pairmask][self.cue_pair_ind]
            self.cue_pair_ind = (self.cue_pair_ind + 1) % 2

        done = self.ep_duration <= self.tstep
        info = {
            "target": self.target,
            "pairmask": self.pairmask,
            "targ_disp": self.targ_disp,
        }

        return self.obs, self.reward, done, info

    def reset(self):
        """
        Reset reset RNG seed; generate new cue bit arrays, and arbitrarily
        select one of the four cues as the "target" cue.
        """
        # Re-seed random functions
        random.seed(self.seed)
        np.random.seed(self.seed)

        # Reset timesteps
        self.tstep = 0
        self.trialTime = 1

        # Initialize cue bit strings
        CUE_MAX = pow(2, self.cuebits)
        cues_ints = np.zeros(NUM_CUES, dtype="int32")
        for i in range(NUM_CUES):
            c = 0
            while np.any(cues_ints == c):
                c = random.randint(2, CUE_MAX)  # 1 reserved for response cue
            cues_ints[i] = c

        self.cues = np.zeros((NUM_CUES, self.cuebits), dtype="int32")
        for i in range(NUM_CUES):
            binarray = np.array(list(np.binary_repr(cues_ints[i]))).astype("int32")
            self.cues[i][: len(binarray)] = binarray

        # Randomly select the target cue for this episode.
        self.target = random.randint(0, NUM_CUES)

        # provide empty default observation
        self.obs = self.zeroArray

        # Reset reward
        self.reward = torch.Tensor(1)
        self.reward[0] = 0  # default reward to 0

        # Instantiate response member, defaulting to 0.
        self.response = 0

        return self.obs

    def render(self):
        """
        Display current input vector.
        """
        print(self.obs)


def driver():
    steps = 200
    cueSim = CueRewardSimulator()
    cueSim.reset()

    observations = np.zeros((steps, 24), dtype="int32")
    for t in range(steps):
        observations[t], reward, done, info = cueSim.step(0)

    meanReward = observations[:, -1][observations[:, -1] != 0].mean()
    print("Mean reward:", meanReward)


if __name__ == "__main__":
    driver()
