import os
import random
from time import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from gym import spaces

# Mappings for changing direction if reflected.
# Cannot cross a row boundary moving right or left.
ROW_CROSSING = {
    1: 2,
    3: -2,
    5: 1,
    6: -1,
    7: 1,
    8: -1,
}

# Cannot cross a column boundary moving up or down.
COL_CROSSING = {
    2: 2,
    4: -2,
    5: 3,
    6: 1,
    7: -1,
    8: -3,
}


class Dot:
    def __init__(self, r: int = 0, c: int = 0, t: int = 1) -> None:
        # Initialize current point and tail to initial point.
        self.row = np.ones(t, dtype="int32") * r
        self.col = np.ones(t, dtype="int32") * c

    def move(self, r: int, c: int):
        """
        Cycle path history and set new current coordinates.

        :param r: row
        :param c: column
        """

        # Cycle the path history.
        for t in reversed(range(1, len(self.row))):
            self.row[t] = self.row[t - 1]
            self.col[t] = self.col[t - 1]

        # Set new current point
        self.row[0] = r
        self.col[0] = c


class DotSimulator:
    """
    This simulator lets us generate dots and make them move.
    It's especially useful in keeping entitled cats occupied,
    but instead of feline neurons, we use this for fake ones.

    Specifically, this generates a grid for each timestep, where a specified
    number of points have values of 1 with fading tails ("decay"), designating
    the current positions and movements of their corresponding dots. All other
    points are set to 0. From timestep to timestep, the dots either remain
    where they are or move one space.

    The 2D observation of the current state is provided every step, as well as
    the reward, completion flag, and sucessful interception flag. It may be
    helpful to amplify the grid values when encoding them as spike trains.

    :param t: int: number of timesteps/samples of grids with dot movements
    :param height: int: height dimension of the grid (rows)
    :param width: int: width dimension of the grid (columns)
    :param decay: int: length of decaying tail behind a dot (its path history)
    :param dots: int: number of target dots
    :param herrs: number of distraction dots (red herrings)
    :param pandas: Bool: print as pandas dataframe versus graphical plots
    :param write: Bool: write grids to file to be plotted later.
    :param mute: Bool: mute graphical rendering (can write to file or print pandas)
    :param speed: int: set movement speed of dots.
    :param randr: float: set the randomization rate of target movements.
    :param allow_stay: bool: allow a dot to remain in place as a movement choice.
    :param seed: int: optional seed for RNG in movement generation.
    :param fname: string: optional filename for saving grids to file
    :param fpath: string: optional file path for saving grids to file
    :param diag: Bool: allow diagonal movement.
    :param bound_hand: str: bounds handling when a dot reaches the world's end.
            'stay':   dots will simply be prevented from crossing the edges.
            'bounce': dot positions and directions will be reflected.
            'trans':  dot positions will be mirrored to the opposite edge.
    :param fit_func: str: Fitness function.
            'euc':  Single Euclidean (Pythagorean) distance value
            'disp': Tuple of x,y displacement values
            'rng' : Range rings--the closer the ring, the lower the number
            'dir' : directional--+1 if moving in the right direction
                                 -1 if moving in the wrong direction
                                  0 if neither.
    :param ring_size: int: set range ring size for range ring fitness function.
    :param bullseye: int: set reward for successful intercept; default = 10.0
    :param teleport: Bool: teleport network dot after intercept; default = true
    """

    def __init__(self, t: int, **kwargs) -> None:

        self.timesteps = t  # total timesteps
        self.ts = 0  # initialize current timestep to 0

        """ Keyword arguments """
        self.h = kwargs.get("height", 28)  # height dimension
        self.w = kwargs.get("width", 28)  # width dimension
        self.decay = kwargs.get("decay", 1)  # length of a dot's tail (path history)
        self.ndots = kwargs.get("dots", 1)  # Number of dots
        self.herrs = kwargs.get("herrs", 0)  # Red herrings (distractions)
        self.pandas = kwargs.get("pandas", False)  # print as pandas DF
        self.write2F = kwargs.get("write", False)  # write grids to file
        self.mute = kwargs.get("mute", False)  # mute displayed rendering
        self.speed = kwargs.get("speed", 1)  # dot movement speed
        self.randr = kwargs.get("randr", 1.0)  # rate of random movement
        self.minch = int(not kwargs.get("allow_stay", True))  # allow stay choice.

        # Grab system time and trim off extra large parts of the number.
        sysTime = time()
        sysTime = int(1e10 * (sysTime - 1e6 * (sysTime // 1e6)))

        # Save off RNG seed.
        self.seed = kwargs.get("seed", 0)
        if self.seed == 0:
            self.seed = sysTime

        # Create filename if one isn't provided.
        path = kwargs.get("fpath", "out")
        self.filename = kwargs.get("fname", "grids_s" + str(self.seed) + ".csv")
        self.fileCnt = 0
        self.filename = (
            path
            + "/"
            + self.filename[:-5]
            + "_"
            + str(self.fileCnt)
            + self.filename[-4:]
        )
        if not os.path.exists(path):
            os.makedirs(path)

        # Expand movement options if diagonal is allowed.
        if kwargs.get("diag", False):
            self.choices = 9
        else:
            self.choices = 5

        # Enumerated bounds handling when a dot traverses the region's edge.
        bh = kwargs.get("bound_hand", "stay")
        if bh == "stay":
            self.b_handling = 0  # Don't move if directed past the edge.
        elif bh == "bounce":
            self.b_handling = 1  # Bounce off edge (reflect coordinates).
        elif bh == "trans":
            self.b_handling = 2  # Translate to opposite side of the region.
        else:
            assert False, "Unsupported bounds handling"

        # Enumerated fitness (reward) function.
        ff = kwargs.get("fit_func", "euc")
        if ff == "euc":
            self.fit_func = 0  # Single Euclidean (Pythagorean) distance value
        elif ff == "disp":
            self.fit_func = 1  # Tuple of x,y displacement values
        elif ff == "rng":
            self.fit_func = 2  # Range rings--the closer the ring, the lower the number
        elif ff == "dir":
            self.fit_func = 3  # direction--moving closer or farther away?
        else:
            assert False, "Unsupported fitness function"

        self.ring_size = kwargs.get("ring_size", 2)  # Range ring size
        self.bullseye = kwargs.get("bullseye", 10.0)  # Intercept reward
        self.teleport = kwargs.get("teleport", True)  # Teleport after intercept

        # Initialize empty lists of relevant and distraction dots.
        self.netDot = Dot(0, 0, self.decay)
        self.dots = []
        self.herrings = []
        self.obs = np.zeros((self.h, self.w))

        self.action_space = spaces.Discrete(self.choices)

        self.newPlot = True  # One-time flag

    def step(self, action):
        """
        Generates a grid for the current timestep.
        See above for full description.

        :param action: network's prediction of the dot movement
        :return obs: observation of grid matrix of shape (h,w)
        :return reward: precision of network's prediction in Euclidean distance
        :return done: indicates termination of simulation
        :return intercept: indicates a successful intercept this step
        """

        # Increment timestep
        self.ts += 1  # Increment timestep

        # If the random rate is high enough, update movement direction.
        if random.uniform(0, 1) <= self.randr:
            self.dotDir = random.randint(
                self.minch, self.choices - 1
            )  # five possible options

        # Initialize empty grid and populate as we update dots.
        self.obs = np.zeros((self.h, self.w))

        # Update network dot according to the network's action.
        self.prevRow = self.netDot.row[0]
        self.prevCol = self.netDot.col[0]
        if action is not None:
            self.movePoint(self.netDot, action)

        # self.obs = self.obs/(self.ndots + self.herrs)  # normalize
        reward, intercept = self.compute_reward()

        # Teleport network dot if intercept is successful.
        if intercept and self.teleport:
            bh1, bh2 = self.h // 5, 4 * self.h // 5
            bw1, bw2 = self.w // 5, 4 * self.w // 5
            r, c = random.randint(bh1, bh2), random.randint(bw1, bw2)
            self.netDot = Dot(r, c, self.decay)

            # Redo grid observation.
            self.obs = np.zeros((self.h, self.w))
            self.obs[r, c] = 1

        # Move all relevant dots in the same direction.
        for d in self.dots:
            self.movePoint(d)

        # Move distraction dots with individually randomized motion.
        for h in self.herrings:
            self.movePoint(h, random.randint(self.minch, self.choices - 1))

        reward = torch.Tensor(np.array(reward))
        done = self.timesteps <= self.ts

        return self.obs, reward, done, intercept

    def reset(self):
        """
        Reset dots to initial positions, and reset RNG seed.
        """

        self.ts = 0  # reset timesteps

        # Reset RNG
        random.seed(self.seed)

        # provide default observation
        self.obs = np.zeros((self.h, self.w))

        # Set boundaries (so we don't spawn points on the edge.
        # Not that there's a real problem with it, but it's boring.
        bh1, bh2 = self.h // 5, 4 * self.h // 5
        bw1, bw2 = self.w // 5, 4 * self.w // 5

        # Start dots in the middle.
        # midr = self.h//2   <= we randomize this now.
        # midc = self.w//2

        # We know that the sum from n=0 to n=N of 1 - n/N = (N + 1)/2
        # Thus, computing the initial grid space for a dot, given the
        # length of its tail N would be (self.decay + 1)/2.
        # But... we also have to cap it at 1. So, who cares?

        # Reinitalize network dot placement.
        r, c = random.randint(bh1, bh2), random.randint(bw1, bw2)
        self.netDot = Dot(r, c, self.decay)
        self.obs[r, c] = 1

        # Reinitalize target dot placement with initial movement direction.
        self.dots = []
        self.dotDir = random.randint(self.minch, self.choices - 1)
        for d in range(self.ndots):
            r, c = random.randint(bh1, bh2), random.randint(bw1, bw2)
            self.dots.append(Dot(r, c, self.decay))
            self.obs[r, c] = 1

        # Reinitalize red herring placement.
        self.herrings = []
        for h in range(self.herrs):
            r, c = random.randint(bh1, bh2), random.randint(bw1, bw2)
            self.herrings.append(Dot(r, c, self.decay))
            self.obs[r, c] = 1

        return self.obs

    def movePoint(self, d: Dot, dotDir: int = -1):
        """
        Apply clockwise directional enumeration.

        :param dotDir: enumerated movement as described above.
        :param/return r: current row    => next row
        :param/return c: current column => next column
        """

        # If not provided, use the known current direction.
        targetDir = False  # flag if we're using the target's direction.
        if dotDir < 0:
            dotDir = self.dotDir
            targetDir = True

        r, c = d.row[0], d.col[0]

        """ Apply clockwise directional enumeration. """
        # 0 means stay, though we also won't go past the edge.
        if dotDir == 1:  # up
            r += self.speed
        elif dotDir == 2:  # right
            c += self.speed
        elif dotDir == 3:  # down
            r -= self.speed
        elif dotDir == 4:  # left
            c -= self.speed
        elif dotDir == 5:  # up and right
            r += self.speed
            c += self.speed
        elif dotDir == 6:  # down and right
            r -= self.speed
            c += self.speed
        elif dotDir == 7:  # down and left
            r -= self.speed
            c -= self.speed
        elif dotDir == 8:  # up and left
            r += self.speed
            c -= self.speed
        elif dotDir != 0:  # Woops
            assert False, "Unsupported dot direction"

        """ When a dot attempts to move past an edge... """
        # Stay put.
        if self.b_handling == 0:
            r = max(min(r, self.h - 1), 0)
            c = max(min(c, self.w - 1), 0)
            # direction stays the same.

        # Bounce: reflect its coordinates back into the region.
        elif self.b_handling == 1:
            if r < 0 or self.h <= r:
                r = self.h - 1 - r % self.h  # reflect row
                if targetDir:
                    self.dotDir += ROW_CROSSING[dotDir]

            if c < 0 or self.w <= c:
                c = self.w - 1 - c % self.w  # reflect column
                if targetDir:
                    self.dotDir += COL_CROSSING[dotDir]

        # Translate: the dot will continue in the same direction
        #            from the opposite side of the region.
        elif self.b_handling == 2:
            r = r % self.h  # Mirror row
            c = c % self.w  # Mirror column
            # direction stays the same.

        # Woops
        else:
            assert False, "Unsupported bounds handling"

        # Update the saved point in the Dot class.
        # This also cycles the path history.
        d.move(r, c)

        # Update the grid with this point and its decaying trail.
        for t in range(self.decay):
            self.obs[d.row[t], d.col[t]] = min(
                self.obs[d.row[t], d.col[t]] + 1 - t / self.decay, 1
            )

    def compute_reward(self):
        """
        Computes reward according to the chosen fitness function.
        Returns reward and flag indicating a successful intercept.
        """
        # Add bull's eye reward (if we're using it)
        if (
            self.bullseye != 0
            and self.dots[0].row[0] == self.netDot.row[0]
            and self.dots[0].col[0] == self.netDot.col[0]
        ):
            return self.bullseye, True

        reward = 0.0

        # Euclidean distance
        if self.fit_func == 0:
            reward = -np.hypot(
                self.dots[0].row[0] - self.netDot.row[0],
                self.dots[0].col[0] - self.netDot.col[0],
            )

        # Displacement tensor
        elif self.fit_func == 1:
            reward = torch.Tensor(
                [
                    self.dots[0].row[0] - self.netDot.row[0],
                    self.dots[0].col[0] - self.netDot.col[0],
                ]
            )

        # Range rings; default range ring size = 2
        elif self.fit_func == 2:
            reward = (
                -np.hypot(
                    self.dots[0].row[0] - self.netDot.row[0],
                    self.dots[0].col[0] - self.netDot.col[0],
                )
                // self.ring_size
            )

        # Directional
        elif self.fit_func == 3:
            rd1 = abs(self.dots[0].row[0] - self.prevRow)
            rd2 = abs(self.dots[0].row[0] - self.netDot.row[0])
            cd1 = abs(self.dots[0].col[0] - self.prevCol)
            cd2 = abs(self.dots[0].col[0] - self.netDot.col[0])

            if rd2 < rd1:
                reward += 1.0  # right row movement
            elif rd1 < rd2:
                reward -= 1.0  # wrong row movement
            if cd2 < cd1:
                reward += 1.0  # right col movement
            elif cd1 < cd2:
                reward -= 1.0  # wrong col movement

        # Woops
        else:
            assert False, "Unsupported fitness function"

        return reward, False

    def render(self):
        """
        Display current state, either in ASCII or graphic plots.
        """

        # Double value of network dot only for visual aid in rendering.
        temp = self.obs
        temp[self.netDot.row, self.netDot.col] *= 2

        # Write to file if requested.
        if self.write2F:
            f = open(self.filename, "ab")
            np.savetxt(f, temp, delimiter=",")
            f.close()

        # Print as pandas dataframe if requested.
        if self.pandas:
            print("Timestep:", self.ts)
            print(pd.DataFrame(temp, dtype="uint32"))

        # Provide graphical rendering if requested.
        if not self.mute:
            # Otherwise, we'll render it as... I don't know yet.
            # print('Timestep:', self.ts)
            # get current figure, clear it, and replot.
            if self.newPlot:
                self.fig = plt.gcf()

            plt.figure(self.fig.number)
            plt.clf()
            plt.imshow(temp, cmap="hot", interpolation="nearest")

            # Only display colorbar once.
            if self.newPlot:
                self.newPlot = False
                # plt.ion()
                plt.colorbar()

            # Pause so that that GUI can do its thing.
            plt.pause(1e-8)

    def cycleOutFiles(self, newInt=-1):
        """
        Increments numbered suffix on output file to start a new one.
        """
        oldStr = "_" + str(self.fileCnt)
        if 0 <= newInt:
            self.fileCnt = newInt
        else:
            self.fileCnt += 1
        self.filename = self.filename.replace(oldStr, "_" + str(self.fileCnt))

    def addFileSuffix(self, suffix):
        """
        Adds suffix to output file (like "train" or "test").
        """
        self.filename = self.filename[:-5] + suffix + "_" + self.filename[-5:]

    def changeFileSuffix(self, sFrom, sTo):
        """
        Adds suffix to output file (like "train" or "test").
        """
        self.filename = self.filename.replace(sFrom, sTo)
        self.cycleOutFiles(newInt=0)  # reset file count.


def driver():
    steps = 200
    dotSim = DotSimulator(200)
    dotSim.reset()

    grids = np.zeros((steps, 28, 28))
    directions = np.zeros(steps)
    done = False
    for t in range(steps):
        grids[t], reward, done, info = dotSim.step(0)
        directions[t] = info["direction"]

    vals, cnts = np.unique(directions, return_counts=True)
    print(vals, cnts / steps)


if __name__ == "__main__":
    driver()
