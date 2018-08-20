import os
import argparse
import numpy as np
import pickle as p

from bindsnet.environment import GymEnvironment

parser = argparse.ArgumentParser()
parser.add_argument('-n', type=int, default=1000000)
parser.add_argument('--render', dest='render', action='store_true')
parser.set_defaults(render=False)

args = parser.parse_args()

n = args.n
render = args.render

# Load SpaceInvaders environment.
env = GymEnvironment('SpaceInvaders-v0')
env.reset()

total = 0
rewards = []
avg_rewards = []
lengths = []
avg_lengths = []

i, j, k = 0, 0, 0
while i < n:
    if render:
        env.render()

    # Select random action.
    a = np.random.choice(6)

    # Step environment with random action.
    obs, reward, done, info = env.step(a)

    total += reward

    rewards.append(reward)
    if i == 0:
        avg_rewards.append(reward)
    else:
        avg = (avg_rewards[-1] * (i - 1)) / i + reward / i
        avg_rewards.append(avg)

    if i % 100 == 0:
        print('Iteration %d: last reward: %.2f, average reward: %.2f' % (i, reward, avg_rewards[-1]))

    if done:
        # Restart game if out of lives.
        env.reset()

        length = i - j
        lengths.append(length)
        if j == 0:
            avg_lengths.append(length)
        else:
            avg = (avg_lengths[-1] * (k - 1)) / k + length / k
            avg_lengths.append(avg)

        print('Episode %d: last length: %.2f, average length: %.2f' % (k, length, avg_lengths[-1]))

        j += length
        k += 1

    i += 1

save = (total, rewards, avg_rewards, lengths, avg_lengths)
p.dump(save, open(os.path.join('..', '..', 'results', 'SI_random_baseline_%d.p' % n), 'wb'))
