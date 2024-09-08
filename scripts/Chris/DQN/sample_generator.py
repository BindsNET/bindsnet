from matplotlib import pyplot as plt
import numpy as np
import pickle as pkl

from scripts.Chris.DQN.Grid_Cells import GC_Module

# Spread of activity between samples for each position
# We want to minimize this (i.e. we want the activity to be consistent across samples)
def intra_positional_spread(env_to_gc):
  spread = {}
  for pos, activities in env_to_gc.items():
    avg_activity = np.mean(activities, axis=0)
    spread[pos] = np.std(avg_activity)
  return spread

# Spread of activity between positions
# We want to maximize this (i.e. we want the activity to be different across positions)
def inter_positional_spread(env_to_gc):
  spread = {}
  for pos1, activities1 in env_to_gc.items():
    for pos2, activities2 in env_to_gc.items():
      if pos1 != pos2:
        avg_activity1 = np.mean(activities1, axis=0)
        avg_activity2 = np.mean(activities2, axis=0)
        spread[(pos1, pos2)] = np.linalg.norm(avg_activity1 - avg_activity2)
  return spread

# Generate grid cell activity for all integer coordinate positions in environment
def sample_generator(scales, offsets, vars, x_range, y_range, samples_per_pos, noise=0.1, padding=2, plot=False):
  print('Generating samples...')
  sorted_samples = {}
  samples = np.zeros((x_range[1] * y_range[1] * samples_per_pos, len(scales)))
  labels = np.zeros((x_range[1] * y_range[1] * samples_per_pos, 2))
  padded_x_range = (x_range[0] - padding, x_range[1] + padding)
  padded_y_range = (y_range[0] - padding, y_range[1] + padding)
  module = GC_Module(padded_x_range, padded_y_range, scales, offsets, vars)
  for i in range(x_range[1]):
    for j in range(y_range[1]):
      for k in range(samples_per_pos):    # Generate multiple samples for each position
        x_sign = 1 if np.random.rand() > 0.5 else -1  # (slight variations in position)
        y_sign = 1 if np.random.rand() > 0.5 else -1
        pos = (i + np.random.rand() * noise * x_sign, j + np.random.rand() * noise * y_sign)
        a, c = module.generate(pos)
        if (i, j) not in sorted_samples:
          sorted_samples[(i, j)] = [a]
        else:
          sorted_samples[(i, j)].append(a)
        ind = i * y_range[1] * samples_per_pos + j * samples_per_pos + k
        samples[ind] = a
        labels[ind] = np.array(pos)
  with open('Data/grid_cell_intensities.pkl', 'wb') as f:
    pkl.dump((samples, labels), f)
  with open('Data/grid_cell_intensities_sorted.pkl', 'wb') as f:
    pkl.dump((sorted_samples), f)

  if plot:
    module.plot_centers()
    plt.title('Grid Cell Centers')
    for i in range(x_range[1]):
      for j in range(y_range[1]):
        plt.plot(i, j, 'r+', markersize=10)
    plt.show()

  return samples, labels, sorted_samples

if __name__ == '__main__':
  ## Constants ##
  WIDTH = 5
  HEIGHT = 5
  SAMPLES_PER_POS = 1000
  WINDOW_FREQ = 10
  WINDOW_SIZE = 10
  # Grid Cells
  num_cells_ = 20
  x_range_ = (0, 5)
  y_range_ = (0, 5)
  x_offsets_ = np.random.uniform(-1, 1, num_cells_)
  y_offsets_ = np.random.uniform(-1, 1, num_cells_)
  offsets_ = list(zip(x_offsets_, y_offsets_))
  scales_ = [1 + 0.01 * i for i in range(num_cells_)]
  vars_ = [0.85]*num_cells_

  # Test spread for set of parameters
  # Shape = (num_samples, num_cells)
  samples_, labels_, sorted_samples_ = sample_generator(scales_, offsets_, vars_, x_range_, y_range_, SAMPLES_PER_POS)
