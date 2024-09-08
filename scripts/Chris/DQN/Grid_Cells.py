from scipy.stats import multivariate_normal
import numpy as np
from matplotlib import pyplot as plt

class Grid_Cell:
  def __init__(self, x_range, y_range, x_offset, y_offset, scale=1, var=1, color='b'):
    self.centers = np.mgrid[x_range[0]:x_range[1]:scale, y_range[0]:y_range[1]:scale].transpose(1, 2, 0).astype(float)
    self.centers[:, :, 0] += x_offset
    self.centers[:, :, 1] += y_offset
    self.centers[:, ::2, 0] += 0.5 * scale
    # self.centers[::2, :, 1] += 0.5 * scale
    self.x_range = x_range
    self.y_range = y_range
    self.color = color
    self.var = var

  # Produce Grid Cell spike behavior relative to position
  # scale: Distance between grid cells
  def generate(self, pos):
    # Find closest center
    distances = np.linalg.norm(self.centers - pos, axis=2)
    closest_center = self.centers[np.unravel_index(np.argmin(distances), distances.shape)]
    mvn = multivariate_normal(mean=closest_center, cov=np.eye(2) * (self.var / (2 * np.pi)))
    activity = mvn.pdf(pos)
    return activity, closest_center

  def plot_activity(self, activity, center, color):
    for i in range(self.centers.shape[0]):
      for j in range(self.centers.shape[1]):
        x, y = self.centers[i, j]
        if np.all(center == (x, y)):
          c = plt.Circle((x, y), activity + 0.01, fill=True, alpha=0.5)
          plt.plot(x, y, '.', alpha=0.5, color=color)
          plt.gca().add_artist(c)
        else:
          plt.plot(x, y, '.', alpha=0.5, color=color)
        # c = plt.Circle((x, y), activity[i, j] + 0.01, color=color, fill=True, alpha=0.5)
    plt.xlim(self.x_range[0]-1, self.x_range[1]+1)
    plt.ylim(self.y_range[0]-1, self.y_range[1]+1)

  def plot_centers(self, color):
    for i in range(self.centers.shape[0]):
      for j in range(self.centers.shape[1]):
        x, y = self.centers[i, j]
        plt.plot(x, y, '.', alpha=0.5, color=color)
        # c = plt.Circle((x, y), activity[i, j] + 0.01, color=color, fill=True, alpha=0.5)
    plt.xlim(self.x_range[0]-1, self.x_range[1]+1)
    plt.ylim(self.y_range[0]-1, self.y_range[1]+1)


# Module of Grid Cell populations, each with a different scale
class GC_Module:
  def __init__(self, x_range, y_range, scales, offsets, vars):
    # self.colors =
    self.grid_cells = [Grid_Cell(x_range, y_range, x_ofst, y_ofst, s, v) for
                        (x_ofst, y_ofst), s, v in zip(offsets, scales, vars)]
    self.scales = scales
    self.x_range = x_range
    self.y_range = y_range
    self.offsets = offsets
    self.colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple', 'brown',
                   'pink', 'gray', 'olive', 'cyan', 'lime', 'teal', 'lavender', 'tan', 'salmon',
                   'gold', 'indigo', 'maroon', 'navy', 'peru', 'sienna', 'tomato', 'violet', 'wheat',]

  # Generate Grid Cell activity
  def generate(self, pos):
    activities = []
    centers = []
    for gc in self.grid_cells:
      a, c = gc.generate(pos)
      activities.append(a)
      centers.append(c)
    return np.array(activities), np.array(centers)

  # Plot Grid Cell activity
  def plot_activity(self, activities, centers):
    for i, gc in enumerate(self.grid_cells):
      gc.plot_activity(activities[i], centers[i], self.colors[i])

  # Plot Grid Cell Centers
  def plot_centers(self):
    for i, gc in enumerate(self.grid_cells):
      gc.plot_centers(self.colors[i])


# Take in grid cell activity vector and turn into spike train
# Activity converted to spike rate
# max_freq: Maximum frequency of spikes
def activity_to_spike(activity, time, max_freq):
  # Normalize [0, 1]
  activity = (activity - min(activity)) / (max(activity) - min(activity))

  # Convert to spike rate
  spike_rate = activity * max_freq
  spike_train = np.zeros((time, len(activity)))
  for i, rate in enumerate(spike_rate):
    if rate != 0:
      spike_train[:, i] = np.zeros(time)
      spike_train[:, i][np.random.rand(time) < rate] = 1
    else:
      spike_train[:, i] = np.zeros(time)

  return spike_train

if __name__ == '__main__':
  np.random.seed(5)
  num_cells = 5

  # Grid Cell activity range
  x_range_ = (0, +5)
  y_range_ = (0, +5)

  # Agent position
  pos_ = (0, 0)

  # Grid Cell offsets
  x_offsets_ = np.random.uniform(-1, 1, num_cells)
  y_offsets_ = np.random.uniform(-1, 1, num_cells)
  offsets = list(zip(x_offsets_, y_offsets_))

  # How far apart Grid Cells are
  scales = [1 + 0.1*i for i in range(num_cells)]

  # Variance for activity sampling around Grid Cell centers
  vars_ = [1]*num_cells

  # Initialize Grid Cell Module
  module = GC_Module(x_range_, y_range_, scales, offsets, vars_)
  a_, c_ = module.generate(pos_)
  # test = activity_to_spike(a_, 50, 0.5)
  print(f"Activity vector: {a_}")
  module.plot_activity(a_, c_)
  plt.show()
