import numpy as np
import pickle as pkl
from sample_generator import sample_generator
from spike_train_generator import spike_train_generator
from store_reservoir import store_reservoir
from recall_reservoir import recall_reservoir
from recalled_mem_preprocessing import recalled_mem_preprocessing
from classify_recalls import classify_recalls

if __name__ == '__main__':
  ## Constants ##
  WIDTH = 5
  HEIGHT = 5
  SAMPLES_PER_POS = 1000
  NOISE = 0.1   # Noise in sampling
  WINDOW_FREQ = 10
  WINDOW_SIZE = 10
  NUM_CELLS = 20
  X_RANGE = (0, 5)
  Y_RANGE = (0, 5)
  SIM_TIME = 50
  MAX_SPIKE_FREQ = 0.3
  GC_MULTIPLES = 1
  RES_SIZE = 250
  STORE_SAMPLES = 100
  hyper_params = {
    'thresh': -55,
    'theta_plus': 0,
    'refrac': 1,
    'reset': -65,
    'tc_theta_decay': 500,
    'tc_decay': 30,
    'nu': (0.01, -0.01),
    'range': [-1, 1],
    'decay': None,
  }
  WINDOW_FREQ = 10
  WINDOW_SIZE = 10
  OUT_DIM = 2
  TRAIN_RATIO = 0.8
  BATCH_SIZE = 10
  PLOT = True

  ## Sample Generation ##
  x_offsets = np.random.uniform(-1, 1, NUM_CELLS)
  y_offsets = np.random.uniform(-1, 1, NUM_CELLS)
  offsets = list(zip(x_offsets, y_offsets))           # Grid Cell x & y offsets
  scales = [np.random.uniform(1.7, 5) for i in range(NUM_CELLS)]   # Dist. between Grid Cell peaks
  vars = [.85] * NUM_CELLS              # Variance of Grid Cell activity
  # samples, labels, sorted_samples = sample_generator(scales, offsets, vars, X_RANGE, Y_RANGE, SAMPLES_PER_POS,
  #                                                    noise=NOISE, padding=1, plot=PLOT)
  #
  # ## Spike Train Generation ##
  # spike_trains, labels, sorted_spike_trains = spike_train_generator(samples, labels, SIM_TIME, GC_MULTIPLES, MAX_SPIKE_FREQ)

  ## Association (Store) ##
  store_reservoir(RES_SIZE, STORE_SAMPLES, NUM_CELLS, GC_MULTIPLES, SIM_TIME, hyper_params, PLOT)

  ## Association (Recall) ##
  # recall_reservoir(RES_SIZE, SIM_TIME, PLOT)

  # # Preprocess Recalls ##
  # recalled_mem_preprocessing(WINDOW_FREQ, WINDOW_SIZE, PLOT)

  ## Train ANN ##
  # classify_recalls(OUT_DIM, TRAIN_RATIO, BATCH_SIZE)
