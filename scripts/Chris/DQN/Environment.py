import random
import numpy as np
from labyrinth.generate import DepthFirstSearchGenerator
from labyrinth.grid import Cell, Direction
from labyrinth.maze import Maze
from labyrinth.solve import MazeSolver
from matplotlib.pyplot import plot as plt
from matplotlib.animation import FuncAnimation

import pickle as pkl
import matplotlib.pyplot as plt
from torch import optim

class Maze_Environment():
  def __init__(self, width, height):

    # Generate basic maze & solve
    self.width = width
    self.height = height
    self.maze = Maze(width=width, height=height, generator=DepthFirstSearchGenerator())
    self.solver = MazeSolver()
    self.path = self.solver.solve(self.maze)
    self.maze.path = self.path    # No idea why this is necessary
    self.agent_cell = self.maze.start_cell
    self.num_actions = 4
    self.history = [(self.agent_cell.coordinates, 0, False, {})]  # (state, reward, done, info)

  def plot(self):
    # Box around maze
    plt.plot([-0.5, self.width-1+0.5], [-0.5, -0.5], color='black')
    plt.plot([-0.5, self.width-1+0.5], [self.height-1+0.5, self.height-1+0.5], color='black')
    plt.plot([-0.5, -0.5], [-0.5, self.height-1+0.5], color='black')
    plt.plot([self.width-1+0.5, self.width-1+0.5], [-0.5, self.height-1+0.5], color='black')

    # Plot maze
    for row in range(self.height):
      for column in range(self.width):
        # Path
        cell = self.maze[column, row]  # Tranpose maze coordinates (just how the maze is stored)
        if cell == self.maze.start_cell:
          plt.plot(row, column, 'go')
        elif cell == self.maze.end_cell:
          plt.plot(row, column,'bo')
        elif cell in self.maze.path:
          plt.plot(row, column, 'ro')

        # Walls
        if Direction.S not in cell.open_walls:
          plt.plot([row-0.5, row+0.5], [column+0.5, column+0.5], color='black')
        if Direction.E not in cell.open_walls:
          plt.plot([row+0.5, row+0.5], [column-0.5, column+0.5], color='black')

  def reset(self):
    self.agent_cell = self.maze.start_cell
    self.history = [(self.agent_cell.coordinates, 0, False, {})]
    return self.agent_cell, {}

  # Takes action
  # Returns next state, reward, done, info
  def step(self, action):
    # Transform action into Direction
    if action == 0:
      action = Direction.N
    elif action == 1:
      action = Direction.E
    elif action == 2:
      action = Direction.S
    elif action == 3:
      action = Direction.W

    # Check if action runs into wall
    if action not in self.agent_cell.open_walls:
      self.history.append((self.agent_cell.coordinates, -0.1, False, {}))
      return self.agent_cell, -0.01, False, {}

    # Move agent
    else:
      self.agent_cell = self.maze.neighbor(self.agent_cell, action)
      if self.agent_cell == self.maze.end_cell:    # Check if agent has reached the end
        self.history.append((self.agent_cell.coordinates, 1, True, {}))
        return self.agent_cell, 1, True, {}
      else:
        self.history.append((self.agent_cell.coordinates, 0, False, {}))
        return self.agent_cell, 0, False, {}

  def save(self, filename):
    with open(filename, 'wb') as f:
      pkl.dump(self, f)

  def animate_history(self, file_name='maze.gif'):
    def update(i):
      plt.clf()
      self.plot()
      plt.plot(self.history[i][0][1], self.history[i][0][0], 'yo')
      plt.title(f'Step {i}, Reward: {self.history[i][1]}')
    ani = FuncAnimation(plt.gcf(), update, frames=len(self.history), repeat=False)
    ani.save(file_name, writer='ffmpeg', fps=5)

class Grid_Cell_Maze_Environment(Maze_Environment):
  def __init__(self, width, height):
    super().__init__(width, height)

    # Load spike train samples
    # {position: [spike_trains]}
    with open('Data/preprocessed_recalls_sorted.pkl', 'rb') as f:
      self.samples = pkl.load(f)

  def reset(self):
    cell, info = super().reset()
    return self.state_to_grid_cell_spikes(cell), info

  def step(self, action):
    obs, reward, done, info = super().step(action)
    obs = self.state_to_grid_cell_spikes(obs)
    return obs, reward, done, info

  def state_to_grid_cell_spikes(self, cell):
    return random.choice(self.samples[cell.coordinates])


if __name__ == '__main__':
  from train_DQN import DQN, ReplayMemory
  from scripts.Chris.DQN.train_DQN import run_episode

  device = 'cpu'
  n_actions = 4
  input_size = 300
  lr = 0.01
  policy_net = DQN(input_size, n_actions).to(device)
  target_net = DQN(input_size, n_actions).to(device)
  target_net.load_state_dict(policy_net.state_dict())
  optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
  memory = ReplayMemory(1000)
  env = Grid_Cell_Maze_Environment(width=5, height=5)

  run_episode(env, policy_net, 'cpu', 100, eps=0.9)
  env.animate_history()
