from labyrinth.generate import DepthFirstSearchGenerator
from labyrinth.grid import Cell, Direction
from labyrinth.maze import Maze
from labyrinth.solve import MazeSolver
import pickle as pkl
import matplotlib.pyplot as plt

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
        cell = self[column, row]  # Tranpose maze coordinates (just how the maze is stored)
        if cell == self.start_cell:
          plt.plot(row, column, 'go')
        elif cell == self.end_cell:
          plt.plot(row, column,'bo')
        elif cell in self.path:
          plt.plot(row, column, 'ro')

        # Walls
        if Direction.S not in cell.open_walls:
          plt.plot([row-0.5, row+0.5], [column+0.5, column+0.5], color='black')
        if Direction.E not in cell.open_walls:
          plt.plot([row+0.5, row+0.5], [column-0.5, column+0.5], color='black')

  def reset(self):
    # self.maze = Maze(width=self.width, height=self.height, generator=DepthFirstSearchGenerator())
    # self.solver = MazeSolver()
    # self.path = self.solver.solve(self.maze)
    # self.maze.path = self.path    # No idea why this is necessary
    # self.agent_cell = self.maze.start_cell
    # return self.agent_cell, {}
    self.agent_cell = self.maze.start_cell
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
      return self.agent_cell, -.5, False, {}

    # Move agent
    else:
      self.agent_cell = self.maze.neighbor(self.agent_cell, action)
      if self.agent_cell == self.maze.end_cell:    # Check if agent has reached the end
        return self.agent_cell, 1, True, {}
      else:
        return self.agent_cell, 0, False, {}

  def save(self, filename):
    with open(filename, 'wb') as f:
      pkl.dump(self, f)




if __name__ == '__main__':
  maze_env = Maze_Environment(width=25, height=25)
  print(maze_env.maze)
  print(f'start: {maze_env.maze.start_cell}')
  print(f'end: {maze_env.maze.end_cell}')
  maze_env.reset()
  print(maze_env.maze)
  print(f'start: {maze_env.maze.start_cell}')
  print(f'end: {maze_env.maze.end_cell}')
