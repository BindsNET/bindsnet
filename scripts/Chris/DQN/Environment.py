from labyrinth.generate import DepthFirstSearchGenerator
from labyrinth.grid import Cell, Direction
from labyrinth.maze import Maze
from labyrinth.solve import MazeSolver
import pickle as pkl
import matplotlib.pyplot as plt

class Maze_Environment(Maze):
  def __init__(self, width, height):

    # Generate basic maze & solve
    super().__init__(width=width, height=height, generator=DepthFirstSearchGenerator())
    solver = MazeSolver()
    self.path = solver.solve(self)
    self.agent_cell = self.start_cell

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
    pass

  # Takes action, returns next state, reward, done, info
  def step(self, action):
    # Check if action runs into wall
    if action not in self.agent_cell.open_walls:
      return self.agent_cell, -1, False, {}

    # Move agent
    else:
      self.agent_cell = self.agent_pos.neighbor(action)
      if self.agent_cell == self.end_cell:
        return self.agent_cell, 1, True, {}
      else:
        return self.agent_cell, 0, False, {}

  def save(self, filename):
    with open(filename, 'wb') as f:
      pkl.dump(self, f)


if __name__ == '__main__':
  maze = Maze_Environment(width=25, height=25)
  solver = MazeSolver()
  path = solver.solve(maze)
  maze.path = path
  print(maze)
  print(f'start: {maze.start_cell}')
  print(f'end: {maze.end_cell}')