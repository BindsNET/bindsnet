import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from scripts.Chris.DQN.Environment import Maze_Environment, Grid_Cell_Maze_Environment

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
  def __init__(self, capacity):
      self.memory = deque([], maxlen=capacity)

  def push(self, *args):
      """Save a transition"""
      self.memory.append(Transition(*args))

  def sample(self, batch_size):
      return random.sample(self.memory, batch_size)

  def __len__(self):
      return len(self.memory)

class DQN(nn.Module):

  def __init__(self, n_observations, n_actions):
    super(DQN, self).__init__()
    self.layer1 = nn.Linear(n_observations, 128)
    self.layer2 = nn.Linear(128, 128)
    self.layer3 = nn.Linear(128, n_actions)

  # Called with either one element to determine next action, or a batch
  # during optimization. Returns tensor([[left0exp,right0exp]...]).
  def forward(self, x):
    x = F.relu(self.layer1(x))
    x = F.relu(self.layer2(x))
    return self.layer3(x)


# Select action using epsilon-greedy policy
def select_action(state, step, eps, policy_net, env):

  # Select action from policy net
  if random.random() > eps:
    with torch.no_grad():
      # t.max(1) will return the largest column value of each row.
      # second column on max result is index of where max element was
      # found, so we pick action with the larger expected reward.
      return policy_net(state).max(1).indices.view(1, 1)

  # Select random action (exploration)
  else:
    return torch.tensor(np.random.choice(env.num_actions)).view(1, 1)


# Optimize DQN
def optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma, device):
  if len(memory) < batch_size:
    return
  transitions = memory.sample(batch_size)
  # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
  # detailed explanation). This converts batch-array of Transitions
  # to Transition of batch-arrays.
  batch = Transition(*zip(*transitions))

  # Compute a mask of non-final states and concatenate the batch elements
  # (a final state would've been the one after which simulation ended)
  non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
  non_final_next_states = torch.cat([s for s in batch.next_state
                                     if s is not None])
  state_batch = torch.cat(batch.state)
  action_batch = torch.cat(batch.action)
  reward_batch = torch.cat(batch.reward)

  # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
  # columns of actions taken. These are the actions which would've been taken
  # for each batch state according to policy_net
  state_action_values = policy_net(state_batch).gather(1, action_batch)

  # Compute V(s_{t+1}) for all next states.
  # Expected values of actions for non_final_next_states are computed based
  # on the "older" target_net; selecting their best reward with max(1).values
  # This is merged based on the mask, such that we'll have either the expected
  # state value or 0 in case the state was final.
  next_state_values = torch.zeros(batch_size, device=device)
  with torch.no_grad():
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
  # Compute the expected Q values
  expected_state_action_values = (next_state_values * gamma) + reward_batch

  # Compute Huber loss
  criterion = nn.SmoothL1Loss()
  loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

  # Optimize the model
  optimizer.zero_grad()
  loss.backward()
  # In-place gradient clipping
  torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
  optimizer.step()


# Run single episode of Maze env for DQN training
def run_episode(env, policy_net, device, max_steps, eps=0):
  # Initialize the environment and get its state
  state, info = env.reset()
  state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
  t = 0
  while t < max_steps:
    action = select_action(state, t, eps, policy_net, env)    # eps = 0 -> no exploration
    observation, reward, terminated, _ = env.step(action.item())

    if terminated:
      next_state = None
    else:
      next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

    # Move to the next state
    state = next_state

    if terminated:
      break

    t+=1



def train_DQN(input_size, env_width, env_height, lr, batch_size, eps_start,
              eps_end, decay_intensity, tau, gamma, max_steps_per_ep, max_total_steps, max_eps, plot):
  device = 'cpu'
  n_actions = 4
  policy_net = DQN(input_size, n_actions).to(device)
  target_net = DQN(input_size, n_actions).to(device)
  target_net.load_state_dict(policy_net.state_dict())
  optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
  memory = ReplayMemory(1000)
  env = Grid_Cell_Maze_Environment(width=env_width, height=env_height)

  ## Pre-training recording ##
  if plot:
    run_episode(env, policy_net, device, 100, eps=0.9)
    env.animate_history("pre_training.gif")

  episode_durations = []
  episodes = 0
  total_steps = 0
  print(env.maze)
  while total_steps < max_total_steps: # and episodes < max_eps:
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
      eps = eps_end + (eps_start - eps_end) * math.exp(-decay_intensity * total_steps / (max_total_steps))
      action = select_action(state, t, eps, policy_net, env)
      observation, reward, terminated, _ = env.step(action.item())
      reward = torch.tensor([reward], device=device)

      if terminated:
        next_state = None
      else:
        next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

      # Store the transition in memory
      memory.push(state, action, next_state, reward)

      # Move to the next state
      state = next_state

      # Perform one step of the optimization (on the policy network)
      optimize_model(memory, batch_size, policy_net, target_net, optimizer, gamma=gamma, device=device)

      # Soft update of the target network's weights
      # θ′ ← τ θ + (1 −τ )θ′
      target_net_state_dict = target_net.state_dict()
      policy_net_state_dict = policy_net.state_dict()
      for key in policy_net_state_dict:
        target_net_state_dict[key] = policy_net_state_dict[key] * tau + target_net_state_dict[key] * (1 - tau)
      target_net.load_state_dict(target_net_state_dict)

      total_steps += 1
      if terminated or t > max_steps_per_ep:
        episode_durations.append(t + 1)
        break
    print(f"Episode {episodes} lasted {t + 1} steps, eps = {round(eps, 2)} total steps = {total_steps}")
    episodes += 1

  ## Post-training recording ##
  if plot:
    env.reset()
    run_episode(env, policy_net, device, 100, eps=0)  # eps = 0 -> no exploration
    env.animate_history("post_training.gif")
    plt.clf()

    plt.plot(episode_durations)
    plt.title("Episode durations")
    plt.ylabel("Duration")
    plt.xlabel("Episode")
    plt.show()
