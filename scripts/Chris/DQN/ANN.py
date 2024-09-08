import pickle as pkl
import random
from collections import namedtuple, deque

from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
from torch.nn import Module, Linear, ReLU, Sequential
from torch.optim import Adam
import torch

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

# https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
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


class ANN(Module):
  def __init__(self, input_dim, output_dim):
    super(ANN, self).__init__()
    self.sequence = Sequential(
        Linear(input_dim, 1000),
        ReLU(),
        Linear(1000, 100),
        ReLU(),
        Linear(100, output_dim)
    )

  def forward(self, x):
    x = x.to(torch.float32)
    return self.sequence(x)


class DQN:
  def __init__(self, input_dim, output_dim, gamma=0.99, batch_size=128, device='cpu'):
    self.policy_net = ANN(input_dim, output_dim)
    self.target_net = ANN(input_dim, output_dim)
    self.optimizer = Adam(self.policy_net.parameters())
    self.memory = ReplayMemory(10000)
    self.gamma = gamma
    self.batch_size = batch_size
    self.device = device

  def select_action(self, state, epsilon):
    # Random action
    if random.random() < epsilon:
      return torch.tensor([[random.randrange(2)]], dtype=torch.float32)

    # ANN action
    else:
      with torch.no_grad():
        return self.policy_net(state).argmax()

  def optimize_model(self):
    if len(self.memory) < self.batch_size:
      return
    transitions = self.memory.sample(self.batch_size)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=self.device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                       if s is not None]).reshape(-1, 2)
    state_batch = torch.cat(batch.state).reshape(-1, 2)
    action_batch = torch.tensor(batch.action).to(torch.int64)
    reward_batch = torch.tensor(batch.reward)

    # Compute Q(s_t, a)
    state_action_values = self.policy_net(state_batch)[action_batch]

    # Compute V(s_{t+1}) for all next states.
    next_state_values = torch.zeros(self.batch_size, device=self.device)
    with torch.no_grad():
      next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values

    # Compute the expected Q values
    expected_state_action_values = (next_state_values * self.gamma) + reward_batch

    # Compute Loss
    criterion = torch.nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    self.optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
    self.optimizer.step()

  def update_target(self, tau=0.005):
    target_net_state_dict = self.target_net.state_dict()
    policy_net_state_dict = self.policy_net.state_dict()
    for key in policy_net_state_dict:
      target_net_state_dict[key] = policy_net_state_dict[key]*tau + target_net_state_dict[key] * (1 - tau)


class Mem_Dataset(torch.utils.data.Dataset):
  def __init__(self, samples, labels):
    self.samples = samples
    self.labels = labels

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    # Compress spike train into windows for dimension reduction
    return self.samples[idx].sum(0).squeeze(), self.labels[idx]


if __name__ == '__main__':
  ### ANN for input spike trains ###
  # Load recalled memory samples ##
  with open('Data/grid_cell_spk_trains.pkl', 'rb') as f:
    samples, labels = pkl.load(f)

  ## Initialize ANN ##
  in_dim = samples[0].shape[1]
  model = ANN(in_dim, 2)
  optimizer = Adam(model.parameters())
  criterion = torch.nn.MSELoss()
  dataset = Mem_Dataset(samples, labels)
  train_size = int(0.8 * len(dataset))
  test_size = len(dataset) - train_size
  train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=32, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=32, shuffle=True)

  ## Training ##
  loss_log = []
  accuracy_log = []
  for epoch in range(10):
    total_loss = 0
    correct = 0
    for memory_batch, positions in train_loader:
      # positions_ = torch.tensor([[positions_[0][i], positions_[1][i]] for i, _ in enumerate(positions_[0])], dtype=torch.float32)
      optimizer.zero_grad()
      outputs = model(memory_batch)
      loss = criterion(outputs, positions.to(torch.float32))
      loss.backward()
      optimizer.step()
      total_loss += loss.item()
      correct += torch.all(outputs.round() == positions.round(),
                           dim=1).sum().item()
    accuracy_log.append(correct / len(train_set))
    loss_log.append(total_loss)

  plt.xlabel('Epoch')
  plt.ylabel('Loss')
  plt.title('Training Loss')
  plt.plot(loss_log)
  plt.show()
  plt.xlabel('Epoch')
  plt.ylabel('Accuracy')
  plt.title('Training Accuracy')
  plt.plot(accuracy_log)
  plt.show()

  ## Testing ##
  total = 0
  correct = 0
  confusion_matrix = torch.zeros(25, 25)
  out_of_bounds = 0
  with torch.no_grad():
    for memories, labels in test_loader:
      outputs = model(memories)
      loss = criterion(outputs, labels)
      total += len(labels)
      correct += torch.all(outputs.round() == labels.round(),
                           dim=1).sum().item()  # Check if prediction for both x and y are correct
      for t, p in zip(labels, outputs):
        label_ind = int(t[0].round() * 5 + t[1].round())
        pred_ind = int(p[0].round() * 5 + p[1].round())
        if label_ind < 0 or label_ind >= 25 or pred_ind < 0 or pred_ind >= 25:
          out_of_bounds += 1
        else:
          confusion_matrix[label_ind, pred_ind] += 1

  plt.imshow(confusion_matrix)
  plt.title('Confusion Matrix')
  plt.xlabel('Predicted')
  plt.ylabel('True Label')
  plt.colorbar()
  plt.show()

  print(f'Accuracy: {round(correct / total, 3)*100}%')

