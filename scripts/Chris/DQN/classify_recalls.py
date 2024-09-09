from torch.optim import Adam
from matplotlib import pyplot as plt
from ANN import ANN
import pickle as pkl
import torch

class Recalled_Mem_Dataset(torch.utils.data.Dataset):
  def __init__(self, samples, labels):
    self.samples = samples
    self.labels = labels

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, idx):
    # Compress spike train into windows for dimension reduction
    return self.samples[idx].flatten(), self.labels[idx]

def classify_recalls(out_dim, train_ratio, batch_size, epochs):
  print("Classifying recalled memories...")

  ## Load recalled memory samples ##
  with open('Data/preprocessed_recalls.pkl', 'rb') as f:
    samples, labels = pkl.load(f)

  ## Initialize ANN ##
  in_dim = samples[0].shape[0]
  model = ANN(in_dim, out_dim)
  optimizer = Adam(model.parameters())
  criterion = torch.nn.MSELoss()
  dataset = Recalled_Mem_Dataset(samples, labels)
  train_size = int(train_ratio * len(dataset))
  test_size = len(dataset) - train_size
  train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
  train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size, shuffle=True)

  ## Training ##
  loss_log = []
  accuracy_log = []
  for epoch in range(epochs):
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


# if __name__ == '__main__':
  # ## Constants ##
  # OUT_DIM = 2
  # TRAIN_RATIO = 0.8
  # BATCH_SIZE = 10
  #
  # ## Load recalled memory samples ##
  # with open('Data/preprocessed_recalls.pkl', 'rb') as f:
  #   samples, labels = pkl.load(f)
  #
  # ## Initialize ANN ##
  # in_dim = samples[0].shape[0] * samples[0].shape[1]
  # model = ANN(in_dim, OUT_DIM)
  # optimizer = Adam(model.parameters())
  # criterion = torch.nn.MSELoss()
  # dataset = Recalled_Mem_Dataset(samples, labels)
  # train_size = int(TRAIN_RATIO * len(dataset))
  # test_size = len(dataset) - train_size
  # train_set, test_set = torch.utils.data.random_split(dataset, [train_size, test_size])
  # train_loader = torch.utils.data.DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
  # test_loader = torch.utils.data.DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)
  #
  # ## Training ##
  # loss_log = []
  # for epoch in range(20):
  #   total_loss = 0
  #   for memory_batch, positions in train_loader:
  #     # positions_ = torch.tensor([[positions_[0][i], positions_[1][i]] for i, _ in enumerate(positions_[0])], dtype=torch.float32)
  #     optimizer.zero_grad()
  #     outputs = model(memory_batch)
  #     loss = criterion(outputs, positions.to(torch.float32))
  #     loss.backward()
  #     optimizer.step()
  #     total_loss += loss.item()
  #   loss_log.append(total_loss)
  #   print(f'Epoch: {epoch}, Total Loss: {total_loss}')
  # plt.xlabel('Epoch')
  # plt.ylabel('Loss')
  # plt.plot(loss_log)
  # plt.show()
  #
  # ## Testing ##
  # total = 0
  # correct = 0
  # with torch.no_grad():
  #   for memories, labels in test_loader:
  #     outputs = model(memories)
  #     loss = criterion(outputs, labels)
  #     total += len(labels)
  #     correct += torch.all(outputs.round() == labels.round(), dim=1).sum().item()  # Check if prediction for both x and y are correct
  #
  # print(f'Accuracy: {round(correct/total, 3)}%')
