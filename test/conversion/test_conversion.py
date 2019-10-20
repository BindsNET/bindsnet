import torch.nn as nn
import torch.nn.functional as F

from bindsnet.conversion import ann_to_snn


class FullyConnectedNetwork(nn.Module):
    # language=rst
    """
    Simply fully-connected network implemented in PyTorch.
    """

    def __init__(self):
        super(FullyConnectedNetwork, self).__init__()

        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


def test_conversion():
    ann = FullyConnectedNetwork()
    snn = ann_to_snn(ann, input_shape=(784,))


def main():
    ann = FullyConnectedNetwork()
    return ann_to_snn(ann, input_shape=(28, 28))


if __name__ == "__main__":
    main()
