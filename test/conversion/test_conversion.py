import torch
import torch.nn as nn
import torch.nn.functional as F

from bindsnet.conversion import ann_to_snn, data_based_normalization


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


def test_conversion_1():
    ann = FullyConnectedNetwork()
    snn = ann_to_snn(ann, input_shape=(784,))


def test_conversion_2():
    data = torch.rand(784, 20)
    ann = FullyConnectedNetwork()
    snn = ann_to_snn(ann, data=data, input_shape=(784,))


def test_conversion_from_path(tmp_path):
    """
    ``ann_to_snn`` accepts a path to a saved network.

    Regression test: this path called ``torch.load`` without
    ``weights_only=False``, so it broke outright when PyTorch 2.6 changed that
    default to ``True``. Only the in-memory ``nn.Module`` form was tested, so
    the breakage went unnoticed.
    """
    ann = FullyConnectedNetwork()
    file_path = str(tmp_path / "ann.pt")
    torch.save(ann, file_path)

    snn = ann_to_snn(file_path, input_shape=(784,))

    assert snn is not None


def test_data_based_normalization_from_path(tmp_path):
    """
    ``data_based_normalization`` accepts a path to a saved network. Same
    PyTorch 2.6 regression as ``test_conversion_from_path``.
    """
    ann = FullyConnectedNetwork()
    file_path = str(tmp_path / "ann.pt")
    torch.save(ann, file_path)

    normalized = data_based_normalization(file_path, data=torch.rand(20, 784))

    assert isinstance(normalized, nn.Module)


def main():
    test_conversion_1()
    test_conversion_2()


if __name__ == "__main__":
    main()
