BindsNET supplies datasets in several different formats that all base on
the `torch.utils.data.Dataset`

# torchvision datasets

Wrappers around all `torchvision.datasets` are provided. This wrapper
(found in `torchvision_wrapper.py`) adds two arguments for encoding the
image and label.

## Tested

- CIFAR10
- CIFAR100
- MNIST
- EMNIST
- KMNIST
- FashionMNIST
- STL10
- SVHN

## Not tested

- Cityscapes
- CocoCaptions
- CocoDetection
- DatasetFolder
- FakeData
- Flickr30k
- Flickr8k
- ImageFolder
- LSUN
- LSUNClass
- Omniglot
- PhotoTour
- SEMEION
- SBU
- VOCDetection
- VOCSegmentation

# SpokenMNIST

File: `spoken_mnist.py`
URL: https://github.com/Jakobovski/free-spoken-digit-dataset
