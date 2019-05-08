from .torchvision_wrapper import torchvision_dataset_wrapper_creator

CIFAR10 = torchvision_dataset_wrapper_creator('CIFAR10')
CIFAR100 = torchvision_dataset_wrapper_creator('CIFAR100')
Cityscapes = torchvision_dataset_wrapper_creator('Cityscapes')
CocoCaptions = torchvision_dataset_wrapper_creator('CocoCaptions')
CocoDetection = torchvision_dataset_wrapper_creator('CocoDetection')
DatasetFolder = torchvision_dataset_wrapper_creator('DatasetFolder')
EMNIST = torchvision_dataset_wrapper_creator('EMNIST')
FakeData = torchvision_dataset_wrapper_creator('FakeData')
FashionMNIST = torchvision_dataset_wrapper_creator('FashionMNIST')
Flickr30k = torchvision_dataset_wrapper_creator('Flickr30k')
Flickr8k = torchvision_dataset_wrapper_creator('Flickr8k')
ImageFolder = torchvision_dataset_wrapper_creator('ImageFolder')
KMNIST = torchvision_dataset_wrapper_creator('KMNIST')
LSUN = torchvision_dataset_wrapper_creator('LSUN')
LSUNClass = torchvision_dataset_wrapper_creator('LSUNClass')
MNIST = torchvision_dataset_wrapper_creator('MNIST')
Omniglot = torchvision_dataset_wrapper_creator('Omniglot')
PhotoTour = torchvision_dataset_wrapper_creator('PhotoTour')
SBU = torchvision_dataset_wrapper_creator('SBU')
SEMEION = torchvision_dataset_wrapper_creator('SEMEION')
STL10 = torchvision_dataset_wrapper_creator('STL10')
SVHN = torchvision_dataset_wrapper_creator('SVHN')
VOCDetection = torchvision_dataset_wrapper_creator('VOCDetection')
VOCSegmentation = torchvision_dataset_wrapper_creator('VOCSegmentation')

from .spoken_mnist import SpokenMNIST

__all__ = ['spike_encoders', 'SpokenMNIST',
           'CIFAR10', 'CIFAR100', 'Cityscapes',
           'CocoCaptions', 'CocoDetection', 'DatasetFolder', 'EMNIST',
           'FakeData', 'FashionMNIST', 'Flickr30k', 'Flickr8k',
           'ImageFolder', 'KMNIST', 'LSUN', 'LSUNClass', 'MNIST',
           'Omniglot', 'PhotoTour', 'SBU', 'SEMEION', 'STL10', 'SVHN',
           'VOCDetection', 'VOCSegmentation']
