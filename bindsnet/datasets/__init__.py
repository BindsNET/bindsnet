from .torchvision_wrapper import create_torchvision_dataset_wrapper
from .spoken_mnist import SpokenMNIST
from .davis import Davis
from .alov300 import ALOV300

from .collate import time_aware_collate
from .dataloader import DataLoader


CIFAR10 = create_torchvision_dataset_wrapper("CIFAR10")
CIFAR100 = create_torchvision_dataset_wrapper("CIFAR100")
Cityscapes = create_torchvision_dataset_wrapper("Cityscapes")
CocoCaptions = create_torchvision_dataset_wrapper("CocoCaptions")
CocoDetection = create_torchvision_dataset_wrapper("CocoDetection")
DatasetFolder = create_torchvision_dataset_wrapper("DatasetFolder")
EMNIST = create_torchvision_dataset_wrapper("EMNIST")
FakeData = create_torchvision_dataset_wrapper("FakeData")
FashionMNIST = create_torchvision_dataset_wrapper("FashionMNIST")
Flickr30k = create_torchvision_dataset_wrapper("Flickr30k")
Flickr8k = create_torchvision_dataset_wrapper("Flickr8k")
ImageFolder = create_torchvision_dataset_wrapper("ImageFolder")
KMNIST = create_torchvision_dataset_wrapper("KMNIST")
LSUN = create_torchvision_dataset_wrapper("LSUN")
LSUNClass = create_torchvision_dataset_wrapper("LSUNClass")
MNIST = create_torchvision_dataset_wrapper("MNIST")
Omniglot = create_torchvision_dataset_wrapper("Omniglot")
PhotoTour = create_torchvision_dataset_wrapper("PhotoTour")
SBU = create_torchvision_dataset_wrapper("SBU")
SEMEION = create_torchvision_dataset_wrapper("SEMEION")
STL10 = create_torchvision_dataset_wrapper("STL10")
SVHN = create_torchvision_dataset_wrapper("SVHN")
VOCDetection = create_torchvision_dataset_wrapper("VOCDetection")
VOCSegmentation = create_torchvision_dataset_wrapper("VOCSegmentation")
