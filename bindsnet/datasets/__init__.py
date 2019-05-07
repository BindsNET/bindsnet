from .torchvision_wrapper import vd_wrapper_creator

MNIST = vd_wrapper_creator('MNIST')
CIFAR10 = vd_wrapper_creator('CIFAR10')
CIFAR100 = vd_wrapper_creator('CIFAR100')
FashionMNIST = vd_wrapper_creator('FashionMNIST')
ImageFolder = vd_wrapper_creator('ImageFolder')

from .spike_encoders import *
