from .preprocess    import *

import os
import sys
import gzip
import torch
import urllib
import shutil
import tarfile
import pickle as p
import numpy  as np

from struct         import unpack
from urllib.request import urlretrieve


class MNIST:
	'''
	Handles loading and saving of the MNIST handwritten digits
	`(link) <http://yann.lecun.com/exdb/mnist/>`_.
	'''
	train_images_pickle = 'train_images.p'
	train_labels_pickle = 'train_labels.p'
	test_images_pickle = 'test_images.p'
	test_labels_pickle = 'test_labels.p'
	
	train_images_file = 'train-images-idx3-ubyte'
	train_labels_file = 'train-labels-idx1-ubyte'
	test_images_file = 't10k-images-idx3-ubyte'
	test_labels_file = 't10k-labels-idx1-ubyte'
	
	train_images_url = 'http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz'
	train_labels_url = 'http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz'
	test_images_url = 'http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz'
	test_labels_url = 'http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz'
	
	def __init__(self, path=os.path.join('data', 'MNIST')):
		'''
		Constructor for the :code:`MNIST` object. Makes the data directory if it doesn't already exist.

		Inputs:
		
			| :code:`path` (:code:`str`): pathname of directory in which to store the MNIST handwritten digit dataset.
		'''
		if not os.path.isdir(path):
			os.makedirs(path)
		
		self.path = path
		
	def get_train(self):
		'''
		Gets the MNIST training images and labels.

		Returns:
		
			| :code:`images` (:code:`torch.Tensor`): The MNIST training images.
			| :code:`labels` (:code:`torch.Tensor`): The MNIST training labels.
		'''
		if not os.path.isfile(os.path.join(self.path, MNIST.train_images_pickle)):
			# Download training images if they aren't on disk.
			print('Downloading training images.\n')
			self.download(MNIST.train_images_url, MNIST.train_images_file)
			images = self.process_images(MNIST.train_images_file)
			
			# Serialize image data on disk for next time.
			p.dump(images, open(os.path.join(self.path, MNIST.train_images_pickle), 'wb'))
		else:
			# Load image data from disk if it has already been processed.
			print('Loading training images from serialized object file.\n')
			images = p.load(open(os.path.join(self.path, MNIST.train_images_pickle), 'rb'))
		
		if not os.path.isfile(os.path.join(self.path, MNIST.train_labels_pickle)):
			# Download training labels if they aren't on disk.
			print('Downloading training labels.\n')
			self.download(MNIST.train_labels_url, MNIST.train_labels_file)
			labels = self.process_labels(MNIST.train_labels_file)
			
			# Serialize label data on disk for next time.
			p.dump(labels, open(os.path.join(self.path, MNIST.train_labels_pickle), 'wb'))
		else:
			# Load label data from disk if it has already been processed.
			print('Loading training labels from serialized object file.\n')
			labels = p.load(open(os.path.join(self.path, MNIST.train_labels_pickle), 'rb'))
		
		return torch.Tensor(images), torch.Tensor(labels)

	def get_test(self):
		'''
		Gets the MNIST test images and labels.

		Returns:
		
			| :code:`images` (:code:`torch.Tensor`): The MNIST test images.
			| :code:`labels` (:code:`torch.Tensor`): The MNIST test labels.
		'''
		if not os.path.isfile(os.path.join(self.path, MNIST.test_images_pickle)):
			# Download test images if they aren't on disk.
			print('Downloading test images.\n')
			self.download(MNIST.test_images_url, MNIST.test_images_file)
			images = self.process_images(MNIST.test_images_file)
			
			# Serialize image data on disk for next time.
			p.dump(images, open(os.path.join(self.path, MNIST.test_images_pickle), 'wb'))
		else:
			# Load image data from disk if it has already been processed.
			print('Loading test images from serialized object file.\n')
			images = p.load(open(os.path.join(self.path, MNIST.test_images_pickle), 'rb'))
		
		if not os.path.isfile(os.path.join(self.path, MNIST.test_labels_pickle)):
			# Download test labels if they aren't on disk.
			print('Downloading test labels.\n')
			self.download(MNIST.test_labels_url, MNIST.test_labels_file)
			labels = self.process_labels(MNIST.test_labels_file)
			
			# Serialize image data on disk for next time.
			p.dump(labels, open(os.path.join(self.path, MNIST.test_labels_pickle), 'wb'))
		else:
			# Load label data from disk if it has already been processed.
			print('Loading test labels from serialized object file.\n')
			labels = p.load(open(os.path.join(self.path, MNIST.test_labels_pickle), 'rb'))
		
		return torch.Tensor(images), torch.Tensor(labels)
				
	def download(self, url, filename):
		'''
		Downloads and unzips an MNIST data file.
		
		Inputs:
		
			| :code:`url` (:code:`str`): The URL of the data file to be downloaded.
			| :code:`filename` (:code:`str`): The name of the file to save the downloaded data to.
		'''
		data = urlretrieve(url, os.path.join(self.path, filename + '.gz'))
		with gzip.open(os.path.join(self.path, filename + '.gz'), 'rb') as _in:
			with open(os.path.join(self.path, filename), 'wb') as _out:
				shutil.copyfileobj(_in, _out)
	
	def process_images(self, filename):
		'''
		Opens a file of MNIST images and processes them into numpy arrays.
		
		Inputs:
		
			| :code:`filename` (:code:`str`): Name of the file containing MNIST images to load.
		
		Returns:
		
			| (:code:`numpy.ndarray`): A numpy array of shape :code:`[n_images, 28,
				28]`, where :code:`n_images` refers to the number of images in the file.
		'''
		filename = os.path.join(self.path, filename)
		data = open(filename, 'rb')
		
		# Get metadata for images.
		data.read(4)
		n_images = unpack('>I', data.read(4))[0]
		rows = unpack('>I', data.read(4))[0]
		cols = unpack('>I', data.read(4))[0]
		
		images = np.zeros((n_images, rows, cols), dtype=np.uint8)
		
		print('\nProcessing images.\n')

		for i in range(n_images):
			if i % 1000 == 0:
				print('Progress: %d / %d' % (i, n_images))
			
			images[i] = [[unpack('>B', data.read(1))[0] for _ in range(cols)] for _ in range(rows)]
			
		print('Progress: %d / %d' % (n_images, n_images))

		return images
	
	def process_labels(self, filename):
		'''
		Opens a file of MNIST label data and processes it into a numpy vector.
		
		Inputs:
		
			| :code:`filename` (:code:`str`): The name of the file containing MNIST label data.
		
		Returns:
		
			| (:code:`np.ndarray`): A one-dimensional array of shape :code:`(n_labels,)`,
				where :code:`n_labels` refers to the number of labels contained in the file.
		'''
		filename = os.path.join(self.path, filename)
		data = open(filename, 'rb')
		
		# Get metadata for labels.
		data.read(4)
		n_labels = unpack('>I', data.read(4))[0]

		labels = np.zeros(n_labels, dtype=np.uint8)
		
		print('\nProcessing labels.\n')

		for i in range(n_labels):
			if i % 1000 == 0:
				print('Progress: %d / %d' % (i, n_labels))
				
			labels[i] = unpack('>B', data.read(1))[0]

		print('Progress: %d / %d' % (n_labels, n_labels))

		return labels


class CIFAR10:
	'''
	Handles loading and saving of the CIFAR-10 image dataset
	`(link) <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
	'''
	data_directory = 'cifar-10-batches-py'
	data_archive = 'cifar-10-python.tar.gz'
	
	train_pickle = 'train.p'
	test_pickle = 'test.p'
	
	train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
	test_files = ['test_batch']
	
	url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
	
	def __init__(self, path=os.path.join('data', 'CIFAR10')):
		'''
		Constructor for the :code:`CIFAR10` object. Makes the data directory if it doesn't already exist.

		Inputs:
		
			| :code:`path` (:code:`str`): pathname of directory in which to store the CIFAR-10 dataset.
		'''
		if not os.path.isdir(path):
			os.makedirs(path)
		
		self.path = path
		self.data_path = os.path.join(self.path, CIFAR10.data_directory)
		
	def get_train(self):
		'''
		Gets the CIFAR-10 training images and labels.

		Returns:
		
			| :code:`images` (:code:`torch.Tensor`): The CIFAR-10 training images.
			| :code:`labels` (:code:`torch.Tensor`): The CIFAR-10 training labels.
		'''
		if not os.path.isdir(os.path.join(self.path, CIFAR10.data_directory)):
			# Download data if it isn't on disk.
			print('Downloading CIFAR-10 data.\n')
			self.download(CIFAR10.url, CIFAR10.data_archive)
			images, labels = self.process_data(CIFAR10.train_files)

			# Serialize image data on disk for next time.
			p.dump((images, labels), open(os.path.join(self.path, CIFAR10.train_pickle), 'wb'))
		else:
			if not os.path.isdir(os.path.join(self.path, CIFAR10.train_pickle)):
				# Process image and label data if pickled file doesn't exist.
				images, labels = self.process_data(CIFAR10.train_files)
				
				# Serialize image data on disk for next time.
				p.dump((images, labels), open(os.path.join(self.path, CIFAR10.train_pickle), 'wb'))
			else:
				# Load image data from disk if it has already been processed.
				print('Loading training images from serialized object file.\n')
				images, labels = p.load(open(os.path.join(self.path, CIFAR10.train_pickle), 'rb'))
			
		return torch.Tensor(images), torch.Tensor(labels)

	def get_test(self):
		'''
		Gets the CIFAR-10 test images and labels.

		Returns:
		
			| :code:`images` (:code:`torch.Tensor`): The CIFAR-10 test images.
			| :code:`labels` (:code:`torch.Tensor`): The CIFAR-10 test labels.
		'''
		if not os.path.isdir(os.path.join(self.path, CIFAR10.data_directory)):
			# Download data if it isn't on disk.
			print('Downloading CIFAR-10 data.\n')
			self.download(CIFAR10.url, CIFAR10.data_archive)
			images, labels = self.process_data(CIFAR10.test_files)

			# Serialize image data on disk for next time.
			p.dump((images, labels), open(os.path.join(self.path, CIFAR10.test_pickle), 'wb'))
		else:
			if not os.path.isdir(os.path.join(self.path, CIFAR10.test_pickle)):
				# Process image and label data if pickled file doesn't exist.
				images, labels = self.process_data(CIFAR10.test_files)
				
				# Serialize image data on disk for next time.
				p.dump((images, labels), open(os.path.join(self.path, CIFAR10.test_pickle), 'wb'))
			else:
				# Load image data from disk if it has already been processed.
				print('Loading test images from serialized object file.\n')
				images, labels = p.load(open(os.path.join(self.path, CIFAR10.test_pickle), 'rb'))
			
		return torch.Tensor(images), torch.Tensor(labels)
				
	def download(self, url, filename):
		'''
		Downloads and unzips all CIFAR-10 data.
		
		Inputs:
		
			| :code:`url` (:code:`str`): The URL of the data archive to be downloaded.
		'''
		data = urlretrieve(url, os.path.join(self.path, filename))
		tar = tarfile.open(os.path.join(self.path, filename), 'r:gz')
		tar.extractall(path=self.path)
		tar.close()
	
	def process_data(self, filenames):
		'''
		Opens files of CIFAR-10 data and processes them into :code:`numpy` arrays.
		
		Inputs:
		
			| :code:`filename` (:code:`str`): Name of the file containing CIFAR-10 images and labels to load.
		
		Returns:
		
			| (:code:`tuple(numpy.ndarray)`): Two :code:`numpy` arrays with image and label data, respectively.
		'''
		d = {'data' : [], 'labels' : []}
		for filename in filenames:
			with open(os.path.join(self.data_path, filename), 'rb') as f:
				temp = p.load(f, encoding='bytes')
				d['data'].append(temp[b'data'].reshape(-1, 3, 32, 32))
				d['labels'].append(temp[b'labels'])
		
		return np.concatenate(d['data']), np.concatenate(d['labels'])


class CIFAR100:
	'''
	Handles loading and saving of the CIFAR-100 image dataset
	`(link) <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
	'''
	data_directory = 'cifar-100-python'
	data_archive = 'cifar-100-python.tar.gz'
	
	train_pickle = 'train.p'
	test_pickle = 'test.p'
	
	train_files = ['train']
	test_files = ['test']
	
	url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'
	
	def __init__(self, path=os.path.join('data', 'CIFAR100')):
		'''
		Constructor for the :code:`CIFAR100` object. Makes the data directory if it doesn't already exist.

		Inputs:
		
			| :code:`path` (:code:`str`): pathname of directory in which to store the CIFAR-10 handwritten digit dataset.
		'''
		if not os.path.isdir(path):
			os.makedirs(path)
		
		self.path = path
		self.data_path = os.path.join(self.path, CIFAR100.data_directory)
		
	def get_train(self):
		'''
		Gets the MNIST training images and labels.

		Returns:
		
			| :code:`images` (:code:`torch.Tensor`): The CIFAR-100 training images.
			| :code:`labels` (:code:`torch.Tensor`): The CIFAR-100 training labels.
		'''
		if not os.path.isdir(os.path.join(self.path, CIFAR100.data_directory)):
			# Download data if it isn't on disk.
			print('Downloading CIFAR-100 data.\n')
			self.download(CIFAR100.url, CIFAR100.data_archive)
			images, labels = self.process_data(CIFAR100.train_files)

			# Serialize image data on disk for next time.
			p.dump((images, labels), open(os.path.join(self.path, CIFAR100.train_pickle), 'wb'))
		else:
			if not os.path.isdir(os.path.join(self.path, CIFAR100.train_pickle)):
				# Process image and label data if pickled file doesn't exist.
				images, labels = self.process_data(CIFAR100.train_files)
				
				# Serialize image data on disk for next time.
				p.dump((images, labels), open(os.path.join(self.path, CIFAR100.train_pickle), 'wb'))
			else:
				# Load image data from disk if it has already been processed.
				print('Loading training images from serialized object file.\n')
				images, labels = p.load(open(os.path.join(self.path, CIFAR100.train_pickle), 'rb'))
			
		return torch.Tensor(images), torch.Tensor(labels)

	def get_test(self):
		'''
		Gets the CIFAR-100 test images and labels.

		Returns:
		
			| :code:`images` (:code:`torch.Tensor`): The MNIST test images.
			| :code:`labels` (:code:`torch.Tensor`): The MNIST test labels.
		'''
		if not os.path.isdir(os.path.join(self.path, CIFAR100.data_directory)):
			# Download data if it isn't on disk.
			print('Downloading CIFAR-100 data.\n')
			self.download(CIFAR100.url, CIFAR100.data_archive)
			images, labels = self.process_data(CIFAR100.test_files)

			# Serialize image data on disk for next time.
			p.dump((images, labels), open(os.path.join(self.path, CIFAR100.test_pickle), 'wb'))
		else:
			if not os.path.isdir(os.path.join(self.path, CIFAR10.test_pickle)):
				# Process image and label data if pickled file doesn't exist.
				images, labels = self.process_data(CIFAR100.test_files)
				
				# Serialize image data on disk for next time.
				p.dump((images, labels), open(os.path.join(self.path, CIFAR100.test_pickle), 'wb'))
			else:
				# Load image data from disk if it has already been processed.
				print('Loading test images from serialized object file.\n')
				images, labels = p.load(open(os.path.join(self.path, CIFAR100.test_pickle), 'rb'))
			
		return torch.Tensor(images), torch.Tensor(labels)
				
	def download(self, url, filename):
		'''
		Downloads and unzips all CIFAR-100 data.
		
		Inputs:
		
			| :code:`url` (:code:`str`): The URL of the data archive to be downloaded.
		'''
		data = urlretrieve(url, os.path.join(self.path, filename))
		tar = tarfile.open(os.path.join(self.path, filename), 'r:gz')
		tar.extractall(path=self.path)
		tar.close()
	
	def process_data(self, filenames):
		'''
		Opens files of CIFAR-100 data and processes them into :code:`numpy` arrays.
		
		Inputs:
		
			| :code:`filename` (:code:`str`): Name of the file containing CIFAR-100 images and labels to load.
		
		Returns:
		
			| (:code:`tuple(numpy.ndarray)`): Two :code:`numpy` arrays with image and label data, respectively.
		'''
		d = {'data' : [], 'labels' : []}
		for filename in filenames:
			with open(os.path.join(self.data_path, filename), 'rb') as f:
				temp = p.load(f, encoding='bytes')
				d['data'].append(temp[b'data'].reshape(-1, 3, 32, 32))
				d['labels'].append(temp[b'fine_labels'])
		
		return np.concatenate(d['data']), np.concatenate(d['labels'])