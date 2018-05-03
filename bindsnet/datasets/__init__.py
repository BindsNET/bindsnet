from .preprocess    import *

import os
import sys
import gzip
import torch
import urllib
import shutil
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