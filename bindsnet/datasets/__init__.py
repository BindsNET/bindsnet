import gzip
import torch
import urllib
import shutil
import os, sys
import pickle as p
import numpy  as np

from struct         import unpack
from urllib.request import urlretrieve


def get_poisson(data, time):
	'''
	Generates Poisson spike trains based on input intensity. Inputs must be
	non-negative. Spike inter-arrival times are inversely proportional to
	input magnitude, so data must be scaled according to desired spike frequency.
    
    Inputs:
        data (torch.Tensor or torch.cuda.Tensor): Tensor of shape [n_samples, n_1,
            ..., n_k], with arbitrarily dimensionality [n_1, ..., n_k].
        time (int): Length of Poisson spike train per input variable.
    
    Yields:
        (torch.Tensor or torch.cuda.Tensor): Tensors with shape [time,
            n_1, ..., n_k], with Poisson-distributed spikes parametrized by the
            data.
	'''
	n_samples = data.shape[0]  # Number of samples
	
	data = np.copy(data)
	
	for i in range(n_samples):
		# Get i-th datum.
		datum = data[i]
		shape, size = datum.shape, datum.size
		datum = datum.ravel()

		# Invert inputs (input intensity inversely
		# proportional to spike inter-arrival time).
		datum[datum != 0] = 1 / datum[datum != 0]

		# Make the spike data.
		s_times = np.random.poisson(datum, [time, size])
		s_times = np.cumsum(s_times, axis=0)
		s_times[s_times >= time] = 0

		# Create spike trains from spike times.
		s = np.zeros([time, size])
		for idx in range(time):
			s[s_times[idx], np.arange(size)] = 1

		s[0, :] = 0
		s = s.reshape([time, *shape])

		# Yield Poisson-distributed spike trains.
		yield s

		
class MNIST:
	'''
	Handles loading and saving of the MNIST handwritten digits
	(http://yann.lecun.com/exdb/mnist/).
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
		if not os.path.isdir(path):
			os.makedirs(path)
		
		self.path = path
		
	def get_train(self):
		'''
		Gets the MNIST training images and labels.

		Returns:
			(numpy.ndarray) images: The MNIST training images.
			(numpy.ndarray) labels: The MNIST training labels.
		'''
		if not os.path.isfile(os.path.join(self.path, MNIST.train_images_pickle)):
			# Download training images if they aren't on disk.
			self.download(MNIST.train_images_url, MNIST.train_images_file)
			images = self.process_images(MNIST.train_images_file)
			
			# Serialize image data on disk for next time.
			p.dump(images, open(os.path.join(self.path, MNIST.train_images_pickle), 'wb'))
		else:
			# Load image data from disk if it has already been processed.
			print('Loading images from serialized object file.\n')
			images = p.load(open(os.path.join(self.path, MNIST.train_images_pickle), 'rb'))
		
		if not os.path.isfile(os.path.join(self.path, MNIST.train_labels_pickle)):
			# Download training labels if they aren't on disk.
			self.download(MNIST.train_labels_url, MNIST.train_labels_file)
			labels = self.process_labels(MNIST.train_labels_file)
			
			# Serialize label data on disk for next time.
			p.dump(labels, open(os.path.join(self.path, MNIST.train_labels_pickle), 'wb'))
		else:
			# Load label data from disk if it has already been processed.
			print('Loading labels from serialized object file.\n')
			labels = p.load(open(os.path.join(self.path, MNIST.train_labels_pickle), 'rb'))
		
		return images, labels

	def get_test(self):
		'''
		Gets the MNIST test images and labels.

		Returns:
			(numpy.ndarray) images: The MNIST test images.
			(numpy.ndarray) labels: The MNIST test labels.
		'''
		if not os.path.isfile(os.path.join(self.path, MNIST.test_images_pickle)):
			# Download test images if they aren't on disk.
			self.download(MNIST.test_images_url, MNIST.test_images_file)
			images = self.process_images(MNIST.test_images_file)
			
			# Serialize image data on disk for next time.
			p.dump(images, open(os.path.join(self.path, MNIST.test_images_pickle), 'wb'))
		else:
			# Load image data from disk if it has already been processed.
			print('Loading images from serialized object file.\n')
			images = p.load(open(os.path.join(self.path, MNIST.test_images_pickle), 'rb'))
		
		if not os.path.isfile(os.path.join(self.path, MNIST.test_labels_pickle)):
			# Download test labels if they aren't on disk.
			self.download(MNIST.test_labels_url, MNIST.test_labels_file)
			labels = self.process_labels(MNIST.test_labels_file)
			
			# Serialize image data on disk for next time.
			p.dump(labels, open(os.path.join(self.path, MNIST.test_labels_pickle), 'wb'))
		else:
			# Load label data from disk if it has already been processed.
			print('Loading labels from serialized object file.\n')
			labels = p.load(open(os.path.join(self.path, MNIST.test_labels_pickle), 'rb'))
		
		return images, labels
				
	def download(self, url, filename):
		'''
		Downloads and unzips an MNIST data file.
		
		Inputs:
			url (str): The URL of the data file to be downloaded.
			filename (str): The name of the file to save the downloaded data to.
		'''
		data = urlretrieve(url, os.path.join(self.path, filename + '.gz'))
		with gzip.open(os.path.join(self.path, filename + '.gz'), 'rb') as _in:
			with open(os.path.join(self.path, filename), 'wb') as _out:
				shutil.copyfileobj(_in, _out)
	
	def process_images(self, filename):
		'''
		Opens a file of MNIST images and processes them into numpy arrays.
		
		Inputs:
			filename (str): Name of the file containing MNIST images to load.
		
		Returns:
			(numpy.ndarray): A numpy array of shape [n_images, 28, 28],
				where n_images refers to the number of images in the file.
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
			filename (str): The name of the file containing MNIST label data.
		
		Returns:
			(np.ndarray): A one-dimensional array of shape (n_labels,), where
				n_labels refers to the number of labels contained in the file.
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