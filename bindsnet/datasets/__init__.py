import os
import sys
import gzip
import torch
import urllib
import shutil
import pickle as p
import numpy  as np
import scipy.io.wavfile
from scipy.fftpack import dct

from struct         import unpack
from urllib.request import urlretrieve


class SpokenMNIST:
	'''
	Data is divided by an 80-20 split into train and test
	'''
	def __init__(self, path=None):
		self.data_dir = '/home/darpan/sem4/free-spoken-digit-dataset/recordings/'
		# self.data_dir = '/mnt/nfs/work1/rkozma/dsanghavi/bindsnet/data/spokenMNIST/recordings/'
		if path:
			self.data_dir = path
		self.files = [f for f in os.listdir(self.data_dir) if os.path.isfile(os.path.join(self.data_dir, f)) and '.wav' in f]
		np.random.shuffle(self.files)
		split = int(0.8*len(self.files))
		self.train_files = self.files[:split]
		self.test_files = self.files[split:]
	
	def get_train(self):
		'''
		Gets the SpokenMNIST training log filter banks and labels.

		Returns:
			List of variable length audios: The spoken MNIST training log filter banks. Each element in the list is of shape (T_i, 40)
			(torch.Tensor or torch.cuda.Tensor) labels: The MNIST training labels.
		'''
		audios = []
		labels = []
		for f in self.train_files:
			filter_banks, label = self.pre_process(f)
			audios.append(torch.Tensor(filter_banks))
			labels.append(label)
		return audios, torch.Tensor(labels)

	def get_test(self):
		'''
		Gets the SpokenMNIST testing log filter banks and labels.

		Returns:
			List of variable length audios: The spoken MNIST testing log filter banks.
			(torch.Tensor or torch.cuda.Tensor) labels: The MNIST training labels.
		'''
		audios = []
		labels = []
		for f in self.test_files:
			filter_banks, label = self.pre_process(f)
			audios.append(filter_banks)
			labels.append(label)
		return audios, torch.Tensor(labels)

	def pre_process(self, file):
		'''
		Returns the 40 dim log filter banks
		'''
		label = int(file[0])

		sample_rate, signal = scipy.io.wavfile.read(os.path.join(self.data_dir,file)) 
		pre_emphasis = 0.97
		emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])
		# Popular settings are 25 ms for the frame size and a 10 ms stride (15 ms overlap)
		frame_size = 0.025
		frame_stride = 0.01
		frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
		signal_length = len(emphasized_signal)
		frame_length = int(round(frame_length))
		frame_step = int(round(frame_step))
		num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

		pad_signal_length = num_frames * frame_step + frame_length
		z = np.zeros((pad_signal_length - signal_length))
		pad_signal = np.append(emphasized_signal, z) # Pad Signal to make sure that all frames have equal number of samples without truncating any samples from the original signal

		indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
		frames = pad_signal[indices.astype(np.int32, copy=False)]

		# Hamming Window
		frames *= np.hamming(frame_length)

		# Fast Fourier Transform and Power Spectrum
		NFFT = 512
		mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
		pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum

		# Log filter banks
		nfilt = 40 
		low_freq_mel = 0
		high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))  # Convert Hz to Mel
		mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
		hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
		bin = np.floor((NFFT + 1) * hz_points / sample_rate)

		fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
		for m in range(1, nfilt + 1):
		    f_m_minus = int(bin[m - 1])   # left
		    f_m = int(bin[m])             # center
		    f_m_plus = int(bin[m + 1])    # right

		    for k in range(f_m_minus, f_m):
		        fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
		    for k in range(f_m, f_m_plus):
		        fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
		filter_banks = np.dot(pow_frames, fbank.T)
		filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
		filter_banks = 20 * np.log10(filter_banks)  # dB

		return filter_banks, label

	


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
		'''
		Constructor for the MNIST object. Makes the data directory if it doesn't already exist.

		Inputs:
			path (str): pathname of directory in which to store the MNIST handwritten digit dataset.
		'''
		if not os.path.isdir(path):
			os.makedirs(path)
		
		self.path = path
		
	def get_train(self):
		'''
		Gets the MNIST training images and labels.

		Returns:
			(torch.Tensor or torch.cuda.Tensor) images: The MNIST training images.
			(torch.Tensor or torch.cuda.Tensor) labels: The MNIST training labels.
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
			(torch.Tensor or torch.cuda.Tensor) images: The MNIST test images.
			(torch.Tensor or torch.cuda.Tensor) labels: The MNIST test labels.
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