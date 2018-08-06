import os
import gzip
import torch
import shutil
import tarfile
import zipfile
import numpy as np
import pickle as p

from struct import unpack
from scipy.io import wavfile
from abc import ABC, abstractmethod
from urllib.request import urlretrieve
from typing import Tuple, List, Iterable, Any

__all__ = [
    'Dataset', 'MNIST', 'SpokenMNIST', 'CIFAR10', 'CIFAR100', 'preprocess'
]


class Dataset(ABC):
    # language=rst
    """
    Abstract base class for dataset.
    """

    def __init__(self, path: str='.', download: bool=False) -> None:
        # language=rst
        """
        Abstract constructor for the Dataset class.

        :param path: Pathname of directory in which to store the dataset.
        :param download: Whether or not to download the dataset (requires internet connection).
        """
        if not os.path.isdir(path):
            os.makedirs(path)

        self.path = path
        self.download = download

    @abstractmethod
    def get_train(self) -> Tuple[Any, ...]:
        # language=rst
        """
        Abstract method stub for fetching training data from a dataset.
        """
        pass

    @abstractmethod
    def get_test(self) -> Tuple[Any, ...]:
        # language=rst
        """
        Abstract method stub for fetching test data from a dataset.
        """


class MNIST(Dataset):
    # language=rst
    """
    Handles loading and saving of the MNIST handwritten digits `(link) <http://yann.lecun.com/exdb/mnist/>`_.
    """
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

    def __init__(self, path: str=os.path.join('data', 'MNIST'), download: bool=False) -> None:
        # language=rst
        """
        Constructor for the ``MNIST`` object. Makes the data directory if it doesn't already exist.

        :param path: Pathname of directory in which to store the dataset.
        :param download: Whether or not to download the dataset (requires internet connection).
        """
        super().__init__(path, download)

    def get_train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the MNIST training images and labels.

        :return: MNIST training images and labels.
        """
        if not os.path.isfile(os.path.join(self.path, MNIST.train_images_pickle)):
            # Download training images if they aren't on disk.
            if self.download:
                print('Downloading training images.\n')
                self._download(MNIST.train_images_url, MNIST.train_images_file)
                images = self.process_images(MNIST.train_images_file)

                # Serialize image data on disk for next time.
                torch.save(images, open(os.path.join(self.path, MNIST.train_images_pickle), 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            # Load image data from disk if it has already been processed.
            print('Loading training images from serialized object file.\n')
            images = torch.load(open(os.path.join(self.path, MNIST.train_images_pickle), 'rb'))

        if not os.path.isfile(os.path.join(self.path, MNIST.train_labels_pickle)):
            # Download training labels if they aren't on disk.
            if self.download:
                print('Downloading training labels.\n')
                self._download(MNIST.train_labels_url, MNIST.train_labels_file)
                labels = self.process_labels(MNIST.train_labels_file)

                # Serialize label data on disk for next time.
                torch.save(labels, open(os.path.join(self.path, MNIST.train_labels_pickle), 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            # Load label data from disk if it has already been processed.
            print('Loading training labels from serialized object file.\n')
            labels = torch.load(open(os.path.join(self.path, MNIST.train_labels_pickle), 'rb'))

        return torch.Tensor(images), torch.Tensor(labels)

    def get_test(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the MNIST test images and labels.

        :return: MNIST test images and labels.
        """
        if not os.path.isfile(os.path.join(self.path, MNIST.test_images_pickle)):
            # Download test images if they aren't on disk.
            if self.download:
                print('Downloading test images.\n')
                self._download(MNIST.test_images_url, MNIST.test_images_file)
                images = self.process_images(MNIST.test_images_file)

                # Serialize image data on disk for next time.
                torch.save(images, open(os.path.join(self.path, MNIST.test_images_pickle), 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            # Load image data from disk if it has already been processed.
            print('Loading test images from serialized object file.\n')
            images = torch.load(open(os.path.join(self.path, MNIST.test_images_pickle), 'rb'))

        if not os.path.isfile(os.path.join(self.path, MNIST.test_labels_pickle)):
            if self.download:
                # Download test labels if they aren't on disk.
                print('Downloading test labels.\n')
                self._download(MNIST.test_labels_url, MNIST.test_labels_file)
                labels = self.process_labels(MNIST.test_labels_file)

                # Serialize image data on disk for next time.
                torch.save(labels, open(os.path.join(self.path, MNIST.test_labels_pickle), 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            # Load label data from disk if it has already been processed.
            print('Loading test labels from serialized object file.\n')
            labels = torch.load(open(os.path.join(self.path, MNIST.test_labels_pickle), 'rb'))

        return torch.Tensor(images), torch.Tensor(labels)

    def _download(self, url: str, filename: str) -> None:
        # language=rst
        """
         Downloads and unzips an MNIST data file.

        :param url: The URL of the data file to be downloaded.
        :param filename: The name of the file to save the downloaded data to.
        """
        urlretrieve(url, os.path.join(self.path, filename + '.gz'))
        with gzip.open(os.path.join(self.path, filename + '.gz'), 'rb') as _in:
            with open(os.path.join(self.path, filename), 'wb') as _out:
                shutil.copyfileobj(_in, _out)

    def process_images(self, filename: str) -> np.ndarray:
        # language=rst
        """
        Opens a file of MNIST images and processes them into numpy arrays.

        :param filename: Name of the file containing MNIST images to load.
        :return: A numpy array of shape ``[n_images, 28, 28]``, where ``n_images`` is the number of images in the file.
        """
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

        print('Progress: %d / %d\n' % (n_images, n_images))

        return images

    def process_labels(self, filename: str) -> np.ndarray:
        # language=rst
        """
        Opens a file of MNIST label data and processes it into a numpy vector.

        :param filename: The name of the file containing MNIST label data.
        :return: An array of shape ``(n_labels,)``, where ``n_labels`` is the number of labels in the file.
        """
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

        print('Progress: %d / %d\n' % (n_labels, n_labels))

        return labels


class SpokenMNIST(Dataset):
    # language=rst
    """
    Handles loading and saving of the Spoken MNIST audio dataset `(link)
    <https://github.com/Jakobovski/free-spoken-digit-dataset>`_.
    """
    train_pickle = 'train.pt'
    test_pickle = 'test.pt'

    url = 'https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip'

    files = []
    for digit in range(10):
        for speaker in ['jackson', 'nicolas', 'theo']:
            for example in range(50):
                files.append('_'.join([str(digit), speaker, str(example)]) + '.wav')

    n_files = len(files)

    def __init__(self, path: str=os.path.join('data', 'SpokenMNIST'), download: bool=False) -> None:
        # language=rst
        """
        Constructor for the ``SpokenMNIST`` object. Makes the data directory if it doesn't already exist.

        :param path: Pathname of directory in which to store the dataset.
        :param download: Whether or not to download the dataset (requires internet connection).
        """
        super().__init__(path, download)
        self.zip_path = os.path.join(path, 'repo.zip')

    def get_train(self, split: float=0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the Spoken MNIST training audio and labels.

        :param split: Train, test split; in range ``(0, 1)``.
        :return: Spoken MNIST training audio and labels.
        """
        split_index = int(split * SpokenMNIST.n_files)
        path = os.path.join(self.path, '_'.join([SpokenMNIST.train_pickle, str(split)]))

        if not all([os.path.isfile(os.path.join(self.path, f)) for f in self.files]):
            # Download data if it isn't on disk.
            if self.download:
                print('Downloading Spoken MNIST data.\n')
                self._download()

                # Process data into audio, label (input, output) pairs.
                audio, labels = self.process_data(SpokenMNIST.files[:split_index])

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                audio, labels = self.process_data(SpokenMNIST.files)

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, 'wb'))
            else:
                # Load image data from disk if it has already been processed.
                print('Loading training data from serialized object file.\n')
                audio, labels = torch.load(open(path, 'rb'))

        return audio, torch.Tensor(labels)

    def get_test(self, split: float=0.8) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # language=rst
        """
        Gets the Spoken MNIST training audio and labels.

        :param split: Train, test split; in range ``(0, 1)``.
        :return: The Spoken MNIST test audio and labels.
        """
        split_index = int(split * SpokenMNIST.n_files)
        path = os.path.join(self.path, '_'.join([SpokenMNIST.test_pickle, str(split)]))

        if not all([os.path.isfile(os.path.join(self.path, f)) for f in self.files]):
            # Download data if it isn't on disk.
            if self.download:
                print('Downloading Spoken MNIST data.\n')
                self._download()

                # Process data into audio, label (input, output) pairs.
                audio, labels = self.process_data(SpokenMNIST.files[split_index:])

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                audio, labels = self.process_data(SpokenMNIST.files)

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, 'wb'))
            else:
                # Load image data from disk if it has already been processed.
                print('Loading test data from serialized object file.\n')
                audio, labels = torch.load(open(path, 'rb'))

        return audio, torch.Tensor(labels)

    def _download(self) -> None:
        # language=rst
        """
        Downloads and unzips all Spoken MNIST data.
        """
        urlretrieve(SpokenMNIST.url, self.zip_path)

        z = zipfile.ZipFile(self.zip_path, 'r')
        z.extractall(path=self.path)
        z.close()

        path = os.path.join(self.path, 'free-spoken-digit-dataset-master', 'recordings')
        for f in os.listdir(path):
            shutil.move(os.path.join(path, f), os.path.join(self.path))

        cwd = os.getcwd()
        os.chdir(self.path)
        shutil.rmtree('free-spoken-digit-dataset-master')
        os.chdir(cwd)

    def process_data(self, file_names: Iterable[str]) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # language=rst
        """
        Opens files of Spoken MNIST data and processes them into ``numpy`` arrays.

        :param file_names: Names of the files containing Spoken MNIST audio to load.
        :return: Processed Spoken MNIST audio and label data.
        """
        audio, labels = [], []

        for f in file_names:
            label = int(f.split('_')[0])

            sample_rate, signal = wavfile.read(os.path.join(self.path, f))
            pre_emphasis = 0.97
            emphasized_signal = np.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])

            # Popular settings are 25 ms for the frame size and a 10 ms stride (15 ms overlap)
            frame_size = 0.025
            frame_stride = 0.01
            frame_length, frame_step = frame_size * sample_rate, frame_stride * sample_rate  # Convert from seconds to samples
            signal_length = len(emphasized_signal)
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))
            num_frames = int(np.ceil(
                float(np.abs(signal_length - frame_length)) / frame_step))  # Make sure that we have at least 1 frame

            pad_signal_length = num_frames * frame_step + frame_length
            z = np.zeros((pad_signal_length - signal_length))
            pad_signal = np.append(emphasized_signal, z)  # Pad signal

            indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + np.tile(
                np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
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
            hz_points = (700 * (10 ** (mel_points / 2595) - 1))  # Convert Mel to Hz
            bin = np.floor((NFFT + 1) * hz_points / sample_rate)

            fbank = np.zeros((nfilt, int(np.floor(NFFT / 2 + 1))))
            for m in range(1, nfilt + 1):
                f_m_minus = int(bin[m - 1])  # left
                f_m = int(bin[m])  # center
                f_m_plus = int(bin[m + 1])  # right

                for k in range(f_m_minus, f_m):
                    fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
                for k in range(f_m, f_m_plus):
                    fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])

            filter_banks = np.dot(pow_frames, fbank.T)
            filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)  # Numerical Stability
            filter_banks = 20 * np.log10(filter_banks)  # dB

            audio.append(filter_banks), labels.append(label)

        return audio, torch.Tensor(labels)


class CIFAR10(Dataset):
    # language=rst
    """
    Handles loading and saving of the CIFAR-10 image dataset `(link) <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """
    data_directory = 'cifar-10-batches-py'
    data_archive = 'cifar-10-python.tar.gz'

    train_pickle = 'train.pt'
    test_pickle = 'test.pt'

    train_files = ['data_batch_1', 'data_batch_2', 'data_batch_3', 'data_batch_4', 'data_batch_5']
    test_files = ['test_batch']

    url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

    def __init__(self, path: str=os.path.join('data', 'CIFAR10'), download: bool=False) -> None:
        # language=rst
        """
        Constructor for the ``CIFAR10`` object. Makes the data directory if it doesn't already exist.

        :param path: Pathname of directory in which to store the dataset.
        :param download: Whether or not to download the dataset (requires internet connection).
        """
        super().__init__(path, download)
        self.data_path = os.path.join(self.path, CIFAR10.data_directory)

    def get_train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the CIFAR-10 training images and labels.

        :return: CIFAR-10 training images and labels.
        """
        path = os.path.join(self.path, CIFAR10.train_pickle)
        if not os.path.isdir(os.path.join(self.path, CIFAR10.data_directory)):
            # Download data if it isn't on disk.
            if self.download:
                print('Downloading CIFAR-10 data.\n')
                self._download(CIFAR10.url, CIFAR10.data_archive)
                images, labels = self.process_data(CIFAR10.train_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                images, labels = self.process_data(CIFAR10.train_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                # Load image data from disk if it has already been processed.
                print('Loading training images from serialized object file.\n')
                images, labels = torch.load(open(path, 'rb'))

        return images, labels

    def get_test(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the CIFAR-10 test images and labels.

        :return: CIFAR-10 test images and labels.
        """
        path = os.path.join(self.path, CIFAR10.test_pickle)
        if not os.path.isdir(os.path.join(self.path, CIFAR10.data_directory)):
            # Download data if it isn't on disk.
            if self.download:
                print('Downloading CIFAR-10 data.\n')
                self._download(CIFAR10.url, CIFAR10.data_archive)
                images, labels = self.process_data(CIFAR10.test_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                images, labels = self.process_data(CIFAR10.test_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                # Load image data from disk if it has already been processed.
                print('Loading test images from serialized object file.\n')
                images, labels = torch.load(open(path, 'rb'))

        return images, labels

    def _download(self, url: str, file_name: str) -> None:
        # language=rst
        """
        Downloads and unzips all CIFAR-10 data.

        :param url: The URL of the data archive to be downloaded.
        :param file_name: The name of the file to store extract the archive to.
        """
        urlretrieve(url, os.path.join(self.path, file_name))
        tar = tarfile.open(os.path.join(self.path, file_name), 'r:gz')
        tar.extractall(path=self.path)
        tar.close()

    def process_data(self, file_names: Iterable[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Opens files of CIFAR-10 data and processes them into ``torch.Tensors``.
        :param file_names: Name of the file containing CIFAR-10 images and labels to load.
        :return: Processed CIFAR-10 image and label data.
        """
        d = {'data': [], 'labels': []}
        for filename in file_names:
            with open(os.path.join(self.data_path, filename), 'rb') as f:
                temp = p.load(f, encoding='bytes')
                d['data'].append(temp[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                d['labels'].append(temp[b'labels'])

        data, labels = np.concatenate(d['data']), np.concatenate(d['labels'])
        return torch.from_numpy(data).float(), torch.from_numpy(labels)


class CIFAR100(Dataset):
    # language=rst
    """
    Handles loading and saving of the CIFAR-100 image dataset `(link) <https://www.cs.toronto.edu/~kriz/cifar.html>`_.
    """
    data_directory = 'cifar-100-python'
    data_archive = 'cifar-100-python.tar.gz'

    train_pickle = 'train.pt'
    test_pickle = 'test.pt'

    train_files = ['train']
    test_files = ['test']

    url = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

    def __init__(self, path: str=os.path.join('data', 'CIFAR100'), download: bool=False) -> None:
        # language=rst
        """
        Constructor for the ``CIFAR100`` object. Makes the data directory if it doesn't already exist.
        :param path: Pathname of directory in which to store the dataset.
        :param download: Whether or not to download the dataset (requires internet connection).
        """
        super().__init__(path, download)
        self.data_path = os.path.join(self.path, CIFAR100.data_directory)

    def get_train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the CIFAR-100 training images and labels.

        :return: CIFAR-100 training images and labels.
        """
        path = os.path.join(self.path, CIFAR100.train_pickle)
        if not os.path.isdir(os.path.join(self.path, CIFAR100.data_directory)):
            # Download data if it isn't on disk.
            if self.download:
                print('Downloading CIFAR-100 data.\n')
                self._download(CIFAR100.url, CIFAR100.data_archive)
                images, labels = self.process_data(CIFAR100.train_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                images, labels = self.process_data(CIFAR100.train_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                # Load image data from disk if it has already been processed.
                print('Loading training images from serialized object file.\n')
                images, labels = torch.load(open(path, 'rb'))

        return images, labels

    def get_test(self) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the CIFAR-100 test images and labels.

        :return: CIFAR-100 test images and labels.
        """
        path = os.path.join(self.path, CIFAR100.test_pickle)
        if not os.path.isdir(os.path.join(self.path, CIFAR100.data_directory)):
            # Download data if it isn't on disk.
            if self.download:
                print('Downloading CIFAR-100 data.\n')
                self._download(CIFAR100.url, CIFAR100.data_archive)
                images, labels = self.process_data(CIFAR100.test_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                msg = 'Dataset not found on disk; specify \'download=True\' to allow downloads.'
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                images, labels = self.process_data(CIFAR100.test_files)

                # Serialize image data on disk for next time.
                torch.save((images, labels), open(path, 'wb'))
            else:
                # Load image data from disk if it has already been processed.
                print('Loading test images from serialized object file.\n')
                images, labels = torch.load(open(path, 'rb'))

        return images, labels

    def _download(self, url: str, filename: str) -> None:
        # language=rst
        """
        Downloads and unzips all CIFAR-100 data.

        :param url: The URL of the data archive to be downloaded.
        :param file_name: The name of the file to store extract the archive to.
        """
        urlretrieve(url, os.path.join(self.path, filename))
        tar = tarfile.open(os.path.join(self.path, filename), 'r:gz')
        tar.extractall(path=self.path)
        tar.close()

    def process_data(self, file_names: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Opens files of CIFAR-100 data and processes them into :code:`numpy` arrays.

        :param file_names: Name of the file containing CIFAR-100 images and labels to load.
        :return: Processed CIFAR-100 image and label data.
        """
        d = {'data': [], 'labels': []}
        for filename in file_names:
            with open(os.path.join(self.data_path, filename), 'rb') as f:
                temp = p.load(f, encoding='bytes')
                d['data'].append(temp[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1))
                d['labels'].append(temp[b'fine_labels'])

        data, labels = np.concatenate(d['data']), np.concatenate(d['labels'])
        return torch.from_numpy(data).float(), torch.from_numpy(labels)
