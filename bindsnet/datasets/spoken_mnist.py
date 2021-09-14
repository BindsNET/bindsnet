import os
import shutil
import zipfile
from typing import Iterable, List, Tuple
from urllib.request import urlretrieve

import numpy as np
import torch
from scipy.io import wavfile


class SpokenMNIST(torch.utils.data.Dataset):
    # language=rst
    """
    Handles loading and saving of the Spoken MNIST audio dataset `(link)
    <https://github.com/Jakobovski/free-spoken-digit-dataset>`_.
    """
    train_pickle = "train.pt"
    test_pickle = "test.pt"

    url = "https://github.com/Jakobovski/free-spoken-digit-dataset/archive/master.zip"

    files = []
    for digit in range(10):
        for speaker in ["jackson", "nicolas", "theo"]:
            for example in range(50):
                files.append("_".join([str(digit), speaker, str(example)]) + ".wav")

    n_files = len(files)

    def __init__(
        self,
        path: str,
        download: bool = False,
        shuffle: bool = True,
        train: bool = True,
        split: float = 0.8,
        num_samples: int = -1,
    ) -> None:
        # language=rst
        """
        Constructor for the ``SpokenMNIST`` object. Makes the data directory if it
        doesn't already exist.

        :param path: Pathname of directory in which to store the dataset.
        :param download: Whether or not to download the dataset (requires internet
            connection).
        :param shuffle: Whether to randomly permute order of dataset.
        :param train: Load training split if true else load test split
        :param split: Train, test split; in range ``(0, 1)``.
        :param num_samples: Number of samples to pass to the batch
        """
        super().__init__()

        if not os.path.isdir(path):
            os.makedirs(path)

        self.path = path
        self.download = download
        self.shuffle = shuffle

        self.zip_path = os.path.join(path, "repo.zip")

        if train:
            self.audio, self.labels = self._get_train(split)
        else:
            self.audio, self.labels = self._get_test(split)

        self.num_samples = num_samples

    def __len__(self):
        return len(self.audio)

    def __getitem__(self, ind):
        audio = self.audio[ind][: self.num_samples, :]
        label = self.labels[ind]

        return {"audio": audio, "label": label}

    def _get_train(self, split: float = 0.8) -> Tuple[torch.Tensor, torch.Tensor]:
        # language=rst
        """
        Gets the Spoken MNIST training audio and labels.

        :param split: Train, test split; in range ``(0, 1)``.
        :return: Spoken MNIST training audio and labels.
        """
        split_index = int(split * SpokenMNIST.n_files)
        path = os.path.join(self.path, "_".join([SpokenMNIST.train_pickle, str(split)]))

        if not all([os.path.isfile(os.path.join(self.path, f)) for f in self.files]):
            # Download data if it isn't on disk.
            if self.download:
                print("Downloading Spoken MNIST data.\n")
                self._download()

                # Process data into audio, label (input, output) pairs.
                audio, labels = self.process_data(SpokenMNIST.files[:split_index])

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, "wb"))
            else:
                msg = (
                    "Dataset not found on disk; specify 'download=True' to allow"
                    " downloads."
                )
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                audio, labels = self.process_data(SpokenMNIST.files)

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, "wb"))
            else:
                # Load image data from disk if it has already been processed.
                print("Loading training data from serialized object file.\n")
                audio, labels = torch.load(open(path, "rb"))

        labels = torch.Tensor(labels)

        if self.shuffle:
            perm = np.random.permutation(np.arange(labels.shape[0]))
            audio, labels = [torch.Tensor(audio[_]) for _ in perm], labels[perm]

        return audio, torch.Tensor(labels)

    def _get_test(self, split: float = 0.8) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        # language=rst
        """
        Gets the Spoken MNIST training audio and labels.

        :param split: Train, test split; in range ``(0, 1)``.
        :return: The Spoken MNIST test audio and labels.
        """
        split_index = int(split * SpokenMNIST.n_files)
        path = os.path.join(self.path, "_".join([SpokenMNIST.test_pickle, str(split)]))

        if not all([os.path.isfile(os.path.join(self.path, f)) for f in self.files]):
            # Download data if it isn't on disk.
            if self.download:
                print("Downloading Spoken MNIST data.\n")
                self._download()

                # Process data into audio, label (input, output) pairs.
                audio, labels = self.process_data(SpokenMNIST.files[split_index:])

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, "wb"))
            else:
                msg = (
                    "Dataset not found on disk; specify 'download=True' to allow"
                    " downloads."
                )
                raise FileNotFoundError(msg)
        else:
            if not os.path.isdir(path):
                # Process image and label data if pickled file doesn't exist.
                audio, labels = self.process_data(SpokenMNIST.files)

                # Serialize image data on disk for next time.
                torch.save((audio, labels), open(path, "wb"))
            else:
                # Load image data from disk if it has already been processed.
                print("Loading test data from serialized object file.\n")
                audio, labels = torch.load(open(path, "rb"))

        labels = torch.Tensor(labels)

        if self.shuffle:
            perm = np.random.permutation(np.arange(labels.shape[0]))
            audio, labels = audio[perm], labels[perm]

        return audio, torch.Tensor(labels)

    def _download(self) -> None:
        # language=rst
        """
        Downloads and unzips all Spoken MNIST data.
        """
        urlretrieve(SpokenMNIST.url, self.zip_path)

        z = zipfile.ZipFile(self.zip_path, "r")
        z.extractall(path=self.path)
        z.close()

        path = os.path.join(self.path, "free-spoken-digit-dataset-master", "recordings")
        for f in os.listdir(path):
            shutil.move(os.path.join(path, f), os.path.join(self.path))

        cwd = os.getcwd()
        os.chdir(self.path)
        shutil.rmtree("free-spoken-digit-dataset-master")
        os.chdir(cwd)

    def process_data(
        self, file_names: Iterable[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        # language=rst
        """
        Opens files of Spoken MNIST data and processes them into ``numpy`` arrays.

        :param file_names: Names of the files containing Spoken MNIST audio to load.
        :return: Processed Spoken MNIST audio and label data.
        """
        audio, labels = [], []

        for f in file_names:
            label = int(f.split("_")[0])

            sample_rate, signal = wavfile.read(os.path.join(self.path, f))
            pre_emphasis = 0.97
            emphasized_signal = np.append(
                signal[0], signal[1:] - pre_emphasis * signal[:-1]
            )

            # Popular settings are 25 ms for the frame size and a 10 ms stride (15 ms overlap)
            frame_size = 0.025
            frame_stride = 0.01

            # Convert from seconds to samples
            frame_length, frame_step = (
                frame_size * sample_rate,
                frame_stride * sample_rate,
            )
            signal_length = len(emphasized_signal)
            frame_length = int(round(frame_length))
            frame_step = int(round(frame_step))

            # Make sure that we have at least 1 frame
            num_frames = int(
                np.ceil(float(np.abs(signal_length - frame_length)) / frame_step)
            )

            pad_signal_length = num_frames * frame_step + frame_length
            z = np.zeros((pad_signal_length - signal_length))
            pad_signal = np.append(emphasized_signal, z)  # Pad signal

            indices = (
                np.tile(np.arange(0, frame_length), (num_frames, 1))
                + np.tile(
                    np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)
                ).T
            )
            frames = pad_signal[indices.astype(np.int32, copy=False)]

            # Hamming Window
            frames *= np.hamming(frame_length)

            # Fast Fourier Transform and Power Spectrum
            NFFT = 512
            mag_frames = np.absolute(np.fft.rfft(frames, NFFT))  # Magnitude of the FFT
            pow_frames = (1.0 / NFFT) * (mag_frames ** 2)  # Power Spectrum

            # Log filter banks
            nfilt = 40
            low_freq_mel = 0
            high_freq_mel = 2595 * np.log10(
                1 + (sample_rate / 2) / 700
            )  # Convert Hz to Mel
            mel_points = np.linspace(
                low_freq_mel, high_freq_mel, nfilt + 2
            )  # Equally spaced in Mel scale
            hz_points = 700 * (10 ** (mel_points / 2595) - 1)  # Convert Mel to Hz
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
            filter_banks = np.where(
                filter_banks == 0, np.finfo(float).eps, filter_banks
            )  # Numerical Stability
            filter_banks = 20 * np.log10(filter_banks)  # dB

            audio.append(filter_banks), labels.append(label)

        return audio, torch.Tensor(labels)
