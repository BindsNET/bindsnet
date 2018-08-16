import hashlib
import itertools
import os
import math
import torch
import pickle

import numpy as np
import pandas as pd
from pyproj import Proj
from typing import Tuple
from abc import abstractmethod, ABC
from random import getrandbits, seed, randint
from tqdm import tqdm


class AbstractEncoder(ABC):
    # language=rst
    """
    Abstract base class for Encoder.
    """

    def __init__(self, csvfile: str, save=False, encodingfile='./encodings/encoding.p') -> None:
        # language=rst
        """
        Abstract constructor for the Encoder class.
        :param csvfile: File to encode
        :param save: Whether to save the encoding to file.
        :param encodingfile: Where to save encoding to. (./encodings/encoding.p) by default
        """
        self.data = pd.read_csv(csvfile)
        self.__save_file = save
        self.__enc = {}
        if save:
            self.__enc['verify'] = self.__gen_hash(csvfile)
            self.file = encodingfile
            if os.path.exists(self.file):
                self.__check_file()

    @abstractmethod
    def _encode(self) -> torch.Tensor:
        # language=rst
        """
        Method for defining how encoding is done from the csv file.
         :return: Generated encoding
        """
        pass

    @staticmethod
    def __gen_hash(filename: str) -> str:
        # language=rst
        """
        Generates an hash for a csv file.
        :param filename: file to generate hash for
         :return: hash for the csv file
        """
        with open(filename, 'r') as f:
            lines = f.readlines()
        m = hashlib.md5(''.join(lines).encode('utf-8'))
        return m.hexdigest()

    def __check_file(self) -> None:
        # language=rst
        """
        Compares the csv file and the save encoding to see if a new encoding needs to be generated.
        """
        try:
            with open(self.file, 'rb') as f:
                temp = pickle.load(f)
        except FileNotFoundError:
            temp = {
                'verify': ''
            }
        if self.__enc['verify'] == temp['verify']:
            self.__enc = temp

    def __save(self) -> None:
        # language=rst
        """
        Creates/Overwrites existing encoding file
        """
        if not os.path.exists(os.path.dirname(self.file)):
            os.makedirs(os.path.dirname(self.file), exist_ok=True)
        with open(self.file, 'wb') as f:
            pickle.dump(self.__enc, f)

    def get_encoding(self) -> torch.tensor:
        # language=rst
        """
        Returns encoding for the csv file
        :return: Encoding for specified csv file
        """
        if 'encoding' in self.__enc and self.__enc['encoding'] is not None:
            return self.__enc['encoding']
        self.__enc['encoding'] = self._encode()
        if self.__save_file:
            self.__save()
        return self.__enc['encoding']


class NumentaEncoder(AbstractEncoder):
    def __init__(self, csvfile: str, scale=5, w=21, n=1000, timestep=10, save=False,
                 encodingfile='./encodings/encoding.p') -> None:
        super().__init__(csvfile, save, encodingfile)
        # language=rst
        """
        Numenta Encoder for geospatial data as decribed here: http://chetansurpur.com/slides/2014/8/5/geospatial-encoder.html
        
        The csv file is expected to have three columns with headers: `speed,latitude,longitude`
        
        :param csvfile: File to encode
        :param save: Whether to save the encoding to file.
        :param encodingfile: Where to save encoding to. (./encodings/encoding.p) by default
        :param scale: how much to zoom in
        :param w: number of neighbor square to choose
        :param n: length of output binary vector for each encoding step
        :param timestep: used to determine radius for considering neighbors
        """
        self.w = w
        self.n = n
        self.scale = scale
        self.timestep = timestep
        self.__map = Proj(init="epsg:3785")  # Spherical Mercator

    def _encode(self) -> torch.tensor:
        # language=rst
        """
        Numenta encoding as described here: http://chetansurpur.com/slides/2014/8/5/geospatial-encoder.html
        :return: Encoding
        """

        speeds = self.data['speed'].tolist()
        latitudes = self.data['latitude'].tolist()
        longitudes = self.data['longitude'].tolist()

        values = []
        for speed, latitude, longitude in tqdm(zip(speeds, latitudes, longitudes), unit='datapt'):
            output = torch.zeros(self.n)
            self.__generate_vector((speed, latitude, longitude), output)

            values.append(output.tolist())

        return torch.tensor(values)

    def __hash_coordinate(self, latitude: float, longitude: float) -> int:
        # language=rst
        """
        Returns the hash for a given coordinate
        :param latitude: latitude value
        :param longitude: longitude value
        :return: integer value to be used as a seed for the coordinate
        """
        coordainte_str = (str(latitude) + ',' + str(longitude)).encode('utf-8')
        m = hashlib.md5(coordainte_str)
        return int(int(m.hexdigest(), 16) % (2 ** 64))

    def __coordinate_order(self, latitude: float, longitude: float) -> int:
        # language=rst
        """
        Returns the order `w` for a given coordinate
        :param latitude: latitude value
        :param longitude: longitude value
        :return: integer value to be used as a seed for the coordinate
        """
        seed(self.__hash_coordinate(latitude, longitude))
        return getrandbits(64)

    def __coordinate_bit(self, latitude: float, longitude: float) -> int:
        # language=rst
        """
        Returns the bit in the output vector for given coordinate
        :param latitude: latitude value
        :param longitude: longitude value
        :return: bit index in the output vector
        """
        seed(self.__hash_coordinate(latitude, longitude))
        return randint(0, self.n)

    def __map_transform(self, latitude: float, longitude: float) -> Tuple[int, int]:
        # language=rst
        """
        Returns the bit in the output vector for given coordinate
        :param latitude: latitude value
        :param longitude: longitude value
        :return: transforms the input coordinates to spherical mercator coordinates
        """
        longitude, latitude = self.__map(longitude, latitude)
        return int(latitude / self.scale), int(longitude / self.scale)

    def __radius(self, speed: float) -> int:
        # language=rst
        """
        Returns the radius for a given speed
        :param speed: speed value in meters per second
        :return: radius
        """
        overlap = 1.5
        coordinates = speed * self.timestep / self.scale
        radius = int(round(float(coordinates) / 2 * overlap))
        min_radius = int(math.ceil((math.sqrt(self.w) - 1) / 2))
        return max(radius, min_radius)

    def __neighbors(self, latitude: int, longitude: int, radius: int) -> np.ndarray:
        # language=rst
        """
        Generates an ndarray of neighbors for a given coordinate and radius
        :param latitude: latitude value
        :param longitude: longitude value
        :param radius: radius to consider order values
        :return:
        """
        ranges = (range(n - radius, n + radius + 1) for n in [latitude, longitude])
        return np.array(list(itertools.product(*ranges)))

    def __select(self, neighbors: np.ndarray) -> np.ndarray:
        # language=rst
        """
        Selects the top `w` neighbors
        :param neighbors: neighbors to consider
        :return: top `w` neighbors
        """
        orders = np.array([self.__coordinate_order(n[0], n[1]) for n in neighbors])
        indices = np.argsort(orders)[-self.w:]
        return np.array(neighbors[indices])

    def __generate_vector(self, data_point: Tuple[float, float, float], output: torch.tensor) -> torch.tensor:
        # language=rst
        """
        Generates a vector of length `n` for a single data point
        :param data_point: Tuple: (speed, latitude, longitude)
        :param output: binary vector containing
        :return: tensor containing binary values
        """
        speed, latitude, longitude = data_point
        latitude, longitude = self.__map_transform(latitude, longitude)
        radius = self.__radius(speed)

        neighbors = self.__neighbors(latitude, longitude, radius)

        top = self.__select(neighbors)
        indices = np.array([self.__coordinate_bit(w[0], w[1]) for w in top])

        output[:] = 0
        output[indices] = 1

        return output
