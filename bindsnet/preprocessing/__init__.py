import hashlib
import itertools
import math
import os
import pickle
import torch
import numpy as np
import pandas as pd

from abc import abstractmethod, ABC
from pyproj import Proj
from random import getrandbits, randint, seed
from tqdm import tqdm
from typing import Tuple


class AbstractPreprocessor(ABC):
    # language=rst
    """
    Abstract base class for Preprocessor.
    """

    def process(self, csvfile: str, use_cache: bool = True, cachedfile: str = './processed/data.p') -> torch.tensor:
        # cache dictionary for storing encodings if previously encoded
        cache = {'verify': '', 'data': None}

        # if the file exists
        if use_cache:
            # generate a hash
            cache['verify'] = self.__gen_hash(csvfile)

            # compare hash, if valid return cached value
            if self.__check_file(cachedfile, cache):
                return cache['data']

        # otherwise process the data
        self._process(csvfile, cache)

        # save if use_cache
        if use_cache:
            self.__save(cachedfile, cache)

        # return data
        return cache['data']

    @abstractmethod
    def _process(self, filename: str, cache: dict):
        # language=rst
        """
        Method for defining how to preprocess the data.
         :param filename: file to load raw data from
         :param cache: dict for caching 'data' needs to be updated for caching to work
        """
        pass

    def __gen_hash(self, filename: str) -> str:
        # language=rst
        """
        Generates an hash for a csv file and the preprocessor name
        :param filename: file to generate hash for
         :return: hash for the csv file
        """
        # read all the lines
        with open(filename, 'r') as f:
            lines = f.readlines()
        # generate md5 hash after concatenating all of the lines
        pre = ''.join(lines) + str(self.__class__.__name__)
        m = hashlib.md5(pre.encode('utf-8'))
        return m.hexdigest()

    @staticmethod
    def __check_file(cachedfile: str, cache: dict) -> bool:
        # language=rst
        """
        Compares the csv file and the saved file to see if a new encoding needs to be generated.
        :param cachedfile: the filename of the cached data
        :param cache: dict containing the current csvfile hash. This is updated if the cachefile has valid data
        :return: whether the cache is valid

        """
        # try opening the cached file
        try:
            with open(cachedfile, 'rb') as f:
                temp = pickle.load(f)
        except FileNotFoundError:
            temp = {
                'verify': '', 'data': None,
            }

        # if the hash matches up, keep the data from the cache
        if cache['verify'] == temp['verify']:
            cache['data'] = temp['data']
            return True

        # otherwise don't do anything
        return False

    @staticmethod
    def __save(filename: str, data: dict) -> None:
        # language=rst
        """
        Creates/Overwrites existing encoding file
        :param filename: filename to save to
        """
        # if the directories in path don't exist create them
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        # save file
        with open(filename, 'wb') as f:
            pickle.dump(data, f)


class NumentaPreprocessor(AbstractPreprocessor):
    def __init__(self, scale=5, w=21, n=1000, timestep=10) -> None:
        # language=rst
        """
        Numenta Encoder for geospatial data as decribed here: http://chetansurpur.com/slides/2014/8/5/geospatial-encoder.html

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

    def _process(self, filename: str, cache: dict) -> None:
        # language=rst
        """
        Numenta encoding as described here: http://chetansurpur.com/slides/2014/8/5/geospatial-encoder.html

        The csv file is expected to have three columns with headers: `speed,latitude,longitude`
        :param filename: csv file containing raw data
        :param cache: dict containing 'data'
        """
        data = pd.read_csv(filename)
        speeds = data['speed'].tolist()
        latitudes = data['latitude'].tolist()
        longitudes = data['longitude'].tolist()

        values = []
        for speed, latitude, longitude in tqdm(zip(speeds, latitudes, longitudes), unit='Entry'):
            output = torch.zeros(self.n)
            self.__generate_vector((speed, latitude, longitude), output)

            values.append(output.tolist())

        cache['data'] = torch.tensor(values)

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
        return randint(0, self.n - 1)

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
