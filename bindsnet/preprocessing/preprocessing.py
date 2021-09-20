import hashlib
import os
import pickle
from abc import ABC, abstractmethod

import torch


class AbstractPreprocessor(ABC):
    # language=rst
    """
    Abstract base class for Preprocessor.
    """

    def process(
        self,
        csvfile: str,
        use_cache: bool = True,
        cachedfile: str = "./processed/data.pt",
    ) -> torch.tensor:
        # cache dictionary for storing encodings if previously encoded
        cache = {"verify": "", "data": None}

        # if the file exists
        if use_cache:
            # generate a hash
            cache["verify"] = self.__gen_hash(csvfile)

            # compare hash, if valid return cached value
            if self.__check_file(cachedfile, cache):
                return cache["data"]

        # otherwise process the data
        self._process(csvfile, cache)

        # save if use_cache
        if use_cache:
            self.__save(cachedfile, cache)

        # return data
        return cache["data"]

    @abstractmethod
    def _process(self, filename: str, cache: dict):
        # language=rst
        """
        Method for defining how to preprocess the data.

        :param filename: File to load raw data from.
        :param cache: Dictionary for caching 'data' needs to be updated for caching to
            work.
        """

    def __gen_hash(self, filename: str) -> str:
        # language=rst
        """
        Generates an hash for a csv file and the preprocessor name.

        :param filename: File to generate hash for.
        :return: Hash for the csv file.
        """
        # read all the lines
        with open(filename, "r") as f:
            lines = f.readlines()

        # generate md5 hash after concatenating all of the lines
        pre = "".join(lines) + str(self.__class__.__name__)
        m = hashlib.md5(pre.encode("utf-8"))
        return m.hexdigest()

    @staticmethod
    def __check_file(cachedfile: str, cache: dict) -> bool:
        # language=rst
        """
        Compares the csv file and the saved file to see if a new encoding needs to be
        generated.

        :param cachedfile: The filename of the cached data.
        :param cache: Dictionary containing the current csv file hash. This is updated
            if the cache file has valid data.
        :return: Whether the cache is valid.
        """
        # try opening the cached file
        try:
            with open(cachedfile, "rb") as f:
                temp = pickle.load(f)
        except FileNotFoundError:
            temp = {"verify": "", "data": None}

        # if the hash matches up, keep the data from the cache
        if cache["verify"] == temp["verify"]:
            cache["data"] = temp["data"]
            return True

        # otherwise don't do anything
        return False

    @staticmethod
    def __save(filename: str, data: dict) -> None:
        # language=rst
        """
        Creates or overwrites existing encoding file.

        :param filename: Filename to save to.
        """
        # if the directories in path don't exist create them
        if not os.path.exists(os.path.dirname(filename)):
            os.makedirs(os.path.dirname(filename), exist_ok=True)

        # save file
        with open(filename, "wb") as f:
            pickle.dump(data, f)
