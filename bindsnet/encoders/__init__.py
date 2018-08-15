import hashlib
import csv
import os
from abc import abstractmethod, ABC

import torch
import pickle


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
        :param download: Where to save encoding to. (./encodings/encoding.p) by default
        """
        self.__csvfile = csvfile
        self.__save_file = save
        self.__enc = {}
        if save:
            self.file = encodingfile
            if not os.path.exists(os.path.dirname(encodingfile)):
                os.makedirs(os.path.dirname(encodingfile), exist_ok=True)
            self.__check_file()

    @abstractmethod
    def _encode(self) -> torch.Tensor:
        # language=rst
        """
        Abstract method for defining how encoding is done from the csv file.
         :return: Generated encoding
        """
        pass

    def __gen_hash(self) -> str:
        # language=rst
        """
        Generates an md5 hash for the csv file.
         :return: md5 hash for the csv file
        """
        with open(self.__csvfile, 'r') as f:
            lines = f.readlines()
        m = hashlib.md5(''.join(lines).encode('utf-8'))
        return m.hexdigest()

    def __check_file(self) -> None:
        # language=rst
        """
        Compares the csv file and the save encoding to see if a new encoding needs to be generated.
        """
        self.__enc['verify'] = self.__gen_hash()
        try:
            with open(self.file, 'rb') as f:
                temp = pickle.load(f)
        except FileNotFoundError:
            temp = {
                'verify': ''
            }
        if 'verify' in self.__enc and self.__enc['verify'] == temp['verify']:
            self.__enc = temp

    def __save(self) -> None:
        # language=rst
        """
        Creates/Overwrites existing encoding file
        """
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
    def __init__(self, csvfile: str, save=False, encodingfile='./encodings/encoding.p'):
        super().__init__(csvfile, save, encodingfile)

    def _encode(self):
        return torch.ones(1000)
