import hashlib
import csv
import os
import torch
import pickle


class AbstractEncoder:
    def __init__(self, csvfile: str, save=False, file='./encodings/encoding.p') -> None:
        self.csvfile = csvfile
        self.save = save
        self.enc = {}
        if save:
            self.file = file
            if not os.path.exists(os.path.dirname(file)):
                os.makedirs(os.path.dirname(file), exist_ok=True)
            self._check_file()

    def _gen_hash(self) -> str:
        with open(self.csvfile, 'r') as f:
            lines = f.readlines()
        m = hashlib.md5(''.join(lines).encode('utf-8'))
        return m.hexdigest()

    def _check_file(self):
        self.enc['verify'] = self._gen_hash()
        try:
            with open(self.file, 'rb') as f:
                temp = pickle.load(f)
        except FileNotFoundError:
            temp = {
                'verify': ''
            }
        if 'verify' in self.enc and self.enc['verify'] == temp['verify']:
            self.enc = temp

    def _save(self):
        with open(self.file, 'wb') as f:
            pickle.dump(self.enc, f)

    def _encode(self) -> torch.Tensor:
        pass

    def get_encoding(self) -> torch.tensor:
        if 'encoding' in self.enc and self.enc['encoding'] is not None:
            return self.enc['encoding']
        self.enc['encoding'] = self._encode()
        if self.save:
            self._save()
        return self.enc['encoding']


class NumentaEncoder(AbstractEncoder):
    def __init__(self, csvfile: str, save=False, file='./encodings/encoding.p'):
        super().__init__(csvfile, save, file)

    def _encode(self):
        return torch.ones(1000)