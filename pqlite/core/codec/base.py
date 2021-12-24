import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


class BaseCodec(ABC):
    def __init__(self, require_train: bool = True):
        self.require_train = require_train
        self._is_trained = False if require_train else True

    @abstractmethod
    def fit(self, *args, **kwargs):
        pass

    @abstractmethod
    def encode(self):
        pass

    @abstractmethod
    def decode(self):
        pass

    def dump(self, target_path: 'Path'):
        pickle.dump(self, target_path.open('wb'), protocol=4)

    @staticmethod
    def load(from_path: 'Path'):
        return pickle.load(from_path.open('rb'))

    @property
    def is_trained(self):
        return self._is_trained

    def _check_trained(self):
        assert self.is_trained is True, f'{self.__class__.__name__} requires training'
