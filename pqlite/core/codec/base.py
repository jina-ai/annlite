from abc import ABC, abstractmethod


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

    @property
    def is_trained(self):
        return self.is_trained

    def _check_trained(self):
        assert self.is_trained is True, 'Codec is untrained'
