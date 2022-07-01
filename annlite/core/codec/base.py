import pickle
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from pathlib import Path
    from typing import Tuple

    import numpy as np


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


class BaseTrainedPQ(ABC):
    """Interface for a trained PQ.
    If you want to use the product quantization in HNSW backend, the PQ class you pass should
    satisfy this interface .
    """

    @abstractmethod
    def get_subspace_splitting(self) -> 'Tuple[int]':
        """Return subspace splitting setting

        :return: tuple of (`n_subvectors`, `n_clusters`, `d_subvector`)
        """
        pass

    @abstractmethod
    def encode(self, x: 'np.ndarray') -> 'np.ndarray':
        """Quantize `x` into integer arrays

        :param x: Input vectors with shape(`N`, `n_subvectors`*`d_subvector`), and dtype=np.float32.
        :return: PQ codes with shape=(`N`, `n_subvectors`), integer dtype, each element < `n_clusters`.
        """
        pass

    @abstractmethod
    def get_codebook(self) -> 'np.ndarray':
        """Return the codebook parameters.

        Expect a 3-dimensional matrix is returned,
        with shape (`n_subvectors`, `n_clusters`, `d_subvector`) and dtype float32
        """
        pass
