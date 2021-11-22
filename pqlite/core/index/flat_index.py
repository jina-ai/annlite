from typing import Optional, List
import numpy as np
from jina.math.distance import cdist
from jina.math.helper import top_k
from .base import BaseIndex


class FlatIndex(BaseIndex):
    def __init__(self, *args, **kwargs):
        super(FlatIndex, self).__init__(*args, **kwargs)
        self._data = np.zeros((self.initial_size, self.dim), dtype=self.dtype)

    def search(
        self, x: np.ndarray, limit: int = 10, indices: Optional[np.ndarray] = None
    ):
        _dim = x.shape[-1]
        assert (
            _dim == self.dim
        ), f'the query embedding dimension does not match with index dimension: {_dim} vs {self.dim}'

        x = x.reshape((-1, self.dim))

        data = self._data
        data_ids = np.arange(self.capacity)

        if indices is not None:
            data = self._data[indices]
            data_ids = data_ids[indices]

        dists = cdist(x, data, metric=self.metric.name.lower())
        dists, idx = top_k(dists, limit, descending=False)

        # TODO: change the shape of return
        ids = ids[0]
        if indices is not None:
            ids = data_idx[ids]
        return dists[0], ids

    def add_with_ids(self, x: np.ndarray, ids: List[int]):
        for idx in ids:
            if idx >= self._capacity:
                self._expand_capacity()

        start = self._size
        end = start + len(x)

        self._data[ids, :] = x
        self._size = end

    def _expand_capacity(self):
        new_block = np.zeros((self.expand_step_size, self.dim), dtype=self.dtype)
        self._data = np.concatenate((self._data, new_block), axis=0)

        self._capacity += self.expand_step_size
        logger.debug(
            f'=> total storage capacity is expanded by {self.expand_step_size}',
        )

    def reset(self):
        pass

    def add_with_ids(self, x: np.ndarray, ids: List[int], **kwargs):
        pass

    def delete(self, ids: List[int]):
        pass

    def update(self, x: np.ndarray, ids: List[int], **kwargs):
        pass
