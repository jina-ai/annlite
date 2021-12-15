from typing import List, Optional

import numpy as np
from docarray.math.distance import cdist
from docarray.math.helper import top_k
from loguru import logger

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

        data = self._data[: self.size]
        data_ids = np.arange(self.size)

        if indices is not None:
            data = self._data[indices]
            data_ids = data_ids[indices]

        dists = cdist(x, data, metric=self.metric.name.lower())
        dists, idx = top_k(dists, limit, descending=False)

        # TODO: change the shape of return
        dists = dists[0]
        data_ids = data_ids[idx[0]]

        return dists, data_ids

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
            f'total storage capacity is expanded by {self.expand_step_size}',
        )

    def reset(self, capacity: Optional[int] = None):
        super().reset(capacity=capacity)
        self._data = np.zeros((self.capacity, self.dim), dtype=self.dtype)

    def delete(self, ids: List[int]):
        raise RuntimeError(
            f'the deletion operation is not allowed for {self.__class__.__name__}!'
        )

    def update_with_ids(self, x: np.ndarray, ids: List[int], **kwargs):
        self._data[ids, :] = x
