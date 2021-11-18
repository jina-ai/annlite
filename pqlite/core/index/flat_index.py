from typing import Optional
import numpy as np
from jina.math.distance import cdist
from jina.math.helper import top_k
from .base_index import BaseIndex


class FlatIndex(BaseIndex):
    def search(
        self, x: np.ndarray, limit: int = 10, indices: Optional[np.ndarray] = None
    ):
        _dim = x.shape[-1]
        assert _dim == self.dim, f'the query embedding dimension does not match with index dimension: {_dim} vs {self.dim}'

        x = x.reshape((-1, self.dim))

        data = self._data
        data_idx = np.arange(self._capacity)

        if indices is not None:
            data = self._data[indices]
            data_idx = data_idx[indices]

        dists = cdist(x, data, metric=self.metric.name.lower())
        dists, ids = top_k(dists, limit, descending=False)

        # TODO: change the shape of return
        ids = ids[0]
        if indices is not None:
            ids = data_idx[ids]
        return dists[0], ids
