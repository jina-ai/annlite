from typing import Optional
import numpy as np
from jina.math.distance import cdist
from jina.math.helper import top_k
from .base_index import BaseIndex


class FlatIndex(BaseIndex):
    def __init__(self, dim: int, dtype: str = 'float32', **kwargs):
        super().__init__(dim, dtype=dtype, **kwargs)

    def search(
        self, query: np.ndarray, limit: int = 10, indices: Optional[np.ndarray] = None
    ):
        assert query.shape == (self.dim,)
        query = query.reshape((1, -1))

        data = self._data
        data_idx = np.arange(self._capacity)

        if indices is not None:
            data = self._data[indices]
            data_idx = data_idx[indices]

        dists = cdist(query, data, metric=self.metric.name.lower())
        dists, ids = top_k(dists, limit, descending=False)
        ids = ids[0]
        if indices is not None:
            ids = data_idx[ids]
        return dists[0], ids
