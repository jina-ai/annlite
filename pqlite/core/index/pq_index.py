from typing import Optional, List
import numpy as np
from jina.math.helper import top_k
from .base_index import BaseIndex
from ..codec.pq import PQCodec
from ...enums import Metric


class PQIndex(BaseIndex):
    def __init__(self,
                 dim: int,
                 pq_codec: PQCodec,
                 dtype: str = 'float32',
                 metric: Metric = Metric.EUCLIDEAN,
                 **kwargs):
        assert pq_codec is not None
        self._dense_dim = dim
        self._dense_dtype = dtype
        super(PQIndex, self).__init__(pq_codec.dim, dtype=pq_codec.code_dtype, metric=metric, **kwargs)
        self._pq_codec = pq_codec

    def add_with_ids(self, x: np.ndarray, ids: List[int]):
        x = self._pq_codec.encode(x)
        super(PQIndex, self).add_with_ids(x, ids)

    def search(self, x: np.ndarray, limit: int = 10, indices: Optional[np.ndarray] = None):
        _dim = x.shape[-1]
        assert _dim == self.dim, f'the query embedding dimension does not match with index dimension: {_dim} vs {self.dim}'

        x = x.reshape((-1, self.dim))

        precomputed = self._pq_codec.precompute_adc(x)

        codes = self._data
        data_idx = np.arange(self._capacity)

        if indices is not None:
            codes = self._data[indices]
            data_idx = data_idx[indices]

        dists = precomputed.adist(codes)  # (10000, )
        dists = np.expand_dims(dists, axis=0)

        dists, ids = top_k(dists, limit, descending=False)

        # TODO: change the shape of return
        ids = ids[0]
        if indices is not None:
            ids = data_idx[ids]

        return dists[0], ids


