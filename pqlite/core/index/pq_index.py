import numpy as np
from .base_index import BaseIndex
from ..codec.pq import PQCodec


class PQIndex(BaseIndex):
    def __init__(self, pq_codec: PQCodec, **kwargs):

        super().__init__(pq_codec.dim, dtype=pq_codec.code_dtype, **kwargs)
        self._pq_codec = pq_codec

    def search(self, query: np.ndarray, limit: int = 10):
        raise NotImplementedError
