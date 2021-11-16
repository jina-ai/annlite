from typing import List, Optional
import numpy as np
from pqlite.hnsw_bind import Index
from ....enums import Metric
from ....helper import str2dtype


class PQIndex():
    def __init__(self,
                 dim: int,
                 dtype: str = 'float32',
                 metric: Metric = Metric.EUCLIDEAN,
                 **kwargs):
        self.dim = dim
        self.dtype = str2dtype(dtype)
        self.metric = metric

        self._index = Index(space='l2', dim=dim)

    def add_with_ids(self, x: np.ndarray, ids: List[int]):
        self._index.add_items(x, ids=ids)

    def search(self, query: np.ndarray, limit: int = 10, indices: Optional[np.ndarray] = None):
        raise NotImplementedError
