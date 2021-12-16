import math
from typing import List, Optional, Union

import numpy as np
from loguru import logger

from pqlite.hnsw_bind import Index

from ....enums import Metric
from ..base import BaseIndex


class HnswIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        dtype: np.dtype = np.float32,
        metric: Metric = Metric.COSINE,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_connection: int = 16,
        **kwargs,
    ):
        """
        :param dim: The dimensionality of vectors to index
        :param metric: Distance metric type, can be 'euclidean', 'inner_product', or 'cosine'
        :param ef_construction: the size of the dynamic list for the nearest neighbors (used during the building).
        :param ef_search: the size of the dynamic list for the nearest neighbors (used during the search).
        :param max_connection: The number of bi-directional links created for every new element during construction.
                    Reasonable range for M is 2-100.
        """
        super().__init__(dim, dtype=dtype, metric=metric, **kwargs)

        self.ef_construction = ef_construction
        self.ef_search = ef_search
        self.max_connection = max_connection

        self._init_hnsw_index()

    def _init_hnsw_index(self):
        self._index = Index(space=self.space_name, dim=self.dim)
        self._index.init_index(
            max_elements=self.capacity,
            ef_construction=self.ef_construction,
            M=self.max_connection,
        )
        self._index.set_ef(self.ef_search)

    def add_with_ids(self, x: np.ndarray, ids: List[int]):
        max_id = max(ids) + 1
        if max_id > self.capacity:
            expand_steps = math.ceil(max_id / self.expand_step_size)
            self._expand_capacity(expand_steps * self.expand_step_size)

        self._index.add_items(x, ids=ids)

    def search(
        self, query: np.ndarray, limit: int = 10, indices: Optional[np.ndarray] = None
    ):
        _dim = query.shape[-1]
        assert (
            _dim == self.dim
        ), f'the query embedding dimension does not match with index dimension: {_dim} vs {self.dim}'

        query = query.reshape((-1, self.dim))

        ef_search = max(self.ef_search, limit)
        self._index.set_ef(ef_search)

        if indices is not None:
            # TODO: add a smart strategy to speed up this case (bruteforce search would be better)
            if len(indices) < limit:
                limit = len(indices)
            ids, dists = self._index.knn_query_with_filter(
                query, filters=indices, k=limit
            )
        else:
            ids, dists = self._index.knn_query(query, k=limit)
        return dists[0], ids[0]

    def delete(self, ids: List[int]):
        raise RuntimeError(
            f'the deletion operation is not allowed for {self.__class__.__name__}!'
        )

    def update_with_ids(self, x: np.ndarray, ids: List[int], **kwargs):
        raise RuntimeError(
            f'the update operation is not allowed for {self.__class__.__name__}!'
        )

    def _expand_capacity(self, new_capacity: int):
        self._capacity = new_capacity
        self._index.resize_index(new_capacity)
        logger.debug(
            f'HNSW index capacity is expanded by {self.expand_step_size}',
        )

    def reset(self, capacity: Optional[int] = None):
        super().reset(capacity=capacity)
        self._init_hnsw_index()

    @property
    def size(self):
        return self._index.element_count

    @property
    def space_name(self):
        if self.metric == Metric.EUCLIDEAN:
            return 'l2'
        elif self.metric == Metric.INNER_PRODUCT:
            return 'ip'
        return 'cosine'
