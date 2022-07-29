import math
import os.path
from argparse import ArgumentError
from functools import wraps
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np
from loguru import logger

from annlite.hnsw_bind import Index

from ....enums import Metric
from ....math import EPS
from ..base import BaseIndex

if TYPE_CHECKING:  # pragma: no cover
    from ...codec.base import BaseCodec


def pre_process(f):
    def dim_check_reshape(x: np.ndarray, dim):
        _dim = x.shape[-1]
        assert (
            _dim == dim
        ), f'the query embedding dimension does not match with index dimension: {_dim} vs {dim}'
        if len(x.shape) not in [1, 2]:
            raise ArgumentError('data embedding must be a 1d/2d array')
        elif len(x.shape) == 1:
            return x.reshape(1, -1)
        else:
            return x

    def l2_normalize(x: np.ndarray):
        return x / (np.sum(x**2, 1) ** 0.5 + EPS).reshape(-1, 1)

    @wraps(f)
    def pre_processed(self: 'HnswIndex', x: np.ndarray, *args, **kwargs):
        x = dim_check_reshape(x, self.dim)
        if self.normalization_enable and not self.pq_enable:
            x = l2_normalize(x)
        elif self.pq_enable:
            if not self.pq_codec.is_trained:
                raise RuntimeError(
                    'Please train the PQ before using HNSW quantization backend'
                )
            elif not self._set_backend_pq:
                self._index.loadPQ(self.pq_codec)
                self._set_backend_pq = True
            x = self.pq_codec.encode(x)
        return f(self, x, *args, **kwargs)

    return pre_processed


class HnswIndex(BaseIndex):
    def __init__(
        self,
        dim: int,
        dtype: np.dtype = np.float32,
        metric: Metric = Metric.COSINE,
        ef_construction: int = 200,
        ef_search: int = 50,
        max_connection: int = 16,
        pq_codec: Optional['BaseCodec'] = None,
        index_file: Optional[Union[str, Path]] = None,
        **kwargs,
    ):
        """
        :param dim: The dimensionality of vectors to index
        :param index_file: A file-like object or a string containing a file name.
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
        self.pq_codec = pq_codec
        self._set_backend_pq = False
        self.index_file = index_file

        self._init_hnsw_index()

    def _init_hnsw_index(self):
        self._index = Index(space=self.space_name, dim=self.dim)
        if self.index_file and os.path.exists(self.index_file):
            logger.info(
                f'indexer will be loaded from {self.index_file}',
            )
            self.load_index(self.index_file)
        else:
            if self.index_file:
                raise FileNotFoundError(
                    f'index path: {self.index_file} does not exist',
                )
            if self.pq_codec is not None and self.pq_codec.is_trained:
                self._index.init_index(
                    max_elements=self.capacity,
                    ef_construction=self.ef_construction,
                    M=self.max_connection,
                    pq_codec=self.pq_codec,
                )
                self._set_backend_pq = True
            else:
                self._index.init_index(
                    max_elements=self.capacity,
                    ef_construction=self.ef_construction,
                    M=self.max_connection,
                    pq_codec=None,
                )
                self._set_backend_pq = False
        self._index.set_ef(self.ef_search)

    def load_index(self, index_file: Union[str, Path]):
        self._index.load_index(index_file)

    def save_index(self, index_file: Union[str, Path]):
        self._index.save_index(index_file)

    @pre_process
    def add_with_ids(self, x: 'np.ndarray', ids: List[int]):
        max_id = max(ids) + 1
        if max_id > self.capacity:
            expand_steps = math.ceil(max_id / self.expand_step_size)
            self._expand_capacity(expand_steps * self.expand_step_size)

        self._index.add_items(x, ids=ids)

    @pre_process
    def search(
        self,
        query: 'np.ndarray',
        limit: int = 10,
        indices: Optional['np.ndarray'] = None,
    ):
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

    def update_with_ids(self, x: 'np.ndarray', ids: List[int], **kwargs):
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

    @property
    def pq_enable(self):
        return self.pq_codec is not None

    @property
    def normalization_enable(self):
        return self.metric == Metric.COSINE
