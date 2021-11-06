from typing import Optional, List

import numpy as np
from jina import DocumentArray
from jina.math.distance import cdist
from jina.math.helper import top_k
from loguru import logger

from .core import VQCodec, PQCodec
from .storage import CellStorage


class PQLite(CellStorage):
    """:class:`PQLite` is an implementation of IVF-PQ being with equipped with SQLite.

    To create a :class:`PQLite` object, simply:

        .. highlight:: python
        .. code-block:: python
            pqlite = PQLite(d_vector=256, metric='euclidean')

    :param d_vector: the dimensionality of input vectors. there are 2 constraints on d_vector:
            (1) it needs to be divisible by n_subvectors; (2) it needs to be a multiple of 4.*
    :param n_subvectors: number of subquantizers, essentially this is the byte size of
            each quantized vector, default is 8.
    :param n_cells:  number of coarse quantizer clusters.
    :param initial_size: initial capacity assigned to each voronoi cell of coarse quantizer.
            ``n_cells * initial_size`` is the number of vectors that can be stored initially.
            if any cell has reached its capacity, that cell will be automatically expanded.
            If you need to add vectors frequently, a larger value for init_size is recommended.
    :param args: Additional positional arguments which are just used for the parent initialization
    :param kwargs: Additional keyword arguments which are just used for the parent initialization

    .. note::
        Remember that the shape of any tensor that contains data points has to be `[n_data, d_vector]`.
    """

    def __init__(
        self,
        d_vector: int,
        n_subvectors: int = 8,
        n_cells: int = 8,
        initial_size: Optional[int] = None,
        expand_step_size: int = 1024,
        metric: str = 'euclidean',
        use_residual: bool = False,
        columns: Optional[List[tuple]] = None,
        *args,
        **kwargs,
    ):
        assert (
            d_vector % n_subvectors == 0
        ), '"d_vector" needs to be divisible by "n_subvectors"'

        super(PQLite, self).__init__(
            code_size=n_subvectors,
            n_cells=n_cells,
            dtype='uint8',
            initial_size=initial_size,
            expand_step_size=expand_step_size,
            columns=columns,
        )

        self.d_vector = d_vector
        self.n_subvectors = n_subvectors
        self.d_subvector = d_vector // n_subvectors
        self.metric = metric
        self.use_residual = use_residual
        self.n_probe = 5

        # if use_residual and (n_cells * 256 * n_subvectors * 4) <= 4 * 1024 ** 3:
        #     self._use_precomputed = True
        # else:
        #     self._use_precomputed = False

        self._use_smart_probing = True
        self._smart_probing_temperature = 30.0

        self.vq_codec = VQCodec(n_cells, metric=metric)
        self.pq_codec = PQCodec(
            d_vector, n_subvectors=n_subvectors, n_clusters=256, metric=metric
        )

    def _sanity_check(self, x: 'np.ndarray'):
        assert len(x.shape) == 2
        assert x.shape[1] == self.d_vector

        return x.shape

    def fit(self, x: 'np.ndarray', force_retrain: bool = False):
        n_data, d_vector = self._sanity_check(x)

        logger.info(f'=> start training VQ codec with {n_data} data...')
        self.vq_codec.fit(x)

        logger.info(f'=> start training PQ codec with {n_data} data...')
        self.pq_codec.fit(x)

        logger.info(f'=> pqlite is successfully trained!')

    def add(
        self, docs: DocumentArray
    ):
        """

        :param docs: The documents to index
        :return:
        """

        x = docs.embeddings

        n_data, _ = self._sanity_check(x)

        assigned_cells = self.vq_codec.encode(x)
        quantized_x = self.encode(x)

        return super(PQLite, self).insert(
            quantized_x, assigned_cells, docs
        )

    def update(
        self,
        docs: DocumentArray
    ):
        """

        :param docs: the documents to update
        :return:
        """
        x = docs.embeddings
        n_data, _ = self._sanity_check(x)

        assigned_cells = self.vq_codec.encode(x)
        quantized_x = self.encode(x)

        return super(PQLite, self).update(
            quantized_x, assigned_cells, docs=docs
        )

    def ivfpq_topk(
        self,
        precomputed,
        cells: 'np.ndarray',
        conditions: Optional[list] = None,
        limit: int = 10,
    ):
        topk_sims = []
        topk_ids = []
        for cell_id in cells:
            indices = []
            doc_ids = []
            for idx, doc in self.cell_table(cell_id).query(conditions=conditions):
                indices.append(idx)
                doc_ids.append(doc.id)

            if len(indices) == 0:
                continue

            indices = np.array(indices, dtype=np.int64)

            doc_ids = np.array(doc_ids, dtype=self._doc_id_dtype)
            doc_ids = np.expand_dims(doc_ids, axis=0)
            codes = self.vecs_storage[cell_id][indices]

            dists = precomputed.adist(codes)  # (10000, )
            dists = np.expand_dims(dists, axis=0)

            _topk_sims, indices = top_k(dists, k=limit)
            _topk_ids = np.take_along_axis(doc_ids, indices, axis=1)

            topk_sims.append(_topk_sims)
            topk_ids.append(_topk_ids)

        topk_sims = np.hstack(topk_sims)
        topk_ids = np.hstack(topk_ids)

        idx = topk_sims.argsort(axis=1)[:, :limit]
        topk_sims = np.take_along_axis(topk_sims, idx, axis=1)
        topk_ids = np.take_along_axis(topk_ids, idx, axis=1)
        return topk_sims, topk_ids

    def search_cells(
        self,
        query: 'np.ndarray',
        cells: 'np.ndarray',
        conditions: Optional[list] = None,
        limit: int = 10,
    ):
        topk_dists, topk_ids = [], []
        for x, cell_idx in zip(query, cells):
            precomputed = self.pq_codec.precompute_adc(x)
            dist, ids = self.ivfpq_topk(
                precomputed, cells=cell_idx, conditions=conditions, limit=limit
            )

            topk_dists.append(dist)
            topk_ids.append(ids)

        topk_dists = np.concatenate(topk_dists, axis=0)
        topk_ids = np.concatenate(topk_ids, axis=0)

        return topk_dists, topk_ids

    def search(self, docs: DocumentArray, conditions: Optional[list] = [], limit: int = 10):
        query = docs.embeddings
        n_data, _ = self._sanity_check(query)
        assert 0 < limit <= 1024

        vq_codebook = self.vq_codec.codebook

        # find n_probe closest cells
        dists = cdist(query, vq_codebook, metric=self.metric)
        dists, cells = top_k(dists, k=self.n_probe)

        # if self.use_smart_probing and self.n_probe > 1:
        #     p = -topk_sims.abs().sqrt()
        #     p = torch.softmax(p / self.smart_probing_temperature, dim=-1)
        #
        #     # p_norm = p.norm(dim=-1)
        #     # sqrt_d = self.n_probe ** 0.5
        #     # score = 1 - (p_norm * sqrt_d - 1) / (sqrt_d - 1) - 1e-6
        #     # n_probe_list = torch.ceil(score * (self.n_probe) ).long()
        #
        #     max_n_probe = torch.tensor(self.n_probe, device=self.device)
        #     normalized_entropy = - torch.sum(p * torch.log2(p) / torch.log2(max_n_probe), dim=-1)
        #     n_probe_list = torch.ceil(normalized_entropy * max_n_probe).long()
        # else:
        #     n_probe_list = None
        #

        return self.search_cells(
            query=query,
            cells=cells,
            conditions=conditions,
            limit=limit,
        )

    def encode(self, x: 'np.ndarray'):
        n_data, _ = self._sanity_check(x)
        y = self.pq_codec.encode(x)
        return y

    def decode(self, x):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_subvectors
        return self.pq_codec.decode(x)

    @property
    def use_smart_probing(self):
        return self._use_smart_probing

    @use_smart_probing.setter
    def use_smart_probing(self, value):
        assert type(value) is bool
        self._use_smart_probing = value

    @property
    def smart_probing_temperature(self):
        return self._smart_probing_temperature

    @smart_probing_temperature.setter
    def smart_probing_temperature(self, value):
        assert value > 0
        assert self.use_smart_probing, 'set use_smart_probing to True first'
        self._smart_probing_temperature = value
