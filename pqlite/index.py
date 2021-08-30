from typing import Optional, List

import numpy as np
from jina.math.distance import cdist
from jina.math.helper import top_k
from loguru import logger

from .container.cell import CellContainer
from .core.codec import VQCodec, PQCodec


class PQLite(CellContainer):
    """:class:`PQLite` is an implementation of IVF-PQ.

    To create a :class:`PQLite` object, simply:

        .. highlight:: python
        .. code-block:: python
            pqlite = PQLite(d_vector=256, metric='cosine')

    :param n_vector: the dimentionality of input vectors. there are 2 constraints on d_vector:
            (1) it needs to be divisible by n_subvectors; (2) it needs to be a multiple of 4.*
    :param n_subvectors: number of subquantizers, essentially this is the byte size of
            each quantized vector, default is 8.
    :param n_cells:  number of coarse quantizer clusters.
    :param init_size: initial capacity assigned to each voronoi cell of coarse quantizer. ``n_cells * init_size``
            is the number of vectors that can be stored initially. if any cell has reached its capacity, that cell
            will be automatically expanded. If you need to add vectors frequently, a larger value for init_size
            is recommended.
    :param args: Additional positional arguments which are just used for the parent initialization
    :param kwargs: Additional keyword arguments which are just used for the parent initialization

    .. note::
        Remember that the shape of any tensor that contains data points has to be [n_data, d_vector].
    """

    def __init__(
        self,
        d_vector: int,
        n_subvectors: int = 8,
        n_cells: int = 64,
        initial_size: Optional[int] = None,
        expand_step_size: int = 128,
        expand_mode: str = 'double',
        metric: str = 'euclidean',
        use_residual: bool = False,
        *args,
        **kwargs,
    ):
        assert d_vector % n_subvectors == 0

        super(PQLite, self).__init__(
            code_size=n_subvectors,
            n_cells=n_cells,
            dtype='uint8',
            initial_size=initial_size,
            expand_step_size=expand_step_size,
            expand_mode=expand_mode,
        )

        self.d_vector = d_vector
        self.n_subvectors = n_subvectors
        self.d_subvector = d_vector // n_subvectors
        self.metric = metric
        self.use_residual = use_residual
        self.n_probe = 1

        if use_residual and (n_cells * 256 * n_subvectors * 4) <= 4 * 1024 ** 3:
            self._use_precomputed = True
        else:
            self._use_precomputed = False

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

        logger.info(f'=> start training VQ codec...')
        self.vq_codec.fit(x)

        logger.info(f'=> start training PQ codec...')
        self.pq_codec.fit(x)

        logger.info(f'=> index is trained successfully!')

    def add(
        self, x: 'np.ndarray', ids: Optional[List] = None, return_address: bool = False
    ):
        n_data, _ = self._sanity_check(x)

        assigned_cells = self.vq_codec.encode(x)
        quantized_x = self.encode(x)

        return super(PQLite, self).add(
            quantized_x,
            cells=assigned_cells,
            ids=ids,
        )

    def ivfpq_topk(self, precomputed, cells: List[int], k: int = 10):
        topk_sims = []
        topk_ids = []
        for cell_id in cells:
            is_empty = self._is_empties[cell_id]
            dists = precomputed.adist(self._storages[cell_id])  # (10000, )
            dists += is_empty * np.iinfo(np.int16).max
            dists = np.expand_dims(dists, axis=0)

            _topk_sims, indices = top_k(dists, k=k)
            _topk_ids = np.array(
                [idx for idx in self.get_id_by_address(cell_id, indices)],
                dtype=f'|S{self._key_length}',
            )
            topk_sims.append(_topk_sims)
            topk_ids.append(_topk_ids)
        topk_sims = np.concatenate(topk_sims, axis=1)
        topk_ids = np.concatenate(topk_ids, axis=1)
        idx = topk_sims.argsort(axis=1)[:, :k]
        topk_sims = np.take_along_axis(topk_sims, idx, axis=1)
        topk_ids = np.take_along_axis(topk_ids, idx, axis=1)
        return topk_sims, topk_ids

    def search_cells(
        self,
        query: 'np.ndarray',
        cells: 'np.ndarray',
        topk_dists: Optional['np.ndarray'] = None,
        n_probe_list=None,
        k: int = 10,
    ):

        topk_val, topk_ids = [], []
        for x, cell_idx in zip(query, cells):
            precomputed = self.pq_codec.precompute_adc(x)
            _topk_val, _topk_ids = self.ivfpq_topk(precomputed, cells=cell_idx, k=k)
            topk_val.append(_topk_val)
            topk_ids.append(_topk_ids)
        topk_val = np.concatenate(topk_val, axis=0)
        topk_ids = np.concatenate(topk_ids, axis=0)

        return topk_val, topk_ids

    def search(self, query: 'np.ndarray', k: int = 10):
        n_data, _ = self._sanity_check(query)
        assert 0 < k <= 1024

        vq_codebook = self.vq_codec.codebook

        # find n_probe closest cells
        dists = cdist(query, vq_codebook, metric='euclidean')
        topk_dists, cells = top_k(dists, k=self.n_probe)

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
            topk_dists=topk_dists,
            n_probe_list=None,
            k=k,
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
