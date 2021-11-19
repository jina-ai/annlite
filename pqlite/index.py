from typing import Optional, List, Union
from pathlib import Path
import numpy as np
from loguru import logger

from jina import DocumentArray
from jina.math.distance import cdist
from jina.math.helper import top_k
from .core import VQCodec, PQCodec
from .storage.cell import CellContainer
from .enums import Metric

from pqlite.utils.asymmetric_distance import dist_pqcodes_to_codebooks

class PQLite(CellContainer):
    """:class:`PQLite` is an implementation of IVF-PQ being with equipped with SQLite.

    To create a :class:`PQLite` object, simply:

        .. highlight:: python
        .. code-block:: python
            pqlite = PQLite(d_vector=256, metric='euclidean')

    :param dim: the dimensionality of input vectors. there are 2 constraints on d_vector:
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
        dim: int,
        metric: Metric = Metric.EUCLIDEAN,
        n_cells: int = 1,
        n_subvectors: Optional[int] = None,
        n_probe: int = 16,
        initial_size: Optional[int] = None,
        expand_step_size: int = 10240,
        columns: Optional[List[tuple]] = None,
        data_path: Union[Path, str] = Path('./data'),
        *args,
        **kwargs,
    ):
        if n_subvectors:
            assert (
                dim % n_subvectors == 0
            ), '"dim" needs to be divisible by "n_subvectors"'

        self.n_subvectors = n_subvectors
        self.n_probe = max(n_probe, n_cells)

        self._use_smart_probing = True
        self._smart_probing_temperature = 30.0

        self.vq_codec = VQCodec(n_cells, metric=metric) if n_cells > 1 else None
        self.pq_codec = (
            PQCodec(dim, n_subvectors=n_subvectors, n_clusters=256, metric=metric)
            if n_subvectors
            else None
        )
        if isinstance(data_path, str):
            data_path = Path(data_path)
        data_path.mkdir(exist_ok=True)

        super(PQLite, self).__init__(
            dim=dim,
            metric=metric,
            pq_codec=self.pq_codec,
            n_cells=n_cells,
            initial_size=initial_size,
            expand_step_size=expand_step_size,
            columns=columns,
            data_path=data_path,
        )

    def _sanity_check(self, x: 'np.ndarray'):
        assert len(x.shape) == 2
        assert x.shape[1] == self.dim

        return x.shape

    def fit(self, x: 'np.ndarray', force_retrain: bool = False):
        n_data, d_vector = self._sanity_check(x)

        logger.info(
            f'=> start training VQ codec (K={self.n_cells}) with {n_data} data...'
        )
        self.vq_codec.fit(x)

        logger.info(
            f'=> start training PQ codec (n_subvectors={self.n_subvectors}) with {n_data} data...'
        )
        self.pq_codec.fit(x)

        logger.info(f'=> pqlite is successfully trained!')

    def index(self, docs: DocumentArray, **kwargs):
        """

        :param docs: The documents to index
        :return:
        """

        x = docs.embeddings

        n_data, _ = self._sanity_check(x)

        assigned_cells = (
            self.vq_codec.encode(x)
            if self.vq_codec
            else np.zeros(n_data, dtype=np.int64)
        )

        return super(PQLite, self).insert(x, assigned_cells, docs)

    def update(self, docs: DocumentArray, **kwargs):
        """

        :param docs: the documents to update
        :return:
        """
        x = docs.embeddings
        n_data, _ = self._sanity_check(x)

        assigned_cells = (
            self.vq_codec.encode(x)
            if self.vq_codec
            else np.zeros(n_data, dtype=np.int64)
        )

        return super(PQLite, self).update(x, assigned_cells, docs)

    def search(
        self,
        docs: DocumentArray,
        conditions: Optional[list] = None,
        limit: int = 10,
        **kwargs,
    ):
        query = docs.embeddings
        n_data, _ = self._sanity_check(query)

        assert 0 < limit <= 1024

        if self.vq_codec:
            vq_codebook = self.vq_codec.codebook
            # find n_probe closest cells
            dists = cdist(query, vq_codebook, metric=self.metric.name.lower())
            dists, cells = top_k(dists, k=self.n_probe)
        else:
            cells = np.zeros((n_data, 1), dtype=np.int64)

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

        match_dists, match_docs = self.search_cells(
            query=query,
            cells=cells,
            conditions=conditions,
            limit=limit,
        )

        for doc, matches in zip(docs, match_docs):
            doc.matches = matches

    def encode(self, x: np.ndarray):
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
