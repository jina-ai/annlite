from typing import Optional, List, Union
import hashlib
from pathlib import Path
import numpy as np
from loguru import logger

from jina import DocumentArray
from jina.math.distance import cdist
from jina.math.helper import top_k
from .core import VQCodec, PQCodec
from .container import CellContainer
from .enums import Metric


class PQLite(CellContainer):
    """:class:`PQLite` is an approximate nearest neighbor search library.

    To create a :class:`PQLite` object, simply:

        .. highlight:: python
        .. code-block:: python
            pqlite = PQLite(dim=256, metric=pqlite.Metric.EUCLIDEAN)

    :param dim: dimensionality of input vectors. there are 2 constraints on dim:
            (1) it needs to be divisible by n_subvectors; (2) it needs to be a multiple of 4.*
    :param n_subvectors: number of sub-quantizers, essentially this is the byte size of
            each quantized vector, default is None.
    :param n_cells:  number of coarse quantizer clusters, default is 1.
    :param n_probe: number of cells to search for each query, default is 16.
    :param initial_size: initial capacity assigned to each voronoi cell of coarse quantizer.
            ``n_cells * initial_size`` is the number of vectors that can be stored initially.
            if any cell has reached its capacity, that cell will be automatically expanded.
            If you need to add vectors frequently, a larger value for init_size is recommended.
    :param data_path: location of directory to store the database.
    :param create: if False, do not create the directory path if it is missing.
    :param read_only: if True, the index is not writable.

    .. note::
        Remember that the shape of any tensor that contains data points has to be `[n_data, dim]`.
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
        create: bool = True,
        read_only: bool = False,
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

        self.vq_codec = VQCodec(n_cells, metric=metric) if n_cells > 1 else None
        self.pq_codec = (
            PQCodec(dim, n_subvectors=n_subvectors, n_clusters=256, metric=metric)
            if n_subvectors
            else None
        )

        if isinstance(data_path, str):
            data_path = Path(data_path)

        self.data_path = data_path
        if create:
            data_path.mkdir(exist_ok=True)

        self.read_only = read_only

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

    def _sanity_check(self, x: np.ndarray):
        assert len(x.shape) == 2
        assert x.shape[1] == self.dim

        return x.shape

    def fit(self, x: np.ndarray, auto_save: bool = True, force_retrain: bool = False):
        """Train pqlite with training data.

        :param x: the ndarray data for training.
        :param auto_save: if False, will not save the trained model.
        :param force_retrain: if True, enforce to retrain the model, and overwrite the model if ``auto_save=True``.

        """
        n_data, _ = self._sanity_check(x)

        if self.is_trained and not force_retrain:
            logger.warning('The pqlite has been trained. Please use ``force_retrain=True`` to retrain.')
            return

        if self.vq_codec:
            logger.info(
                f'=> start training VQ codec (K={self.n_cells}) with {n_data} data...'
            )
            self.vq_codec.fit(x)

        if self.pq_codec:
            logger.info(
                f'=> start training PQ codec (n_subvectors={self.n_subvectors}) with {n_data} data...'
            )
            self.pq_codec.fit(x)

        logger.info(f'=> pqlite is successfully trained!')

        if auto_save:
            logger.info(f'==> save the trained parameters to {self.data_path / self._model_hash}')
            self.create_model_dir()
            self.vq_codec.dump(self._vq_codec_path)
            self.pq_codec.dump(self._pq_codec_path)

    def index(self, docs: DocumentArray, **kwargs):
        """

        :param docs: The documents to index
        :return:
        """

        if self.read_only:
            logger.warning('The pqlite is readonly, cannot add documents')
            return

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
        if self.read_only:
            logger.warning('The pqlite is readonly, cannot update documents')
            return

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

    def model_dir_exists(self):
        """Check whether the model directory exists at the data path"""
        model_path = self.data_path / self._model_hash
        return model_path.exists()

    def create_model_dir(self):
        """Create a new directory at the data path to save model."""
        model_path = self.data_path / self._model_hash
        model_path.mkdir(exist_ok=True)

    @property
    def is_trained(self):
        if self.vq_codec and (not self.vq_codec.is_trained):
            return False
        if self.pq_codec and (not self.pq_codec.is_trained):
            return False
        return True

    @property
    def _model_hash(self):
        key = f'{self.n_cells} x {self.n_subvectors}'
        return hashlib.md5(key.encode()).hexdigest()

    @property
    def _vq_codec_path(self):
        return self.data_path / self._model_hash / 'vq_codec.bin'

    @property
    def _pq_codec_path(self):
        return self.data_path / self._model_hash / 'pq_codec.bin'

    @property
    def use_smart_probing(self):
        return self._use_smart_probing

    @use_smart_probing.setter
    def use_smart_probing(self, value):
        assert type(value) is bool
        self._use_smart_probing = value

    # @property
    # def smart_probing_temperature(self):
    #     return self._smart_probing_temperature
    #
    # @smart_probing_temperature.setter
    # def smart_probing_temperature(self, value):
    #     assert value > 0
    #     assert self.use_smart_probing, 'set use_smart_probing to True first'
    #     self._smart_probing_temperature = value
