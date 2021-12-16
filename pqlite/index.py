import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
from docarray import DocumentArray
from docarray.math.distance import cdist
from docarray.math.helper import top_k
from loguru import logger

from .container import CellContainer
from .core import PQCodec, VQCodec
from .enums import Metric
from .filter import Filter


class PQLite(CellContainer):
    """:class:`PQLite` is an approximate nearest neighbor search library.

    To create a :class:`PQLite` object, simply:

        .. highlight:: python
        .. code-block:: python
            pqlite = PQLite(dim=256, metric=pqlite.Metric.EUCLIDEAN)

    :param dim: dimensionality of input vectors. there are 2 constraints on dim:
            (1) it needs to be divisible by n_subvectors; (2) it needs to be a multiple of 4.*
    :param metric: distance metric type, can be 'euclidean', 'inner_product', or 'cosine'.
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
        metric: Union[str, Metric] = Metric.COSINE,
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
        self.n_cells = n_cells

        if isinstance(metric, str):
            metric = Metric.from_string(metric)
        self.metric = metric

        self._use_smart_probing = True

        self.read_only = read_only

        data_path = Path(data_path)
        if create:
            data_path.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path

        self.vq_codec = None
        if self._vq_codec_path.exists() and n_cells > 1:
            logger.info(
                f'Load trained VQ codec (K={self.n_cells}) from {self.model_path}'
            )
            self.vq_codec = VQCodec.load(self._vq_codec_path)
        elif n_cells > 1:
            logger.info(f'Initialize VQ codec (K={self.n_cells})')
            self.vq_codec = VQCodec(self.n_cells, metric=self.metric)

        self.pq_codec = None
        if self._pq_codec_path.exists() and n_subvectors:
            logger.info(
                f'Load trained PQ codec (n_subvectors={self.n_subvectors}) from {self.model_path}'
            )
            self.pq_codec = PQCodec.load(self._pq_codec_path)
        elif n_subvectors:
            logger.info(f'Initialize PQ codec (n_subvectors={self.n_subvectors})')
            self.pq_codec = PQCodec(
                dim, n_subvectors=n_subvectors, n_clusters=256, metric=self.metric
            )

        super(PQLite, self).__init__(
            dim=dim,
            metric=metric,
            pq_codec=self.pq_codec,
            n_cells=n_cells,
            initial_size=initial_size,
            expand_step_size=expand_step_size,
            columns=columns,
            data_path=data_path,
            **kwargs,
        )

        if self.total_docs > 0:
            self._rebuild_index()

    def _sanity_check(self, x: np.ndarray):
        assert len(x.shape) == 2
        assert x.shape[1] == self.dim

        return x.shape

    def train(self, x: np.ndarray, auto_save: bool = True, force_retrain: bool = False):
        """Train pqlite with training data.

        :param x: the ndarray data for training.
        :param auto_save: if False, will not dump the trained model to ``model_path``.
        :param force_retrain: if True, enforce to retrain the model, and overwrite the model if ``auto_save=True``.

        """
        n_data, _ = self._sanity_check(x)

        if self.is_trained and not force_retrain:
            logger.warning(
                'The pqlite has been trained or is not trainable. Please use ``force_retrain=True`` to retrain.'
            )
            return

        if self.vq_codec:
            logger.info(
                f'Start training VQ codec (K={self.n_cells}) with {n_data} data...'
            )
            self.vq_codec.fit(x)

        if self.pq_codec:
            logger.info(
                f'Start training PQ codec (n_subvectors={self.n_subvectors}) with {n_data} data...'
            )
            self.pq_codec.fit(x)

        logger.info(f'The pqlite is successfully trained!')

        if auto_save:
            self.dump_model()

    def index(self, docs: DocumentArray, **kwargs):
        """Index new documents

        :param docs: the documents to index
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
        """Update existing documents

        :param docs: the documents to update
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
        filter: Dict = {},
        limit: int = 10,
        include_metadata: bool = True,
        **kwargs,
    ):
        """Search the index, and attach matches to the query Documents in `docs`

        :param docs: the query documents to search
        :param filter: the filtering conditions
        :param limit: the number of results to get for each query document in search
        :param include_metadata: whether to return document metadata in response.
        """
        query_np = docs.embeddings

        match_dists, match_docs = self._search_documents(
            query_np, filter, limit, include_metadata
        )

        for doc, matches in zip(docs, match_docs):
            doc.matches = matches

    def _search_documents(
        self,
        query_np,
        filter: Dict = {},
        limit: int = 10,
        include_metadata: bool = True,
    ):

        cells = self._cell_selection(query_np, limit)
        where_clause, where_params = Filter(filter).parse_where_clause()

        match_dists, match_docs = self.search_cells(
            query=query_np,
            cells=cells,
            where_clause=where_clause,
            where_params=where_params,
            limit=limit,
            include_metadata=include_metadata,
        )
        return match_dists, match_docs

    def _cell_selection(self, query_np, limit):

        n_data, _ = self._sanity_check(query_np)
        assert 0 < limit <= 1024

        if self.vq_codec:
            dists = cdist(
                query_np, self.vq_codec.codebook, metric=self.metric.name.lower()
            )
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

        return cells

    def search_numpy(
        self,
        query_np: np.ndarray,
        filter: Dict = {},
        limit: int = 10,
        include_metadata: bool = True,
        **kwargs,
    ):
        """Search the index and return distances to the query and ids of the closest documents.

        :param query_np: matrix containing query vectors as rows
        :param filter: the filtering conditions
        :param limit: the number of results to get for each query document in search
        :param include_metadata: whether to return document metadata in response.
        """

        dists, doc_ids = self._search_numpy(query_np, filter, limit)
        return dists, doc_ids

    def _search_numpy(self, query_np, filter: Dict = {}, limit: int = 10):
        """Search approximate nearest vectors in different cells, returns distances and ids

        :param query_np: matrix containing query vectors as rows
        :param filter: the filtering conditions
        :param limit: the number of results to get for each query document in search
        """
        cells = self._cell_selection(query_np, limit)
        where_clause, where_params = Filter(filter).parse_where_clause()

        dists, ids = self._search_cells(
            query=query_np,
            cells=cells,
            where_clause=where_clause,
            where_params=where_params,
            limit=limit,
        )
        return dists, ids

    def delete(self, docs: Union[DocumentArray, List[str]]):
        """Delete entries from the index by id

        :param docs: the documents to delete
        """
        doc_ids = docs.get_attributes('id') if isinstance(docs, DocumentArray) else docs
        super().delete(doc_ids)

    def clear(self):
        """Clear the whole database"""
        for cell_id in range(self.n_cells):
            logger.info(f'Clear the index of cell-{cell_id}')
            self.vec_index(cell_id).reset()
            self.cell_table(cell_id).clear()
            self.doc_store(cell_id).clear()
        self.meta_table.clear()

    def close(self):
        for cell_id in range(self.n_cells):
            self.doc_store(cell_id).close()

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
        return self.model_path.exists()

    def create_model_dir(self):
        """Create a new directory at the data path to save model."""
        self.model_path.mkdir(exist_ok=True)

    def dump_model(self):
        logger.info(f'Save the trained parameters to {self.model_path}')
        self.create_model_dir()
        if self.vq_codec:
            self.vq_codec.dump(self._vq_codec_path)
        if self.pq_codec:
            self.pq_codec.dump(self._pq_codec_path)

    def _rebuild_index(self):
        for cell_id in range(self.n_cells):
            cell_size = self.doc_store(cell_id).size
            logger.info(f'Rebuild the index of cell-{cell_id} ({cell_size} docs)...')
            self.vec_index(cell_id).reset(capacity=cell_size)
            for docs in self.documents_generator(cell_id, batch_size=10240):
                x = docs.embeddings
                assigned_cells = np.ones(len(docs), dtype=np.int64) * cell_id
                super().insert(x, assigned_cells, docs)

    @property
    def is_trained(self):
        if self.vq_codec and (not self.vq_codec.is_trained):
            return False
        if self.pq_codec and (not self.pq_codec.is_trained):
            return False
        return True

    @property
    def _model_hash(self):
        key = f'{self.n_cells} x {self.n_subvectors} x {self.metric.name}'
        return hashlib.md5(key.encode()).hexdigest()

    @property
    def model_path(self):
        return self.data_path / self._model_hash

    @property
    def _vq_codec_path(self):
        return self.model_path / 'vq_codec.bin'

    @property
    def _pq_codec_path(self):
        return self.model_path / 'pq_codec.bin'

    @property
    def use_smart_probing(self):
        return self._use_smart_probing

    @use_smart_probing.setter
    def use_smart_probing(self, value):
        assert type(value) is bool
        self._use_smart_probing = value

    @property
    def stat(self):
        """Get information on status of the indexer."""
        return {
            'total_docs': self.total_docs,
            'index_size': self.index_size,
            'n_cells': self.n_cells,
            'dim': self.dim,
            'metric': self.metric,
            'is_trained': self.is_trained,
        }

    # @property
    # def smart_probing_temperature(self):
    #     return self._smart_probing_temperature
    #
    # @smart_probing_temperature.setter
    # def smart_probing_temperature(self, value):
    #     assert value > 0
    #     assert self.use_smart_probing, 'set use_smart_probing to True first'
    #     self._smart_probing_temperature = value
