import hashlib
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Union

import numpy as np
from docarray.math.ndarray import to_numpy_array
from loguru import logger

if TYPE_CHECKING:
    from docarray import DocumentArray

from .container import CellContainer
from .core import PQCodec, ProjectorCodec, VQCodec
from .enums import Metric
from .filter import Filter
from .helper import setup_logging
from .math import cdist, top_k

MAX_TRAINING_DATA_SIZE = 10240


class AnnLite(CellContainer):
    """:class:`AnnLite` is an approximate nearest neighbor search library.

    To create a :class:`AnnLite` object, simply:

        .. highlight:: python
        .. code-block:: python
            pqlite = AnnLite(256, metric='cosine')

    :param dim: dimensionality of input vectors. there are 2 constraints on dim:
            (1) it needs to be divisible by n_subvectors; (2) it needs to be a multiple of 4.*
    :param metric: distance metric type, can be 'euclidean', 'inner_product', or 'cosine'.
    :param n_subvectors: number of sub-quantizers, essentially this is the byte size of
            each quantized vector, default is None.
    :param n_cells:  number of coarse quantizer clusters, default is 1.
    :param n_probe: number of cells to search for each query, default is 16.
    :param n_components: number of components to keep.
    :param initial_size: initial capacity assigned to each voronoi cell of coarse quantizer.
            ``n_cells * initial_size`` is the number of vectors that can be stored initially.
            if any cell has reached its capacity, that cell will be automatically expanded.
            If you need to add vectors frequently, a larger value for init_size is recommended.
    :param data_path: path to the directory where the data is stored.
    :param create_if_missing: if False, do not create the directory path if it is missing.
    :param read_only: if True, the index is not writable.
    :param verbose: if True, will print the debug logging info.

    .. note::
        Remember that the shape of any tensor that contains data points has to be `[n_data, dim]`.
    """

    def __init__(
        self,
        dim: int,
        metric: Union[str, Metric] = 'cosine',
        n_cells: int = 1,
        n_subvectors: Optional[int] = None,
        n_clusters: Optional[int] = 256,
        n_probe: int = 16,
        n_components: Optional[int] = None,
        initial_size: Optional[int] = None,
        expand_step_size: int = 10240,
        columns: Optional[List[tuple]] = None,
        data_path: Union[Path, str] = Path('./data'),
        create_if_missing: bool = True,
        read_only: bool = False,
        verbose: bool = False,
        *args,
        **kwargs,
    ):
        setup_logging(verbose)

        if n_subvectors:
            assert (
                dim % n_subvectors == 0
            ), '"dim" needs to be divisible by "n_subvectors"'
        self.dim = dim
        self.n_components = n_components
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters
        self.n_probe = max(n_probe, n_cells)
        self.n_cells = n_cells

        if isinstance(metric, str):
            metric = Metric.from_string(metric)
        self.metric = metric

        self._use_smart_probing = True

        self.read_only = read_only

        data_path = Path(data_path)
        if create_if_missing:
            data_path.mkdir(parents=True, exist_ok=True)
        self.data_path = data_path

        self.projector_codec = None
        if self._projector_codec_path.exists():
            logger.info(
                f'Load pre-trained projector codec (n_components={self.n_components}) from {self.model_path}'
            )
            self.projector_codec = ProjectorCodec.load(self._projector_codec_path)
        elif n_components:
            logger.info(
                f'Initialize Projector codec (n_components={self.n_components})'
            )
            self.projector_codec = ProjectorCodec(dim, n_components=self.n_components)

        self.vq_codec = None
        if self._vq_codec_path.exists():
            logger.info(
                f'Load trained VQ codec (K={self.n_cells}) from {self.model_path}'
            )
            self.vq_codec = VQCodec.load(self._vq_codec_path)
        elif n_cells > 1:
            logger.info(f'Initialize VQ codec (K={self.n_cells})')
            self.vq_codec = VQCodec(self.n_cells, metric=self.metric)

        self.pq_codec = None
        if self._pq_codec_path.exists():
            logger.info(
                f'Load trained PQ codec (n_subvectors={self.n_subvectors}) from {self.model_path}'
            )
            self.pq_codec = PQCodec.load(self._pq_codec_path)
        elif n_subvectors:
            logger.info(f'Initialize PQ codec (n_subvectors={self.n_subvectors})')
            self.pq_codec = PQCodec(
                dim=dim
                if not self.projector_codec
                else self.projector_codec.n_components,
                n_subvectors=self.n_subvectors,
                n_clusters=self.n_clusters,
                metric=self.metric,
            )

        super(AnnLite, self).__init__(
            dim=dim,
            metric=metric,
            projector_codec=self.projector_codec,
            pq_codec=self.pq_codec,
            n_cells=n_cells,
            initial_size=initial_size,
            expand_step_size=expand_step_size,
            columns=columns,
            data_path=data_path,
            **kwargs,
        )

        if not self.is_trained and self.total_docs > 0:
            # train the index from scratch based on the data in the data_path
            logger.info(f'Train the index by reading data from {self.data_path}')
            total_size = 0
            # TODO: add a progress bar
            for docs in self.documents_generator(0, batch_size=1024):
                x = to_numpy_array(docs.embeddings)
                total_size += x.shape[0]
                self.partial_train(x, auto_save=True, force_train=True)
                if total_size >= MAX_TRAINING_DATA_SIZE:
                    break
            logger.info(f'Total training data size: {total_size}')

        if self.total_docs > 0:
            self._rebuild_index()

    def _sanity_check(self, x: 'np.ndarray'):
        assert x.ndim == 2, 'inputs must be a 2D array'
        assert (
            x.shape[1] == self.dim
        ), f'inputs must have the same dimension as the index , got {x.shape[1]}, expected {self.dim}'

        return x.shape

    def train(self, x: 'np.ndarray', auto_save: bool = True, force_train: bool = False):
        """Train pqlite with training data.

        :param x: the ndarray data for training.
        :param auto_save: if False, will not dump the trained model to ``model_path``.
        :param force_train: if True, enforce to retrain the model, and overwrite the model if ``auto_save=True``.

        """
        n_data, _ = self._sanity_check(x)

        if self.is_trained and not force_train:
            logger.warning(
                'The indexer has been trained or is not trainable. Please use ``force_train=True`` to retrain.'
            )
            return

        if self.projector_codec:
            logger.info(
                f'Start training Projector codec (n_components={self.n_components}) with {n_data} data...'
            )
            self.projector_codec.fit(x)

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

        logger.info(f'The annlite is successfully trained!')

        if auto_save:
            self.dump_model()

    def partial_train(
        self, x: np.ndarray, auto_save: bool = True, force_train: bool = False
    ):
        """Train indexer parameters with a minibatch of  data.

        :param x: the ndarray data for training.
        :param auto_save: if False, will not dump the trained model to ``model_path``.
        :param force_train: if True, enforce to retrain the model, and overwrite the model if ``auto_save=True``.

        """
        n_data, _ = self._sanity_check(x)

        if self.is_trained and not force_train:
            logger.warning(
                'The annlite has been trained or is not trainable. Please use ``force_train=True`` to retrain.'
            )
            return

        if self.projector_codec:
            logging.info(
                f'Partial training Projector codec (n_components={self.n_components}) with {n_data} data...'
            )
            self.projector_codec.partial_fit(x)

        if self.vq_codec:
            logger.info(
                f'Partial training VQ codec (K={self.n_cells}) with {n_data} data...'
            )
            self.vq_codec.partial_fit(x)

        if self.pq_codec:
            logger.info(
                f'Partial training PQ codec (n_subvectors={self.n_subvectors}) with {n_data} data...'
            )
            self.pq_codec.partial_fit(x)

        if auto_save:
            self.dump_model()

    def index(self, docs: 'DocumentArray', **kwargs):
        """Index new documents.

        :param docs: the document array to be indexed.
        """

        if self.read_only:
            logger.error('The indexer is readonly, cannot add new documents')
            return

        if not self.is_trained:
            raise RuntimeError(f'The indexer is not trained, cannot add new documents')

        x = to_numpy_array(docs.embeddings)
        n_data, _ = self._sanity_check(x)

        assigned_cells = (
            self.vq_codec.encode(x)
            if self.vq_codec
            else np.zeros(n_data, dtype=np.int64)
        )
        return super(AnnLite, self).insert(x, assigned_cells, docs)

    def update(
        self, docs: 'DocumentArray', raise_errors_on_not_found: bool = False, **kwargs
    ):
        """Update existing documents.

        :param raise_errors_on_not_found: whether to raise exception when id not found.
        :param docs: the document array to be updated.
        """
        if self.read_only:
            logger.error('The indexer is readonly, cannot update documents')
            return

        if not self.is_trained:
            raise RuntimeError(f'The indexer is not trained, cannot add new documents')

        x = to_numpy_array(docs.embeddings)
        n_data, _ = self._sanity_check(x)

        assigned_cells = (
            self.vq_codec.encode(x)
            if self.vq_codec
            else np.zeros(n_data, dtype=np.int64)
        )

        return super(AnnLite, self).update(
            x, assigned_cells, docs, raise_errors_on_not_found
        )

    def search(
        self,
        docs: 'DocumentArray',
        filter: Dict = {},
        limit: int = 10,
        include_metadata: bool = True,
        **kwargs,
    ):
        """Search the index, and attach matches to the query Documents in `docs`

        :param docs: the document array to be searched.
        :param filter: the filter to be applied to the search.
        :param limit: the number of results to get for each query document in search
        :param include_metadata: whether to return document metadata in response.
        """
        if not self.is_trained:
            raise RuntimeError(f'The indexer is not trained, cannot add new documents')

        query_np = to_numpy_array(docs.embeddings)

        match_dists, match_docs = self._search_documents(
            query_np, filter, limit, include_metadata
        )

        for doc, matches in zip(docs, match_docs):
            doc.matches = matches

    def _search_documents(
        self,
        query_np: 'np.ndarray',
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

    def filter(
        self,
        filter: Dict = {},
        limit: int = 10,
        include_metadata: bool = True,
    ):

        cells = [x for x in range(self.n_cells)]
        where_clause, where_params = Filter(filter).parse_where_clause()

        match_docs = self.filter_cells(
            cells=cells,
            where_clause=where_clause,
            where_params=where_params,
            include_metadata=include_metadata,
        )
        return match_docs[0:limit]

    def _cell_selection(self, query_np, limit):

        n_data, _ = self._sanity_check(query_np)

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
        query_np: 'np.ndarray',
        filter: Dict = {},
        limit: int = 10,
        **kwargs,
    ):
        """Search the index and return distances to the query and ids of the closest documents.

        :param query_np: matrix containing query vectors as rows
        :param filter: the filtering conditions
        :param limit: the number of results to get for each query document in search
        """

        if not self.is_trained:
            raise RuntimeError(f'The indexer is not trained, cannot add new documents')

        dists, doc_ids = self._search_numpy(query_np, filter, limit)
        return dists, doc_ids

    def _search_numpy(self, query_np: 'np.ndarray', filter: Dict = {}, limit: int = 10):
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

    def delete(
        self,
        docs: Union['DocumentArray', List[str]],
        raise_errors_on_not_found: bool = False,
    ):
        """Delete entries from the index by id

        :param raise_errors_on_not_found: whether to raise exception when id not found.
        :param docs: the documents to delete
        """

        super().delete(
            docs if isinstance(docs, list) else docs[:, 'id'], raise_errors_on_not_found
        )

    def clear(self):
        """Clear the whole database"""
        for cell_id in range(self.n_cells):
            logger.debug(f'Clearing cell {cell_id}')
            self.vec_index(cell_id).reset()
            self.cell_table(cell_id).clear()
            self.doc_store(cell_id).clear()
        self.meta_table.clear()

    def close(self):
        for cell_id in range(self.n_cells):
            self.doc_store(cell_id).close()

    def encode(self, x: 'np.ndarray'):
        n_data, _ = self._sanity_check(x)

        if self.projector_codec:
            x = self.projector_codec.encode(x)

        if self.vq_codec:
            x = self.pq_codec.encode(x)

        return x

    def decode(self, x: 'np.ndarray'):
        assert len(x.shape) == 2
        assert x.shape[1] == self.n_subvectors

        if self.pq_codec:
            x = self.pq_codec.decode(x)

        if self.projector_codec:
            x = self.projector_codec.decode(x)

        return x

    @property
    def params_hash(self):
        model_metas = (
            f'dim: {self.dim} '
            f'metric: {self.metric} '
            f'n_cells: {self.n_cells} '
            f'n_components: {self.n_components} '
            f'n_subvectors: {self.n_subvectors}'
        )
        return hashlib.md5(f'{model_metas}'.encode()).hexdigest()

    @property
    def model_path(self):
        return self.data_path / f'parameters-{self.params_hash}'

    @property
    def _vq_codec_path(self):
        return self.model_path / f'vq_codec.params'

    @property
    def _pq_codec_path(self):
        return self.model_path / f'pq_codec.params'

    @property
    def _projector_codec_path(self):
        return self.model_path / f'projector_codec.params'

    @property
    def index_hash(self):
        latest_commit = self.meta_table.get_latest_commit()
        date_time = latest_commit[-1] if latest_commit else None
        if date_time:
            return date_time.isoformat('#', 'seconds')

        return None

    @property
    def index_path(self):
        if self.index_hash:
            return (
                self.data_path
                / f'snapshot-{self.params_hash}'
                / f'{self.index_hash}-SNAPSHOT'
            )
        return None

    @property
    def snapshot_path(self):
        paths = list(
            (self.data_path / f'snapshot-{self.params_hash}').glob(f'*-SNAPSHOT')
        )
        if paths:
            return paths[0]
        return None

    def dump_model(self):
        logger.info(f'Save the parameters to {self.model_path}')
        self.model_path.mkdir(parents=True, exist_ok=True)
        if self.projector_codec:
            self.projector_codec.dump(self._projector_codec_path)
        if self.vq_codec:
            self.vq_codec.dump(self._vq_codec_path)
        if self.pq_codec:
            self.pq_codec.dump(self._pq_codec_path)

    def dump_index(self):
        logger.info(f'Save the indexer to {self.index_path}')

        try:
            self.index_path.mkdir(parents=True)

            self.meta_table.dump(self.index_path / 'meta_table.db')

            for cell_id in range(self.n_cells):
                self.vec_index(cell_id).dump(self.index_path / f'cell_{cell_id}.hnsw')
                self.cell_table(cell_id).dump(self.index_path / f'cell_{cell_id}.db')
        except Exception as ex:
            logger.error(f'Failed to dump the indexer, {ex!r}')
            import shutil

            shutil.rmtree(self.index_path)

    def dump(self):
        self.dump_model()
        self.dump_index()

    def _rebuild_index(self):
        if self.snapshot_path:
            logger.info(f'Load the indexer from snapshot {self.snapshot_path}')
            self.meta_table.load(self.snapshot_path / 'meta_table.db')
            for cell_id in range(self.n_cells):
                self.vec_index(cell_id).load(
                    self.snapshot_path / f'cell_{cell_id}.hnsw'
                )
                self.cell_table(cell_id).load(self.snapshot_path / f'cell_{cell_id}.db')
        else:
            logger.info(f'Rebuild the indexer from scratch')
            dump_after_building = False
            for cell_id in range(self.n_cells):
                cell_size = self.doc_store(cell_id).size

                if cell_size == 0:
                    continue  # skip empty cell

                dump_after_building = True

                logger.debug(
                    f'Rebuild the index of cell-{cell_id} ({cell_size} docs)...'
                )
                for docs in self.documents_generator(cell_id, batch_size=10240):
                    x = to_numpy_array(docs.embeddings)

                    assigned_cells = np.ones(len(docs), dtype=np.int64) * cell_id
                    super().insert(x, assigned_cells, docs, only_index=True)
                logger.debug(f'Rebuild the index of cell-{cell_id} done')

            if dump_after_building:
                self.dump()

    @property
    def is_trained(self):
        if self.projector_codec and (not self.projector_codec.is_trained):
            return False
        if self.vq_codec and (not self.vq_codec.is_trained):
            return False
        if self.pq_codec and (not self.pq_codec.is_trained):
            return False
        return True

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
            'n_components': self.n_components,
            'metric': self.metric.name,
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
