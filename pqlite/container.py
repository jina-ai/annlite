from typing import Optional, List, Union

import numpy as np
from loguru import logger
from pathlib import Path

from jina import DocumentArray
from .storage.base import ExpandMode
from .storage.kv import DocStorage
from .storage.table import CellTable, MetaTable
from .core.index.pq_index import PQIndex
from .core.index.flat_index import FlatIndex
from .core.index.hnsw import HnswIndex
from .enums import Metric
from .core.codec.pq import PQCodec


class CellContainer:
    def __init__(
        self,
        dim: int,
        metric: Metric = Metric.EUCLIDEAN,
        pq_codec: Optional[PQCodec] = None,
        n_cells: int = 1,
        initial_size: Optional[int] = None,
        expand_step_size: Optional[int] = 1024,
        expand_mode: ExpandMode = ExpandMode.STEP,
        columns: Optional[List[tuple]] = None,
        data_path: Path = Path('./data'),
    ):
        self.dim = dim
        self.metric = metric
        self.n_cells = n_cells
        self.data_path = data_path

        if pq_codec is not None:
            self._vec_indexes = [
                PQIndex(
                    dim,
                    pq_codec,
                    metric=metric,
                    initial_size=initial_size,
                    expand_step_size=expand_step_size,
                    expand_mode=expand_mode,
                )
                for _ in range(n_cells)
            ]
        elif columns is None:
            self._vec_indexes = [
                HnswIndex(
                    dim,
                    metric=metric,
                    initial_size=initial_size,
                    expand_step_size=expand_step_size,
                    expand_mode=expand_mode,
                )
                for _ in range(n_cells)
            ]
        else:
            self._vec_indexes = [
                FlatIndex(
                    dim,
                    metric=metric,
                    initial_size=initial_size,
                    expand_step_size=expand_step_size,
                    expand_mode=expand_mode,
                )
                for _ in range(n_cells)
            ]
        self._doc_stores = [DocStorage(data_path / f'cell_{_}') for _ in range(n_cells)]

        self._cell_tables = [
            CellTable(f'table_{c}', columns=columns) for c in range(n_cells)
        ]

        self._meta_table = MetaTable('metas', data_path=data_path, in_memory=True)

    def ivf_search(
        self,
        x: np.ndarray,
        cells: np.ndarray,
        conditions: Optional[list] = None,
        limit: int = 10,
    ):
        dists = []

        doc_idx = []
        cell_ids = []
        count = 0
        for cell_id in cells:
            indices = None
            cell_table = self.cell_table(cell_id)
            if (conditions is not None) or (cell_table.deleted_count() > 0):
                indices = []
                for doc in cell_table.query(conditions=conditions):
                    indices.append(doc['_id'])

                if len(indices) == 0:
                    continue

                indices = np.array(indices, dtype=np.int64)

            if indices is not None:
                _dists, _doc_idx = self.vec_index(cell_id).search(
                    x, limit=limit, indices=indices
                )

                if count >= limit and _dists[0] > dists[-1][-1]:
                    continue

                dists.append(_dists)
                doc_idx.append(_doc_idx)
                cell_ids.extend([cell_id] * len(_dists))
                count += len(_dists)

        if len(dists) > 0:
            cell_ids = np.array(cell_ids, dtype=np.int64)
            dists = np.hstack(dists)
            doc_idx = np.hstack(doc_idx)
            indices = dists.argsort(axis=0)[:limit]
            dists = dists[indices]
            cell_ids = cell_ids[indices]
            doc_idx = doc_idx[indices]
        else:
            logger.debug(f'=> No matches retrieved in any cell')

        doc_ids = []
        for cell_id, offset in zip(cell_ids, doc_idx):
            doc_id = self.cell_table(cell_id).get_docid_by_offset(offset)
            doc_ids.append(doc_id)
        return dists, doc_ids, cell_ids

    def search_cells(
        self,
        query: np.ndarray,
        cells: np.ndarray,
        conditions: Optional[list] = None,
        limit: int = 10,
    ):
        topk_dists, topk_docs = [], []
        for x, cell_idx in zip(query, cells):
            # x.shape = (self.dim,)
            dists, doc_ids, cells = self.ivf_search(
                x, cells=cell_idx, conditions=conditions, limit=limit
            )

            topk_dists.append(dists)

            match_docs = DocumentArray()
            for dist, doc_id, cell_id in zip(dists, doc_ids, cells):
                doc = self.doc_store(cell_id).get([doc_id])[0]
                doc.scores[self.metric.name.lower()].value = dist
                match_docs.append(doc)
            topk_docs.append(match_docs)

        return topk_dists, topk_docs

    def insert(
        self,
        data: np.ndarray,
        cells: np.ndarray,
        docs: DocumentArray,
    ):
        assert len(docs) == len(data)

        offsets = []
        for doc, cell_id in zip(docs, cells):
            # Write-Ahead-Log (WAL)
            self.doc_store(cell_id).insert([doc])

            # update cell_table and meta_table
            offset = self.cell_table(cell_id).insert([doc])[0]
            self._meta_table.add_address(doc.id, cell_id, offset)
            offsets.append(offset)

        offsets = np.array(offsets, dtype=np.int64)
        self._add_vecs(data, cells, offsets)

        logger.debug(f'=> {len(docs)} new docs added')

    def _add_vecs(self, data: np.ndarray, cells: np.ndarray, offsets: np.ndarray):
        assert data.shape[0] == cells.shape[0]
        assert data.shape[1] == self.dim

        unique_cells, _ = np.unique(cells, return_counts=True)

        for cell_id in unique_cells:
            indices = cells == cell_id
            x = data[indices, :]
            ids = offsets[indices]

            self.vec_index(cell_id).add_with_ids(x, ids)

    def update(
        self,
        data: np.ndarray,
        cells: np.ndarray,
        docs: DocumentArray,
    ):
        new_data = []
        new_cells = []
        new_docs = []

        for (
            x,
            doc,
            cell_id,
        ) in zip(data, docs, cells):
            _cell_id, _offset = self._meta_table.get_address(doc.id)
            if cell_id == _cell_id:
                self.vec_index(cell_id).add_with_ids(x.reshape(1, -1), [_offset])
                self.cell_table(cell_id).undo_delete_by_offset(_offset)

            elif _cell_id is None:
                new_data.append(x)
                new_cells.append(cell_id)
                new_docs.append(doc)
            else:
                # relpace
                self.cell_table(_cell_id).delete_by_offset(_offset)

                new_data.append(x)
                new_cells.append(cell_id)
                new_docs.append(doc)

        if len(new_data) > 0:
            new_data = np.stack(new_data)
            new_cells = np.array(new_cells, dtype=np.int64)

            self.insert(new_data, new_cells, new_docs)

        logger.debug(f'=> {len(docs)} items updated')

    def delete(self, ids: List[str]):
        for doc_id in ids:
            cell_id, offset = self._meta_table.get_address(doc_id)
            if cell_id is not None:
                self.cell_table(cell_id).delete_by_offset(offset)

        logger.debug(f'=> {len(ids)} items deleted')

    def documents_generator(self, cell_id: int, batch_size: int = 256):
        for docs in self.doc_store(cell_id).batched_iterator(batch_size=batch_size):
            yield docs

    @property
    def cell_tables(self):
        return self._cell_tables

    def cell_table(self, cell_id: int):
        return self._cell_tables[cell_id]

    def doc_store(self, cell_id: int):
        return self._doc_stores[cell_id]

    def vec_index(self, cell_id: int):
        return self._vec_indexes[cell_id]

    @property
    def meta_table(self):
        return self._meta_table

    @property
    def total_docs(self):
        return sum([store.size for store in self._doc_stores])

    @property
    def index_size(self):
        return sum([table.size for table in self._cell_tables])
