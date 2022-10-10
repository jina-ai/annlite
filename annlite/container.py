import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
from docarray import Document, DocumentArray
from loguru import logger

if TYPE_CHECKING:
    from .core.codec.pq import PQCodec
    from .core.codec.projector import ProjectorCodec

from .core.index.hnsw import HnswIndex
from .enums import Metric
from .storage.base import ExpandMode
from .storage.kv import DocStorage
from .storage.table import CellTable, MetaTable

VALID_FILTERABLE_DATA_TYPES = [int, str, float]


class CellContainer:
    def __init__(
        self,
        n_dim: int,
        metric: Metric = Metric.COSINE,
        n_cells: int = 1,
        projector_codec: Optional['ProjectorCodec'] = None,
        pq_codec: Optional['PQCodec'] = None,
        initial_size: Optional[int] = None,
        expand_step_size: int = 50000,
        expand_mode: 'ExpandMode' = ExpandMode.STEP,
        filterable_attrs: Optional[Dict] = None,
        serialize_config: Optional[Dict] = None,
        data_path: 'Path' = Path('./data'),
        **kwargs,
    ):
        self.n_dim = n_dim
        self.metric = metric
        self.n_cells = n_cells
        self.n_components = projector_codec.n_components if projector_codec else None
        self.data_path = data_path
        self.serialize_config = serialize_config

        self._pq_codec = pq_codec
        self._projector_codec = projector_codec

        self._vec_indexes = [
            HnswIndex(
                dim=self.n_components or n_dim,
                metric=metric,
                initial_size=initial_size,
                expand_step_size=expand_step_size,
                expand_mode=expand_mode,
                pq_codec=pq_codec,
                **kwargs,
            )
            for _ in range(n_cells)
        ]

        self._doc_stores = [
            DocStorage(
                data_path / f'cell_{_}',
                serialize_config=serialize_config or {},
                lock=True,
            )
            for _ in range(n_cells)
        ]

        columns = []
        if filterable_attrs:
            for attr_name, attr_type in filterable_attrs.items():
                if isinstance(attr_type, str):
                    attr_type = eval(attr_type)

                if attr_type not in VALID_FILTERABLE_DATA_TYPES:
                    raise ValueError(
                        f'Invalid filterable attribute type `{attr_type}` for attribute `{attr_name}`. '
                    )
                columns.append((attr_name, attr_type))

        self._cell_tables = [
            CellTable(f'table_{c}', columns=columns) for c in range(n_cells)
        ]

        self._meta_table = MetaTable('metas', data_path=data_path, in_memory=False)

    def ivf_search(
        self,
        x: 'np.ndarray',
        cells: 'np.ndarray',
        where_clause: str = '',
        where_params: Tuple = (),
        limit: int = 10,
    ):
        dists = []

        doc_idx = []
        cell_ids = []
        count = 0
        for cell_id in cells:
            cell_table = self.cell_table(cell_id)
            cell_size = cell_table.count()
            if cell_size == 0:
                continue

            indices = None
            if where_clause or (cell_table.deleted_count() > 0):
                indices = cell_table.query(
                    where_clause=where_clause, where_params=where_params
                )

                if len(indices) == 0:
                    continue

                indices = np.array(indices, dtype=np.int64)

            _dists, _doc_idx = self.vec_index(cell_id).search(
                x, limit=min(limit, cell_size), indices=indices
            )

            if count >= limit and _dists[0] > dists[-1][-1]:
                continue

            dists.append(_dists)
            doc_idx.append(_doc_idx)
            cell_ids.extend([cell_id] * len(_dists))
            count += len(_dists)

        cell_ids = np.array(cell_ids, dtype=np.int64)
        if len(dists) != 0:
            dists = np.hstack(dists)
            doc_idx = np.hstack(doc_idx)

            indices = dists.argsort(axis=0)[:limit]
            dists = dists[indices]
            cell_ids = cell_ids[indices]
            doc_idx = doc_idx[indices]

        doc_ids = []
        for cell_id, offset in zip(cell_ids, doc_idx):
            doc_id = self.cell_table(cell_id).get_docid_by_offset(offset)
            doc_ids.append(doc_id)
        return dists, doc_ids, cell_ids

    def filter_cells(
        self,
        cells: 'np.ndarray',
        where_clause: str = '',
        where_params: Tuple = (),
        limit: int = -1,
        offset: int = 0,
        order_by: Optional[str] = None,
        ascending: bool = True,
        include_metadata: bool = False,
    ):
        result = DocumentArray()
        if len(cells) > 1 and offset > 0:
            raise ValueError('Offset is not supported for multiple cells')

        for cell_id in cells:
            cell_table = self.cell_table(cell_id)
            cell_size = cell_table.count()
            if cell_size == 0:
                continue

            indices = cell_table.query(
                where_clause=where_clause,
                where_params=where_params,
                order_by=order_by,
                limit=limit,
                offset=offset,
                ascending=ascending,
            )

            if len(indices) == 0:
                continue

            for offset in indices:
                doc_id = self.cell_table(cell_id).get_docid_by_offset(offset)
                doc = Document(id=doc_id)
                if include_metadata or (len(cells) > 1 and order_by):
                    doc = self.doc_store(cell_id).get([doc_id])[0]

                result.append(doc)

            if not order_by and len(result) >= limit > 0:
                break

        # reordering the results from multiple cells
        if order_by and len(cells) > 1:
            result = sorted(
                result, key=lambda d: d.tags.get(order_by), reverse=not ascending
            )
            if limit > 0:
                result = result[:limit]
            result = DocumentArray(result)

        return result

    def search_cells(
        self,
        query: 'np.ndarray',
        cells: 'np.ndarray',
        where_clause: str = '',
        where_params: Tuple = (),
        limit: int = 10,
        include_metadata: bool = False,
    ):
        if self._projector_codec:
            query = self._projector_codec.encode(query)

        topk_dists, topk_docs = [], []
        for x, cell_idx in zip(query, cells):
            # x.shape = (self.n_dim,)
            dists, doc_ids, cells = self.ivf_search(
                x,
                cells=cell_idx,
                where_clause=where_clause,
                where_params=where_params,
                limit=limit,
            )

            topk_dists.append(dists)
            match_docs = DocumentArray()
            for dist, doc_id, cell_id in zip(dists, doc_ids, cells):
                doc = Document(id=doc_id)
                if include_metadata:
                    doc = self.doc_store(cell_id).get([doc_id])[0]

                doc.scores[self.metric.name.lower()].value = dist
                match_docs.append(doc)
            topk_docs.append(match_docs)

        return topk_dists, topk_docs

    def _search_cells(
        self,
        query: 'np.ndarray',
        cells: 'np.ndarray',
        where_clause: str = '',
        where_params: Tuple = (),
        limit: int = 10,
    ):
        if self._projector_codec:
            query = self._projector_codec.encode(query)

        topk_dists, topk_ids = [], []
        for x, cell_idx in zip(query, cells):
            dists, ids, cells = self.ivf_search(
                x,
                cells=cell_idx,
                where_clause=where_clause,
                where_params=where_params,
                limit=limit,
            )
            topk_dists.append(dists)
            topk_ids.append(ids)

        return topk_dists, [np.array(ids, dtype=int) for ids in topk_ids]

    def insert(
        self,
        data: 'np.ndarray',
        cells: 'np.ndarray',
        docs: 'DocumentArray',
        only_index: bool = False,
    ):
        assert len(docs) == len(data)

        if self._projector_codec:
            data = self._projector_codec.encode(data)

        unique_cells, unique_cell_counts = np.unique(cells, return_counts=True)

        if len(unique_cells) == 1:
            cell_id = unique_cells[0]

            offsets = self.cell_table(cell_id).insert(docs)
            offsets = np.array(offsets, dtype=np.int64)

            self.vec_index(cell_id).add_with_ids(data, offsets)

            if not only_index:
                self.doc_store(cell_id).insert(docs)
                self._meta_table.bulk_add_address([d.id for d in docs], cells, offsets)
        else:
            for cell_id, cell_count in zip(unique_cells, unique_cell_counts):
                # TODO: Jina should allow boolean filtering in docarray to avoid this
                # and simply use cells == cell_index
                indices = np.where(cells == cell_id)[0]
                cell_docs = docs[indices.tolist()]

                cell_offsets = self.cell_table(cell_id).insert(cell_docs)
                cell_offsets = np.array(cell_offsets, dtype=np.int64)

                cell_data = data[indices, :]

                self.vec_index(cell_id).add_with_ids(cell_data, cell_offsets)

                if not only_index:
                    self.doc_store(cell_id).insert(cell_docs)
                    self._meta_table.bulk_add_address(
                        [d.id for d in cell_docs], [cell_id] * cell_count, cell_offsets
                    )
        logger.debug(f'{len(docs)} new docs added')

    def _add_vecs(self, data: 'np.ndarray', cells: 'np.ndarray', offsets: 'np.ndarray'):
        assert data.shape[0] == cells.shape[0]
        assert data.shape[1] == self.n_dim

        unique_cells, _ = np.unique(cells, return_counts=True)

        for cell_id in unique_cells:
            indices = cells == cell_id
            x = data[indices, :]
            ids = offsets[indices]

            self.vec_index(cell_id).add_with_ids(x, ids)

    def update(
        self,
        data: 'np.ndarray',
        cells: 'np.ndarray',
        docs: 'DocumentArray',
        insert_if_not_found: bool = True,
        raise_errors_on_not_found: bool = False,
    ):
        update_success = 0

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
                self.doc_store(cell_id).update([doc])
                self.meta_table.add_address(doc.id, cell_id, _offset)
                update_success += 1
            elif _cell_id is None:
                if raise_errors_on_not_found and not insert_if_not_found:
                    raise Exception(
                        f'The document (id={doc.id}) cannot be updated as'
                        f'it is not found in the index'
                    )
                elif not (raise_errors_on_not_found or insert_if_not_found):
                    warnings.warn(
                        f'The document (id={doc.id}) cannot be updated as '
                        f'it is not found in the index',
                        RuntimeWarning,
                    )
                elif insert_if_not_found:
                    new_data.append(x)
                    new_cells.append(cell_id)
                    new_docs.append(doc)
                    update_success += 1
                else:
                    continue
            else:
                # DELETE and INSERT
                self.vec_index(_cell_id).delete(_offset)
                self.cell_table(_cell_id).delete_by_offset(_offset)
                self.doc_store(_cell_id).delete([doc.id])

                new_data.append(x)
                new_cells.append(cell_id)
                new_docs.append(doc)
                update_success += 1

        if len(new_data) > 0:
            new_data = np.stack(new_data)
            new_cells = np.array(new_cells, dtype=np.int64)

            self.insert(new_data, new_cells, new_docs)

        logger.debug(
            f'total items for updating: {len(docs)}, ' f'success: {update_success}'
        )

    def delete(
        self,
        ids: List[str],
        raise_errors_on_not_found: bool = False,
    ):
        delete_success = 0

        for doc_id in ids:
            cell_id, offset = self._meta_table.get_address(doc_id)
            print(f'{doc_id} {cell_id} {offset}')
            if cell_id is not None:
                self.vec_index(cell_id).delete([offset])
                self.cell_table(cell_id).delete_by_offset(offset)
                self.doc_store(cell_id).delete([doc_id])
                self.meta_table.delete_address(doc_id)
                delete_success += 1
            else:
                if raise_errors_on_not_found:
                    raise Exception(
                        f'The document (id={doc_id}) cannot be updated as'
                        f'it is not found in the index'
                    )
                else:
                    continue

        logger.debug(
            f'total items for updating: {len(ids)}, ' f'success: {delete_success}'
        )

    def _rebuild_database(self):
        """rebuild doc_store and meta_table after annlite download databse from hubble"""

        self._doc_stores = [
            DocStorage(
                self.data_path / f'cell_{_}',
                serialize_config=self.serialize_config or {},
                lock=True,
            )
            for _ in range(self.n_cells)
        ]
        self._meta_table = MetaTable('metas', data_path=self.data_path, in_memory=False)

    def _get_doc_by_id(self, doc_id: str):
        cell_id = 0
        if self.n_cells > 1:
            cell_id, _ = self._meta_table.get_address(doc_id)

        da = self.doc_store(cell_id).get([doc_id])
        return da[0] if len(da) > 0 else None

    def documents_generator(self, cell_id: int, batch_size: int = 1000):
        for docs in self.doc_store(cell_id).batched_iterator(batch_size=batch_size):
            yield docs

    @property
    def cell_tables(self):
        return self._cell_tables

    @property
    def cell_indexes(self):
        return self._vec_indexes

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
