from typing import Optional, List, Union
from loguru import logger

import numpy as np
from jina import DocumentArray
from .base import Storage
from .table import CellTable, MetaTable
from ..helper import str2dtype
from .base import ExpandMode


class CellStorage(Storage):
    def __init__(
        self,
        code_size: int,
        n_cells: int,
        dtype: str = 'uint8',
        initial_size: Optional[int] = None,
        expand_step_size: Optional[int] = 1024,
        expand_mode: ExpandMode = ExpandMode.STEP,
        columns: Optional[List[tuple]] = None,
        key_length: int = 64,
    ):
        if initial_size is None:
            initial_size = expand_step_size

        super().__init__(
            initial_size=initial_size * n_cells,
            expand_step_size=expand_step_size,
            expand_mode=expand_mode,
        )
        assert n_cells > 0
        assert code_size > 0

        self.n_cells = n_cells
        self.code_size = code_size
        self.dtype = str2dtype(dtype)
        self.initial_size = initial_size

        self._doc_id_dtype = f'|S{key_length}'
        self._vecs_storage = []
        for _ in range(n_cells):
            cell_vecs = np.zeros((initial_size, code_size), dtype=dtype)
            self._vecs_storage.append(cell_vecs)

        self._cell_size = np.zeros(n_cells, dtype=np.int64)
        self._cell_capacity = np.zeros(n_cells, dtype=np.int64) + initial_size

        self._cell_tables = [CellTable(f'cell_table_{c}') for c in range(self.n_cells)]
        if columns is not None:
            for name, dtype, create_index in columns:
                self._add_column(name, dtype, create_index=create_index)
        self._create_tables()

        self._meta_table = MetaTable()

    @property
    def capacity(self) -> int:
        return self._cell_capacity.sum()

    @property
    def vecs_storage(self):
        return self._vecs_storage

    def clean(self):
        # TODO:
        pass

    def insert(
        self,
        data: np.ndarray,
        cells: np.ndarray,
        docs: DocumentArray,
    ):
        assert len(docs) == len(data)

        for doc, cell_id in zip(docs, cells):
            offset = self.cell_table(cell_id).insert([doc])[0]
            self._meta_table.add_address(doc.id, cell_id, offset)

        self._add_vecs(data, cells)

        logger.debug(f'=> {len(docs)} new docs added')

    def _add_vecs(self, data: np.ndarray, cells: np.ndarray):
        assert data.shape[0] == cells.shape[0]
        assert data.shape[1] == self.code_size

        unique_cells, unique_cell_counts = np.unique(cells, return_counts=True)
        self._expand(unique_cells, unique_cell_counts)

        for cell_index in unique_cells:
            indices = cells == cell_index
            x = data[indices, :]

            start = self._cell_size[cell_index]
            end = start + len(x)

            self.vecs_storage[cell_index][start:end, :] = x

        # update number of stored items in each cell
        self._cell_size[unique_cells] += unique_cell_counts

    def _expand(self, cells: np.ndarray, cell_counts: np.ndarray):
        total_expand = 0
        for cell_id, cell_count in zip(cells, cell_counts):
            free_space = (
                self._cell_capacity[cell_id] - self._cell_size[cell_id] - cell_count
            )
            if free_space > 0:
                continue

            n_new = self.expand_step_size - free_space

            new_block = np.zeros((n_new, self.code_size), dtype=self.dtype)
            self.vecs_storage[cell_id] = np.concatenate(
                (self.vecs_storage[cell_id], new_block), axis=0
            )

            self._cell_capacity[cell_id] += n_new
            total_expand += n_new

        logger.debug(
            f'=> total storage capacity is expanded by {total_expand} for {cells.shape[0]} cells',
        )

    def update(
        self,
        data: np.ndarray,
        cells: np.ndarray,
        docs: DocumentArray,
    ):
        new_data = []
        new_cells = []
        new_docs = []
        new_ids = []

        for (
            x,
            doc,
            cell_id,
        ) in zip(data, docs, cells):
            _cell_id, _offset = self._meta_table.get_address(doc.id)
            if cell_id == _cell_id:
                self.vecs_storage[cell_id][_offset, :] = x
                self._undo_delete_at(_cell_id, _offset)
            elif _cell_id is None:
                new_data.append(x)
                new_cells.append(cell_id)
                new_ids.append(doc.id)
                new_docs.append(doc)
            else:
                # relpace
                self._delete_at(_cell_id, _offset)

                new_data.append(x)
                new_cells.append(cell_id)
                new_ids.append(doc.id)
                new_docs.append(doc)

        new_data = np.stack(new_data)
        new_cells = np.array(new_cells, dtype=np.int64)

        self.insert(new_data, new_cells, new_docs)

        logger.debug(f'=> {len(new_docs)} items updated')

    def _delete_at(self, cell_id: int, offset: int):
        self.cell_table(cell_id).delete_by_offset(offset)
        self._cell_size[cell_id] -= 1

    def _undo_delete_at(self, cell_id: int, offset: int):
        self.cell_table(cell_id).undo_delete_by_offset(offset)
        self._cell_size[cell_id] += 1

    def delete(self, ids: List[str]):
        for doc_id in ids:
            cell_id, offset = self._meta_table.get_address(doc_id)
            if cell_id is not None:
                self._delete_at(cell_id, offset)

        logger.debug(f'=> {len(ids)} items deleted')

    def get_size(self, cell_id: int):
        return self._cell_size[cell_id]

    @property
    def size(self):
        return np.sum(self._cell_size, dtype=np.int64)

    @property
    def cell_tables(self):
        return self._cell_tables

    def cell_table(self, cell_id):
        return self._cell_tables[cell_id]

    def _add_column(
        self, name: str, dtype: Union[str, type], create_index: bool = False
    ):
        for table in self.cell_tables:
            table.add_column(name, dtype, create_index=create_index)

    def _create_tables(self):
        for table in self.cell_tables:
            table.create_table()
