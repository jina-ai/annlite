from typing import Optional, List, Union
from loguru import logger

import numpy as np

from .base import Storage
from .table import Table
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
        key_length: int = 36,
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
        self._storage = []
        for _ in range(n_cells):
            cell_storage = np.zeros((initial_size, code_size), dtype=dtype)
            self._storage.append(cell_storage)

        self._cell_size = np.zeros(n_cells, dtype=np.int64)
        self._cell_capacity = np.zeros(n_cells, dtype=np.int64) + initial_size

        self._cell_tables = [Table(f'cell_table_{c}') for c in range(self.n_cells)]
        if columns is not None:
            for name, dtype, create_index in columns:
                self._add_column(name, dtype, create_index=create_index)
        self._create_tables()

    @property
    def capacity(self) -> int:
        return self._cell_capacity.sum()

    @property
    def storage(self):
        return self._storage

    def clean(self):
        pass

    @staticmethod
    def get_ioa(cells, unique_cells=None):
        if unique_cells is None:
            unique_cells = np.unique(cells)  # [n_unique_clusters]

        expanded_cells = np.repeat(
            cells[:, None], unique_cells.shape[0], axis=-1
        )  # [n_data, n_unique_clusters]
        mask = expanded_cells == unique_cells[None, :]  # [n_data, n_unique_clusters]
        mcs = np.cumsum(mask, axis=0)
        mcs[tuple([~mask])] = 0
        ioa = mcs.sum(axis=1) - 1
        return ioa

    def add(
        self,
        data: np.ndarray,
        cells: np.ndarray,
        ids: Optional[List[str]],
        doc_tags: Optional[List[dict]] = None,
    ):
        if ids is not None:
            assert len(ids) == len(data)
        else:
            raise NotImplemented('The auto-generated UUID is not supported yet')

        if doc_tags is None:
            doc_tags = [{'_doc_id': k} for k in ids]
        else:
            for k, doc in zip(ids, doc_tags):
                doc.update({'_doc_id': k})

        for doc, cell_idx in zip(doc_tags, cells):
            self.cell_table(cell_idx).insert([doc])

        self._add_vecs(data, cells, ids=ids)

    def _add_vecs(
        self, data: np.ndarray, cells: np.ndarray, ids: Optional[List[str]] = None
    ):
        assert data.shape[0] == cells.shape[0]
        assert data.shape[1] == self.code_size

        n_data = data.shape[0]

        unique_cells, unique_cell_counts = np.unique(cells, return_counts=True)
        ioa = self.get_ioa(cells, unique_cells)

        # expand storage if necessary
        while True:
            free_space = self._cell_capacity[cells] - self._cell_size[cells] - (ioa + 1)
            expansion_required = np.unique(cells[free_space < 0])
            if expansion_required.shape[0] == 0:
                break
            self.expand(expansion_required)

        for cell_index in unique_cells:
            indices = cells == cell_index
            x = data[indices, :]

            start = self._cell_size[cell_index]
            end = start + len(x)

            self.storage[cell_index][start:end, :] = x

        # update number of stored items in each cell
        self._cell_size[unique_cells] += unique_cell_counts

        logger.debug(f'=> {n_data} new items added')

    def expand(self, cells):
        total = 0
        for cell_index in cells:
            if self.expand_mode == ExpandMode.STEP:
                n_new = self.expand_step_size
            elif self.expand_mode == ExpandMode.DOUBLE:
                n_new = self._cell_capacity[cell_index]
            else:
                now = self._cell_capacity[cell_index]
                if now < 102400:
                    n_new = self.expand_step_size
                elif now >= 1024000:
                    n_new = int(0.1 * now)
                else:
                    n_new = now

            new_block = np.zeros((n_new, self.code_size), dtype=self.dtype)
            self.storage[cell_index] = np.concatenate(
                (self.storage[cell_index], new_block), axis=0
            )

            self._cell_capacity[cell_index] += n_new
            total += n_new

        logger.debug(
            f'=> total storage capacity is expanded by {total} for {cells.shape[0]} cells',
        )

    def delete(self, ids: List[str]):
        # TODO: refactor using SQLite
        # for _id in ids:
        #     cell, offset = self.get_address_by_id(_id)
        #
        #     self._is_empties[cell][offset] = 1
        #     self._ids2address[cell][offset] = ''
        #     self._cell_size[cell] -= 1

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
