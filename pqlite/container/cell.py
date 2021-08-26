from typing import Optional, List
from loguru import logger

import numpy as np

from .base import Container
from ..helper import str2dtype


class CellContainer(Container):
    def __init__(
        self,
        code_size: int,
        n_cells: int,
        dtype: str = 'float32',
        initial_size: Optional[int] = None,
        expand_step_size: Optional[int] = 1024,
        expand_mode: str = 'double',
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
        self._key_length = key_length

        self._storages = []
        self._is_empties = []
        self._address2id = []
        self._ids2address = {}
        for _ in range(n_cells):
            storage = np.zeros((initial_size, code_size), dtype=dtype)
            is_empty = np.ones(initial_size, dtype=np.uint8)
            ids = np.empty(initial_size, dtype=f'|S{self._key_length}')
            self._storages.append(storage)
            self._is_empties.append(is_empty)
            self._address2id.append(ids)

        self._cell_size = np.zeros(n_cells, dtype=np.int64)
        self._cell_capacity = np.zeros(n_cells, dtype=np.int64) + initial_size

    def get_id_by_address(self, cell: int, offset: int):
        return self._address2id[cell][offset]

    def get_address_by_id(self, id: str):
        return self._ids2address[id]

    @property
    def capacity(self) -> int:
        return self._cell_capacity.sum()

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

    def add(self, data: np.ndarray, cells: np.ndarray, ids: Optional[List[str]] = None):
        assert data.shape[0] == cells.shape[0]
        assert data.shape[1] == self.code_size

        n_data = data.shape[0]
        if ids is not None:
            assert len(ids) == n_data
        else:
            raise NotImplemented('The auto-generated UUID is not supported yet')

        ids = np.array(ids, dtype=f'|S{self._key_length}')

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

            _ids = ids[indices]

            start = self._cell_size[cell_index]
            end = start + len(x)

            self._storages[cell_index][start:end, :] = x
            self._address2id[cell_index][start:end] = _ids
            self._is_empties[cell_index][start:end] = 0
            for i, _id in enumerate(_ids):
                self._ids2address[_id] = (cell_index, start + i)

        # update number of stored items in each cell
        self._cell_size[unique_cells] += unique_cell_counts

        logger.debug(f'=> {n_data} new items added')

    def expand(self, cells):
        total = 0
        for cell_index in cells:
            if self.expand_mode == 'step':
                n_new = self.expand_step_size
            elif self.expand_mode == 'double':
                n_new = self._cell_capacity[cell_index]
            new_block = np.zeros((n_new, self.code_size), dtype=self.dtype)
            self._storages[cell_index] = np.concatenate(
                (self._storages[cell_index], new_block), axis=0
            )

            new_is_empty = np.ones(n_new, dtype=np.uint8)
            self._is_empties[cell_index] = np.concatenate(
                (self._is_empties[cell_index], new_is_empty), axis=0
            )

            new_ids = np.empty(n_new, dtype=f'|S{self._key_length}')
            self._address2id[cell_index] = np.concatenate(
                (self._address2id[cell_index], new_ids), axis=0
            )

            self._cell_capacity[cell_index] += n_new
            total += n_new

        logger.debug(
            f'=> total storage capacity is expanded by {total} for {cells.shape[0]} cells',
        )

    def remove(self, ids: List[str]):
        for _id in ids:
            cell, offset = self.get_address_by_id(_id)

            self._is_empties[cell][offset] = 1
            self._ids2address[cell][offset] = ''
            self._cell_size[cell] -= 1

        logger.debug(f'=> {len(ids)} items deleted')

    @property
    def size(self):
        return np.sum(self._cell_size, dtype=np.int64)
