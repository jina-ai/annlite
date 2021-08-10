import numpy as np

from .base import BaseContainer


class CellContainer(BaseContainer):
    def __init__(
        self,
        code_size,
        n_cells,
        dtype=np.float32,
        initial_size=None,
        expand_step_size=1024,
        expand_mode='double',
        use_inverse_id_mapping=False,
        contiguous_size=1,
        verbose=0,
    ):
        if initial_size is None:
            initial_size = expand_step_size

        super().__init__(
            initial_size=initial_size * n_cells,
            expand_step_size=expand_step_size,
            expand_mode=expand_mode,
            use_inverse_id_mapping=use_inverse_id_mapping,
        )
        assert n_cells > 0
        assert code_size > 0
        assert code_size % contiguous_size == 0

        self.n_cells = n_cells
        self.code_size = code_size
        self.dtype = dtype
        self.contiguous_size = contiguous_size
        self.initial_size = initial_size
        self.verbose = verbose

        self._storage = np.zeros(
            code_size // contiguous_size,
            n_cells * initial_size,
            contiguous_size,
            dtype=dtype,
        )

        self._cell_start = np.arange(n_cells) * initial_size
        self._cell_size = np.zeros(n_cells, dtype=np.int64)
        self._cell_capacity = np.zeros(n_cells, dtype=np.int64) + initial_size
        self._is_empty = np.ones(n_cells * initial_size, dtype=np.uint8)

    @property
    def n_items(self):
        return self._cell_size.sum().item()

    def _get_cell_by_address(self, address, cell_start, cell_end):
        n_address = address.shape[0]
        mask1 = (
            cell_start[
                None,
            ]
            <= address[:, None]
        )
        mask2 = (
            cell_end[
                None,
            ]
            > address[:, None]
        )  # [n_address, n_cq_clusters]
        mask = mask1 & mask2
        not_found = mask.sum(dim=1) == 0
        mask[not_found, 0] = True
        cells = np.nonzero(mask)
        cells[not_found, 1] = -1
        return cells[:, 1]

    def get_cell_by_address(self, address):
        cell_start = self._cell_start
        cell_end = cell_start + self._cell_capacity

        return self._get_cell_by_address(address, cell_start, cell_end)
