from typing import Optional, List, Union

import numpy as np
from loguru import logger

from ...enums import ExpandMode, Metric
from ...helper import str2dtype


class BaseIndex:
    def __init__(
        self,
        dim: int,
        dtype: Union[np.dtype, str] = np.float32,
        metric: Metric = Metric.EUCLIDEAN,
        initial_size: Optional[int] = None,
        expand_step_size: int = 10240,
        expand_mode: ExpandMode = ExpandMode.STEP,
        **kwargs,
    ):
        assert expand_step_size > 0

        self.initial_size = initial_size or expand_step_size
        self.expand_step_size = expand_step_size
        self.expand_mode = expand_mode

        self.dim = dim
        self.dtype = str2dtype(dtype) if isinstance(dtype, str) else dtype
        self.metric = metric

        self._data = np.zeros((self.initial_size, dim), dtype=self.dtype)
        self._size = 0
        self._capacity = self.initial_size

    def add_with_ids(self, x: np.ndarray, ids: List[int]):
        for idx in ids:
            if idx >= self._capacity:
                self._expand_capacity()

        start = self._size
        end = start + len(x)

        self._data[ids, :] = x
        self._size = end

    def _expand_capacity(self):
        new_block = np.zeros((self.expand_step_size, self.dim), dtype=self.dtype)
        self._data = np.concatenate((self._data, new_block), axis=0)

        self._capacity += self.expand_step_size
        logger.debug(
            f'=> total storage capacity is expanded by {self.expand_step_size}',
        )
