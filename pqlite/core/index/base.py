import abc
from typing import List, Optional, Union

import numpy as np

from ...enums import ExpandMode, Metric
from ...helper import str2dtype


class BaseIndex(abc.ABC):
    def __init__(
        self,
        dim: int,
        dtype: Union[np.dtype, str] = np.float32,
        metric: Metric = Metric.COSINE,
        initial_size: Optional[int] = None,
        expand_step_size: int = 10240,
        expand_mode: ExpandMode = ExpandMode.STEP,
        *args,
        **kwargs
    ):
        assert expand_step_size > 0
        self.initial_size = initial_size or expand_step_size

        self.expand_step_size = expand_step_size
        self.expand_mode = expand_mode

        self.dim = dim
        self.dtype = str2dtype(dtype) if isinstance(dtype, str) else dtype
        self.metric = metric

        self._size = 0
        self._capacity = self.initial_size

    @property
    def capacity(self) -> int:
        return self._capacity

    @property
    def size(self):
        return self._size

    @abc.abstractmethod
    def add_with_ids(self, x: np.ndarray, ids: List[int], **kwargs):
        ...

    @abc.abstractmethod
    def delete(self, ids: List[int]):
        ...

    @abc.abstractmethod
    def update_with_ids(self, x: np.ndarray, ids: List[int], **kwargs):
        ...

    def reset(self, capacity: Optional[int] = None):
        self._size = 0
        self._capacity = capacity or self.initial_size
