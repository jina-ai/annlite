from typing import Optional
import abc
from ..enums import ExpandMode


class Storage(abc.ABC):
    def __init__(
        self,
        initial_size: Optional[int] = None,
        expand_step_size: int = 10240,
        expand_mode: ExpandMode = ExpandMode.ADAPTIVE,
    ):
        if initial_size is None:
            initial_size = expand_step_size
        assert initial_size >= 0
        assert expand_step_size > 0

        self.initial_size = initial_size
        self.expand_step_size = expand_step_size
        self.expand_mode = expand_mode

    @property
    @abc.abstractmethod
    def capacity(self) -> int:
        ...

    @property
    @abc.abstractmethod
    def size(self):
        ...

    @abc.abstractmethod
    def clean(self):
        ...

    @abc.abstractmethod
    def add(self):
        ...

    @abc.abstractmethod
    def delete(self):
        ...
