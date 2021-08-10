from abc import ABC, abstractmethod


class BaseIndex(ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def add(self):
        pass

    @abstractmethod
    def delete(self):
        pass

    @abstractmethod
    def query(self):
        pass
