from typing import Optional
import abc


class Container(abc.ABC):
    def __init__(
        self,
        initial_size: Optional[int] = None,
        expand_step_size: int = 1024,
        expand_mode: str = 'double',
    ):
        if initial_size is None:
            initial_size = expand_step_size
        assert expand_mode in ['step', 'double']
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

    #
    # def get_id_by_address(self, address):
    #     mask = (0 <= address) & (address < self.capacity)
    #     ids = np.ones_like(address) * -1
    #     ids[mask] = self._address2id[address[mask]]
    #     return ids
    #
    # def _get_address_by_id(self, ids):
    #     n_ids = ids.shape[0]
    #     address = np.zeros(n_ids, dtype=np.int64)
    #     for i in range(n_ids):
    #         id = ids[i]
    #         adr = np.nonzero(self._address2id == id)
    #         if adr.shape[0] > 0:
    #             address[i] = adr[0, 0]
    #         else:
    #             address[i] = -1
    #     return address
    #
    # def get_address_by_id(self, ids):
    #     if self.use_inverse_id_mapping:
    #         if self._id2address is None:
    #             self.create_inverse_id_mapping()
    #
    #         # assume ids are non-negative
    #         mask = (0 <= ids) & (ids <= self.max_id)
    #         address = np.ones_like(ids) * -1
    #         address[mask] = self._id2address[ids[mask]]
    #     else:
    #
    #         address = self._get_address_by_id(ids)
    #
    #     return address
    #
    # def create_inverse_id_mapping(self):
    #     del self._id2address
    #     a2i_v, a2i_i = self._address2id.sort()
    #     a2i_mask = a2i_v >= 0
    #     _id2address = np.ones(self.max_id + 1, dtype=np.int64) * -1
    #     _id2address[a2i_v[a2i_mask]] = a2i_i[a2i_mask]
    #     self._id2address = _id2address

    # def expand(self):
    #     if self.expand_mode == 'double':
    #         self.expand_step_size *= 2
    #
    #     _address2id = self._address2id
    #     del self._address2id
    #     new_a2i = np.ones(self.expand_step_size, dtype=np.int64) * -1
    #     self._address2id = np.cat([_address2id, new_a2i], dim=0)

    @abc.abstractmethod
    def add(self):
        ...

    @abc.abstractmethod
    def remove(self):
        ...
