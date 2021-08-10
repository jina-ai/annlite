from typing import Optional, List

import numpy as np


class PQLite:
    """:class:`PQLite` is an implementation of IVF-PQ.

    To create a :class:`PQLite` object, simply:

        .. highlight:: python
        .. code-block:: python
            pqlite = PQLite(d_vector=256, metric='cosine')

    :param n_vector: the dimentionality of input vectors. there are 2 constraints on d_vector:
            (1) it needs to be divisible by n_subvectors; (2) it needs to be a multiple of 4.*
    :param n_subvectors: number of subquantizers, essentially this is the byte size of
            each quantized vector, default is 8.
    :param n_cells:  number of coarse quantizer clusters.
    :param init_size: initial capacity assigned to each voronoi cell of coarse quantizer. ``n_cells * init_size``
            is the number of vectors that can be stored initially. if any cell has reached its capacity, that cell
            will be automatically expanded. If you need to add vectors frequently, a larger value for init_size
            is recommended.
    :param args: Additional positional arguments which are just used for the parent initialization
    :param kwargs: Additional keyword arguments which are just used for the parent initialization

    .. note::
        Remember that the shape of any tensor that contains data points has to be [n_data, d_vector].
    """

    def __init__(
        self,
        d_vector: int,
        n_subvectors: int = 8,
        n_cells: int = 64,
        init_size: Optional[int] = None,
        expand_step_size: int = 128,
        expand_mode: str = 'double',
        metric: str = 'cosine',
        use_residual: bool = False,
        *args,
        **kwargs,
    ):
        assert d_vector % n_subvectors == 0
        self.d_vector = d_vector
        self.n_subvectors = n_subvectors
        self.d_subvector = d_vector // n_subvectors
        self.metric = metric
        self.use_residual = use_residual
        self.n_probe = 1

        if use_residual and (n_cells * 256 * n_subvectors * 4) <= 4 * 1024 ** 3:
            self._use_precomputed = True
        else:
            self._use_precomputed = False

        self._use_smart_probing = True
        self._smart_probing_temperature = 30.0

    def _sanity_check(self, x: 'np.ndarray'):
        assert len(x.shape) == 2
        assert x.shape[1] == self.d_vector

        return x.shape

    def fit(self, x: 'np.ndarray', force_retrain: bool = False):
        n_data, d_vector = self._sanity_check(x)

        print(f'=> start training VQ codec...')

        print(f'=> start training PQ codec...')

    def add(self, x: 'np.ndarray', ids: Optional[List] = None):
        n_data, _ = self._sanity_check(x)

    def search(self, x: 'np.ndarray', top_k: int = 10):
        n_data, _ = self._sanity_check(x)

    def encode(self, x: 'np.ndarray'):
        n_data, _ = self._sanity_check(x)

    def decode(self, x):
        pass

    @property
    def use_smart_probing(self):
        return self._use_smart_probing

    @use_smart_probing.setter
    def use_smart_probing(self, value):
        assert type(value) is bool
        self._use_smart_probing = value

    @property
    def smart_probing_temperature(self):
        return self._smart_probing_temperature

    @smart_probing_temperature.setter
    def smart_probing_temperature(self, value):
        assert value > 0
        assert self.use_smart_probing, 'set use_smart_probing to True first'
        self._smart_probing_temperature = value
