import numpy as np
from scipy.cluster.vq import kmeans2, vq
from .base import BaseCodec


class VQCodec(BaseCodec):
    def __init__(self, n_clusters: int, *args, **kwargs):
        super(VQCodec, self).__init__(require_train=True)
        self.n_clusters = n_clusters

        self._codebook = None

    def fit(self, x: 'np.ndarray', iter: int = 20):
        """Given training vectors, run k-means for each sub-space and create
            codewords for each sub-space.

        :param x: Training vectors with shape=(N, D) and dtype=np.float32.
        :param iter: The number of iteration for k-means
        """

        assert x.dtype == np.float32
        assert x.ndim == 2
        self._codebook, _ = kmeans2(x, self.n_clusters, iter=iter, minit='points')

        self._is_trained = True

    def encode(self, x: 'np.ndarray'):
        self._check_trained()
        assert x.dtype == np.float32
        assert x.ndim == 2

        codes, _ = vq(x, self.codebook)
        return codes

    @property
    def codebook(self):
        self._check_trained()
        return self._codebook
