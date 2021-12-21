import numpy as np
from loguru import logger
from scipy.cluster.vq import kmeans2, vq
from sklearn.cluster import MiniBatchKMeans

from ...enums import Metric
from .base import BaseCodec


class VQCodec(BaseCodec):
    def __init__(
        self,
        n_clusters: int,
        metric: Metric = Metric.EUCLIDEAN,
        iter: int = 100,
        *args, **kwargs
    ):
        super(VQCodec, self).__init__(require_train=True)
        self.n_clusters = n_clusters

        #assert (
        #    metric == Metric.EUCLIDEAN
        #), f'The distance metric `{metric.name}` is not supported yet!'
        self.metric = metric

        self._codebook = None
        self.iter = iter
        self._mini_batch_kmeans = None

    def fit(self, x: 'np.ndarray'):
        """Given training vectors, run k-means for each sub-space and create
            codewords for each sub-space.

        :param x: Training vectors with shape=(N, D) and dtype=np.float32.
        :param iter: The number of iteration for k-means
        """

        assert x.dtype == np.float32
        assert x.ndim == 2

        self._codebook, _ = kmeans2(x, self.n_clusters, iter=self.iter, minit='points')

        self._is_trained = True

    def partial_fit(self, x: 'np.ndarray'):
        """Given a batch of training vectors, update the internal MiniBatchKMeans.
        This method is specially designed to be used when data does not fit in memory.

        :param x: Training vectors with shape=(N, D)
        """
        assert x.ndim == 2
        if self._mini_batch_kmeans:
            self._mini_batch_kmeans.partial_fit(x)
        else:
            self._mini_batch_kmeans = MiniBatchKMeans(n_clusters=self.n_clusters)

    def build_codebook(self):
        """Constructs a codebook from the current MiniBatchKmeans
           Note that this is not necessary if full KMeans is used calling `.fit`.
        """
        self._codebook = self._mini_batch_kmeans.cluster_centers_
        self._is_trained = True

    def encode(self, x: 'np.ndarray'):
        self._check_trained()
        assert x.dtype == np.float32
        assert x.ndim == 2

        codes, _ = vq(x, self.codebook)
        return codes

    def decode(self, x: 'np.ndarray'):
        return None

    @property
    def codebook(self):
        self._check_trained()
        return self._codebook
