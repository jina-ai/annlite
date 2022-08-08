import numpy as np
from scipy.cluster.vq import vq

from ...enums import Metric
from .base import BaseCodec


class VQCodec(BaseCodec):
    def __init__(
        self,
        n_clusters: int,
        metric: Metric = Metric.EUCLIDEAN,
        iter: int = 100,
        n_init: int = 4,
        *args,
        **kwargs
    ):
        super(VQCodec, self).__init__(require_train=True)
        self.n_clusters = n_clusters

        # assert (
        #    metric == Metric.EUCLIDEAN
        # ), f'The distance metric `{metric.name}` is not supported yet!'
        self.metric = metric
        self._codebook = None
        self.iter = iter
        self.kmeans = None
        self.n_init = n_init

    def __hash__(self):
        return hash((self.__class__.__name__, self.n_clusters, self.metric))

    def fit(self, x: 'np.ndarray'):
        """Given training vectors, run k-means for each sub-space and create
           codewords for each sub-space.

        :param x: Training vectors with shape=(N, D) and dtype=np.float32.
        :param iter: The number of iteration for k-means
        """

        from sklearn.cluster import KMeans

        assert x.dtype == np.float32
        assert x.ndim == 2

        self.kmeans = KMeans(self.n_clusters, max_iter=self.iter, n_init=self.n_init)
        self.kmeans.fit(x)
        self._codebook = self.kmeans.cluster_centers_
        self._is_trained = True

    def partial_fit(self, x: 'np.ndarray'):
        """Given a batch of training vectors, update the internal MiniBatchKMeans.
        This method is specially designed to be used when data does not fit in memory.

        :param x: Training vectors with shape=(N, D)
        """
        assert x.ndim == 2
        if self.kmeans:
            self.kmeans.partial_fit(x)
        else:
            from sklearn.cluster import MiniBatchKMeans

            self.kmeans = MiniBatchKMeans(
                n_clusters=self.n_clusters, max_iter=self.iter
            )
            self.kmeans.partial_fit(x)

    def build_codebook(self):
        """Constructs a codebook from the current MiniBatchKmeans
        This step is not necessary if full KMeans is trained used calling `.fit`.
        """
        self._codebook = self.kmeans.cluster_centers_
        self._is_trained = True

    def encode(self, x: 'np.ndarray'):
        """Encodes each row of the input array `x` it's closest cluster id."""
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
