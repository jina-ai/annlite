from argparse import ArgumentError

import numpy as np
from scipy.cluster.vq import vq

from annlite import pq_bind

from ...enums import Metric
from ...math import l2_normalize
from ...profile import time_profile
from .base import BaseCodec

# from pqlite.pq_bind import precompute_adc_table, dist_pqcodes_to_codebooks


class PQCodec(BaseCodec):
    """Implementation of Product Quantization (PQ) [Jegou11]_.

    For the indexing phase of database vectors,
    a `D`-dim input vector is divided into `M` `D`/`M`-dim sub-vectors.
    Each sub-vector is quantized into a small integer via `Ks` codewords.

    For the querying phase, given a new `D`-dim query vector, the distance between the query
    and the database PQ-codes are efficiently approximated via Asymmetric Distance.
    All vectors must be np.ndarray with np.float32

    .. [Jegou11] H. Jegou et al., "Product Quantization for Nearest Neighbor Search", IEEE TPAMI 2011

    :param d_vector: the dimensionality of input vectors
    :param n_subvectors: The number of sub-space
    :param n_clusters: The number of codewords for each subspace
            (typically 256, so that each sub-vector is quantized
            into 256 bits pqlite.utils.asymmetric_distance= 1 byte = uint8)
    :param n_init: Number of times K-Means is trained with different centroid seeds. Best result of
                   the `n_init` consecutive runs is selected.
    """

    def __init__(
        self,
        dim: int,
        n_subvectors: int = 8,
        n_clusters: int = 256,
        metric: Metric = Metric.EUCLIDEAN,
        n_init: int = 4,
    ):
        super(PQCodec, self).__init__(require_train=True)
        self.dim = dim
        self.n_subvectors = n_subvectors
        self.n_clusters = n_clusters

        assert (
            dim % n_subvectors == 0
        ), 'input dimension must be dividable by number of sub-space'
        self.d_subvector = dim // n_subvectors

        self.code_dtype = (
            np.uint8
            if n_clusters <= 2**8
            else (np.uint16 if n_clusters <= 2**16 else np.uint32)
        )

        # assert (
        #    metric == Metric.EUCLIDEAN
        # ), f'The distance metric `{metric.name}` is not supported yet!'
        self.metric = metric

        self.normalize_input = False
        if self.metric == Metric.COSINE:
            self.normalize_input = True

        self._codebooks = np.zeros(
            (self.n_subvectors, self.n_clusters, self.d_subvector), dtype=np.float32
        )
        self.kmeans = []
        self.n_init = n_init

    def __hash__(self):
        return hash(
            (
                self.__class__.__name__,
                self.dim,
                self.n_subvectors,
                self.n_clusters,
                self.metric,
                self.code_dtype,
            )
        )

    def fit(self, x: 'np.ndarray', iter: int = 100):
        """Train the K-Means for each cartesian product

        :param x: Training vectors with shape=(N, D)
        :param iter: Number of iterations in Kmeans
        """
        from sklearn.cluster import KMeans

        assert x.dtype == np.float32
        assert x.ndim == 2

        if self.normalize_input:
            x = l2_normalize(x)

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self._codebooks = np.zeros(
            (self.n_subvectors, self.n_clusters, self.d_subvector), dtype=np.float32
        )
        for m in range(self.n_subvectors):
            kmeans = KMeans(
                n_clusters=self.n_clusters, max_iter=iter, n_init=self.n_init
            )
            self.kmeans.append(kmeans)
            self.kmeans[m].fit(x[:, m * self.d_subvector : (m + 1) * self.d_subvector])
            self._codebooks[m] = self.kmeans[m].cluster_centers_

        self._is_trained = True

    def partial_fit(self, x: 'np.ndarray'):
        """Given a batch of training vectors, update the internal MiniBatchKMeans.
        This method is specially designed to be used when data does not fit in memory.

        :param x: Training vectors with shape=(N, D)
        """
        assert x.ndim == 2

        if self.normalize_input:
            x = l2_normalize(x)

        if len(self.kmeans) > 0:
            for m in range(self.n_subvectors):
                self.kmeans[m].partial_fit(
                    x[:, m * self.d_subvector : (m + 1) * self.d_subvector]
                )
        else:
            from sklearn.cluster import MiniBatchKMeans

            for m in range(self.n_subvectors):
                self.kmeans.append(MiniBatchKMeans(n_clusters=self.n_clusters))

            for m in range(self.n_subvectors):
                self.kmeans[m].partial_fit(
                    x[:, m * self.d_subvector : (m + 1) * self.d_subvector]
                )

    def build_codebook(self):
        """Constructs sub-codebooks from the current parameters of the models in `self.kmeans`
        This step is not necessary if full KMeans is trained used calling `.fit`.
        """

        self._codebooks = np.zeros(
            (self.n_subvectors, self.n_clusters, self.d_subvector), dtype=np.float32
        )

        for m in range(self.n_subvectors):
            self._codebooks[m] = self.kmeans[m].cluster_centers_

        self._is_trained = True

    def encode(self, x: 'np.ndarray'):
        """Encode input vectors into PQ-codes.

        :param x: Input vectors with shape=(N, D) and dtype=np.float32.
        :return: np.ndarray: PQ codes with shape=(N, M) and dtype=self.code_dtype
        """
        assert x.dtype == np.float32
        assert x.ndim == 2
        N, D = x.shape
        assert (
            D == self.d_subvector * self.n_subvectors
        ), 'input dimension must be Ds * M'

        # codes[n][m] : code of n-th vec, m-th subspace
        codes = np.empty((N, self.n_subvectors), dtype=self.code_dtype)
        for m in range(self.n_subvectors):
            sub_vecs = x[:, m * self.d_subvector : (m + 1) * self.d_subvector]
            codes[:, m], _ = vq(sub_vecs, self.codebooks[m])

        return codes

    def decode(self, codes: 'np.ndarray'):
        """Given PQ-codes, reconstruct original D-dimensional vectors
        approximately by fetching the codewords.

        :param codes: PQ-cdoes with shape=(N, M) and dtype=self.code_dtype.
            Each row is a PQ-code
        :return: Reconstructed vectors with shape=(N, D) and dtype=np.float32
        """
        assert codes.ndim == 2
        N, M = codes.shape
        assert M == self.n_subvectors
        assert codes.dtype == self.code_dtype

        vecs = np.empty((N, self.d_subvector * self.n_subvectors), dtype=np.float32)
        for m in range(self.n_subvectors):
            vecs[:, m * self.d_subvector : (m + 1) * self.d_subvector] = self.codebooks[
                m
            ][codes[:, m], :]

        return vecs

    def precompute_adc(self, query: object) -> object:
        """Compute a distance table for a query vector.
        The distances are computed by comparing each sub-vector of the query
        to the codewords for each sub-subspace.
        `dtable[m][ks]` contains the squared Euclidean distance between
        the `m`-th sub-vector of the query and the `ks`-th codeword
        for the `m`-th sub-space (`self.codewords[m][ks]`).

        :param query: Input vector with shape=(D, ) and dtype=np.float32
        :return: Distance table. which contains dtable with shape=(M, Ks)
            and dtype=np.float32
        """
        assert query.dtype == np.float32
        assert query.ndim == 1, 'input must be a single vector'

        # dtable[m] : distance between m-th subvec and m-th codewords (m-th subspace)
        # dtable[m][ks] : distance between m-th subvec and ks-th codeword of m-th codewords

        # Warning: the following line produces `ValueError: buffer source array is read-only`
        # if no `const` is used in the cython implementation using a memoryview
        dtable = pq_bind.precompute_adc_table(
            query, self.d_subvector, self.n_clusters, self.codebooks
        )

        return DistanceTable(dtable)

    @property
    def codebooks(self):
        return self._codebooks

    # trained pq interface ----------------
    def get_codebook(self) -> 'np.ndarray':
        """Return the codebook parameters.

        Expect a 3-dimensional matrix is returned,
        with shape (`n_subvectors`, `n_clusters`, `d_subvector`) and dtype float32
        """
        return np.ascontiguousarray(self.codebooks, dtype='float32')

    def get_subspace_splitting(self):
        """Return subspace splitting setting

        :return: tuple of (`n_subvectors`, `n_clusters`, `d_subvector`)
        """
        return (self.n_subvectors, self.n_clusters, self.d_subvector)

    # def get_dist_mat(self, x: np.ndarray):
    #     """Return the distance tables in form of matrix for multiple queries

    #     :param query: shape('N', 'D'),

    #     :return: ndarray with shape('N', `n_subvectors`, `n_clusters`)

    #     .. note::
    #         _description_
    #     """
    #     assert x.dtype == np.float32
    #     assert x.ndim == 2
    #     N, D = x.shape
    #     assert (
    #         D == self.d_subvector * self.n_subvectors
    #     ), 'input dimension must be Ds * M'
    #     if self.normalize_input:
    #         x = l2_normalize(x)

    #     x = x.reshape(
    #         N,
    #         self.n_subvectors,
    #         1,
    #         self.d_subvector,
    #     )
    #     if self.metric == Metric.EUCLIDEAN:
    #         # (1, n_subvectors, n_clusters, d_subvector)
    #         codebook = self.codebooks[np.newaxis, ...]

    #         # broadcast to (N, n_subvectors, n_clusters, d_subvector)
    #         dist_vector = (x - codebook) ** 2

    #         # reduce to (N, n_subvectors, n_clusters)
    #         dist_mat = np.sum(dist_vector, axis=3)
    #     elif self.metric in [Metric.INNER_PRODUCT, Metric.COSINE]:
    #         # (1, n_subvectors, n_clusters, d_subvector)
    #         codebook = self.codebooks[np.newaxis, ...]

    #         # broadcast to (N, n_subvectors, n_clusters, d_subvector)
    #         dist_vector = x * codebook

    #         # reduce to (N, n_subvectors, n_clusters)
    #         dist_mat = 1 / self.n_clusters - np.sum(dist_vector, axis=3)
    #     else:
    #         raise ArgumentError(f'Unable support metrics {self.metric}')
    #     return np.ascontiguousarray(dist_mat, dtype='float32')

    def get_dist_mat(self, x: np.ndarray):
        """Return the distance tables in form of matrix for multiple queries

        :param query: shape('N', 'D'),

        :return: ndarray with shape('N', `n_subvectors`, `n_clusters`)

        .. note::
            _description_
        """
        assert x.dtype == np.float32
        assert x.ndim == 2
        N, D = x.shape
        assert (
            D == self.d_subvector * self.n_subvectors
        ), 'input dimension must be Ds * M'
        if self.normalize_input:
            x = l2_normalize(x)

        if self.metric == Metric.EUCLIDEAN:
            dist_mat = pq_bind.batch_precompute_adc_table(
                x, self.d_subvector, self.n_clusters, self.codebooks
            )
        elif self.metric in [Metric.INNER_PRODUCT, Metric.COSINE]:
            dist_mat = 1 / self.n_clusters - np.array(
                pq_bind.batch_precompute_adc_table_ip(
                    x, self.d_subvector, self.n_clusters, self.codebooks
                ),
                dtype='float32',
            )
        else:
            raise ArgumentError(f'Unable support metrics {self.metric}')
        return np.ascontiguousarray(dist_mat, dtype='float32')

    # -------------------------------------


class DistanceTable(object):
    """Distance table from query to codeworkds.
    Given a query vector, a PQ/OPQ instance compute this DistanceTable class
    using :func:`PQ.dtable` or :func:`OPQ.dtable`.
    The Asymmetric Distance from query to each database codes can be computed
    by :func:`DistanceTable.adist`.
    Args:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32
            computed by :func:`PQ.dtable` or :func:`OPQ.dtable`
    Attributes:
        dtable (np.ndarray): Distance table with shape=(M, Ks) and dtype=np.float32.
            Note that dtable[m][ks] contains the squared Euclidean distance between
            (1) m-th sub-vector of query and (2) ks-th codeword for m-th subspace.
    """

    def __init__(self, dtable: 'np.ndarray'):

        assert dtable.ndim == 2
        self.dtable = dtable

    def adist(self, codes):
        """Given PQ-codes, compute Asymmetric Distances between the query (self.dtable)
        and the PQ-codes.
        Args:
            codes (np.ndarray): PQ codes with shape=(N, M) and
                dtype=pq.code_dtype where pq is a pq instance that creates the codes
        Returns:
            np.ndarray: Asymmetric Distances with shape=(N, ) and dtype=np.float32
        """

        assert codes.ndim == 2
        dists = pq_bind.dist_pqcodes_to_codebooks(self.dtable, codes)

        # The above line is equivalent to the followings:
        # dists = np.zeros((N, )).astype(np.float32)
        # for n in range(N):
        #     for m in range(M):
        #         dists[n] += self.dtable[m][codes[n][m]]
        return dists
