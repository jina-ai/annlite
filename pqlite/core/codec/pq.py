import numpy as np
from loguru import logger
from scipy.cluster.vq import kmeans2, vq

from pqlite import pq_bind

from ...enums import Metric
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
    """

    def __init__(
        self,
        dim: int,
        n_subvectors: int = 8,
        n_clusters: int = 256,
        metric: Metric = Metric.EUCLIDEAN,
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
            if n_clusters <= 2 ** 8
            else (np.uint16 if n_clusters <= 2 ** 16 else np.uint32)
        )

        assert (
            metric == Metric.EUCLIDEAN
        ), f'The distance metric `{metric.name}` is not supported yet!'
        self.metric = metric

        self._codebooks = None

    def fit(self, x: 'np.ndarray', iter: int = 100):
        assert x.dtype == np.float32
        assert x.ndim == 2

        # [m][ks][ds]: m-th subspace, ks-the codeword, ds-th dim
        self._codebooks = np.zeros(
            (self.n_subvectors, self.n_clusters, self.d_subvector), dtype=np.float32
        )
        for m in range(self.n_subvectors):
            sub_vecs = x[:, m * self.d_subvector : (m + 1) * self.d_subvector]
            self._codebooks[m], _ = kmeans2(
                sub_vecs, self.n_clusters, iter=iter, minit='points'
            )

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
