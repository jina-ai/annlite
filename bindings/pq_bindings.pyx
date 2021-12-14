# distutils: language = c++

import numpy as np

cimport cython
from libc.stdint cimport (
    int8_t,
    int16_t,
    int32_t,
    int64_t,
    uint8_t,
    uint16_t,
    uint32_t,
    uint64_t,
)
from libcpp.vector cimport vector

ctypedef fused any_int:
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    int8_t
    int16_t
    int32_t
    int64_t

@cython.boundscheck(False)
@cython.wraparound(False)
cdef inline dist_pqcode_to_codebook(long M,const float[:,:] adtable, any_int[:] pq_code):
    """Compute the distance between each codevector and the pq_code of a query.

    :param M: Number of sub-vectors in the original feature space.
    :param adtable: 2D Memoryview[float] containing precomputed Asymmetric Distances.
    :param pq_code: 1D Memoriview[any_int] containing a pq code.

    :return: Distance between pq code and query according to the Asymmetric Distance table.

    """
    cdef:
        float dist = 0
        int m

    for m in range(M):
        dist += adtable[m, pq_code[m]]

    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dist_pqcodes_to_codebooks(const float[:,:] adtable, any_int[:,:] pq_codes):
    """
    Compute the distance between each row in pq_codes and each codevector using a adtable.

    :param adtable: 2D Memoryview of precomputed Asymmetric Distances.
    :param pq_codes: 2D Memoryview of pq_codes.

    :return: List of Asymmetric Distances distances between pq_codes and the query.

    This function is equivalent to:
    '''
        dists = np.zeros((N, )).astype(np.float32)
        for n in range(N):
            for m in range(M):
                dists[n] += self.adtable[m][codes[n][m]]
    '''

    """

    cdef:
        int m
        int N = pq_codes.shape[0]
        int M = pq_codes.shape[1]
        vector[float] dists

    for n in range(N):
        dists.push_back(dist_pqcode_to_codebook(M, adtable, pq_codes[n,:]))

    return dists


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef precompute_adc_table(const float[:] query,
                                      long d_subvector,
                                      long n_clusters,
                                      const float[:,:,:] codebooks):
    """
    Compute the  Asymmetric Distance Table between a query and a PQ space.

    :param query: Memoryview a query in the original feature space (not a pqcode).
    :param d_subvector: Number of dimensions in a subvector.
    :param n_clusters: Number of clusters per sub-space (number of prototypes per sub-space).
    :param codebooks: Memoryview containing the learned codevectors for each slice.
        This is a 3D view with (slice index, prototype index, vector values).

    :return: Memoryview with a 2D matrix containing the Asymmetric Distance Computation.

    This function is equivalent to
        '''
        def numpy_adc_table(query, n_subvectors, n_clusters, d_subvector, codebooks):
        adtable = np.empty((n_subvectors, n_clusters), dtype=np.float32)
        for m in range(n_subvectors):
            query_sub = query[m * d_subvector: (m + 1) * d_subvector]
            adtable[m, :] = np.linalg.norm(codebooks[m] - query_sub, axis=1) ** 2

        return adtable
        '''
    But avoids generating views and calling numpy functions.
    """

    cdef:
        int D = len(query)
        int M = int(D/d_subvector)
        int n_subvectors = int(D/d_subvector)
        int m, i, k, ind_prototype, j
        float[:, ::1] adtable = np.empty((M, n_clusters), dtype=np.float32)
        float[:] query_subvec = np.empty(d_subvector, dtype=np.float32)
        float[:] query_subcodeword = np.empty(d_subvector, dtype=np.float32)
        float dist_subprototype_to_subquery, coord_j

    for m in range(n_subvectors):

        # load m'th subquery
        i = 0
        for k in range(m * d_subvector, (m + 1) * d_subvector):
            query_subvec[i] = query[k]
            i += 1

        for ind_prototype in range(n_clusters):

            # load prototype ind_prototype for the m'th subspace
            for i in range(d_subvector):
                query_subcodeword[i] = codebooks[m, ind_prototype, i]

            # compute the distance between subprototype and subquery
            dist_subprototype_to_subquery = 0.
            for j in range(d_subvector):
                coord_j = query_subcodeword[j] - query_subvec[j]
                dist_subprototype_to_subquery += coord_j * coord_j

            adtable[m, ind_prototype] = dist_subprototype_to_subquery

    return adtable
