# distutils: language = c++

import numpy as np
cimport cython
from libcpp.vector cimport vector

from libc.stdint cimport (uint8_t, uint16_t, uint32_t, uint64_t,
                          int8_t, int16_t, int32_t, int64_t)

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
cpdef dist_pqcode_to_codebook(long M, float[:,:] dtable, any_int[:] pq_code):
    ```Compute the distance between each codevector and the pq_code of a query.
    ```
    cdef:
        float dist = 0
        int m
    
    for m in range(M):
        dist += dtable[m, pq_code[m]]

    return dist


@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dist_pqcodes_to_codebooks(long M, float[:,:] dtable, any_int[:,:] pq_codes):
    """
    Commpute the distance between each row in pq_codes and each codevector using dtable.

    This function is equivalent to:
    ```
        dists = np.zeros((N, )).astype(np.float32)
        for n in range(N):
            for m in range(M):
                dists[n] += self.dtable[m][codes[n][m]]
    ```
    """

    cdef:
        int m
        int N = pq_codes.shape[0]
        vector[float] dists

    for n in range(N):
        dists.push_back(dist_pqcode_to_codebook(M, dtable, pq_codes[n,:]))

    return dists

             
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:,:] precompute_adc_table(float[:] query, 
                                      long d_subvector,
                                      long n_clusters,
                                      float[:,:,:] codebooks):
    """
    This function is equivalent to
        ```
        def numpy_adc_table(query, n_subvectors, n_clusters, d_subvector, codebooks):
        dtable = np.empty((n_subvectors, n_clusters), dtype=np.float32)
        for m in range(n_subvectors):
            query_sub = query[m * d_subvector: (m + 1) * d_subvector]
            dtable[m, :] = np.linalg.norm(codebooks[m] - query_sub, axis=1) ** 2

        return dtable
        ```
    """
            
    cdef:
        int D = len(query)
        int M = int(D/d_subvector) 
        int n_subvectors = int(D/d_subvector)
        int m, i, k, ind_prototype, j
        float[:, ::1] dtable = np.empty((M, n_clusters), dtype=np.float32)
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
            
            dtable[m, ind_prototype] = dist_subprototype_to_subquery
    
    return dtable


