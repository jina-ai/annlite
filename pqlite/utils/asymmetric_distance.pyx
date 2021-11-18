
import numpy as np
cimport cython
    

@cython.boundscheck(False)
@cython.wraparound(False)
cpdef dist_pqcode_to_codebooks(long M, float[:,:] dtable, int[:] pq_code):
    cdef:
        float dist = 0
        int m
    
    for m in range(M):
        dist += dtable[m, pq_code[m]]

    return dist

             
@cython.boundscheck(False)
@cython.wraparound(False)
cpdef float[:,:] precompute_adc_table(float[:] query, 
                                      long d_subvector,
                                      long n_clusters,
                                      float[::,:,:] codebooks):
            
    cdef:
        int D = len(query)
        int M = int(D/d_subvector) 
        int n_subvectors = int(D/d_subvector)
        int m, i, k, ind_prototype, j
        float[::, :] dtable = np.empty((M, n_clusters), dtype=np.float32)
        float[::] query_subvec = np.empty(d_subvector, dtype=np.float32)
        float[::] query_subcodeword = np.empty(d_subvector, dtype=np.float32)
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


