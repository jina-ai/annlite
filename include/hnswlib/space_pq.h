#pragma once
#include "hnswlib.h"
#include <stdint.h>

namespace hnswlib {

typedef struct pq_dist_param_s {
  size_t n_subvectors;
  size_t n_clusters;
  float *dist_mat;
} pq_dist_param_t;

template <typename CODETYPE>
static float PQLookup(const void *pVect1v, const void *pVect2v,
                      const void *qty_ptr) {
  CODETYPE *pVect1 = (CODETYPE *)pVect1v;
  CODETYPE *pVect2 = (CODETYPE *)pVect2v;
  pq_dist_param_t *qty = (pq_dist_param_t *)qty_ptr;
  size_t subspace_mat_size = (qty->n_clusters) * (qty->n_clusters);
  size_t n_clusters = qty->n_clusters;
  float res = 0;

  for (size_t i = 0; i < qty->n_subvectors; i++) {
    res += qty->dist_mat[i * subspace_mat_size + (*pVect1) * n_clusters +
                         (*pVect2)];
    pVect1++;
    pVect2++;
  }
  return (res);
}

template <typename CODETYPE> class PQ_L2Space : public SpaceInterface<float> {
  DISTFUNC<float> fstdistfunc_;
  size_t data_size_, d_subvectors;
  pq_dist_param_t param;
  float *codebook;

public:
  PQ_L2Space(size_t n_subvectors, size_t n_clusters, size_t d_subvectors,
             float *codebook) {
    param.n_subvectors = n_subvectors;
    param.n_clusters = n_clusters;
    this->codebook = codebook;
    this->d_subvectors = d_subvectors;
    data_size_ = n_subvectors * sizeof(CODETYPE);
    compute_dist_mat();
    fstdistfunc_ = PQLookup<CODETYPE>;
  }

  void compute_dist_mat() {
    param.dist_mat = (float *)malloc(param.n_subvectors * param.n_clusters *
                                     param.n_clusters * sizeof(float));
    size_t subspace_mat_size = param.n_clusters * param.n_clusters;
    size_t subspace_size = param.n_clusters * d_subvectors;
    for (size_t i = 0; i < param.n_subvectors; i++) {
      for (size_t j = 0; j < param.n_clusters; j++) {
        param.dist_mat[i * subspace_mat_size + j * param.n_clusters] = 0;
        for (size_t k = j + 1; k < param.n_clusters; k++) {
          float dist = 0;
          float temp;
          for (size_t d = 0; d < d_subvectors; d++) {
            temp = (codebook[i * subspace_size + j * d_subvectors + d] -
                    codebook[i * subspace_size + k * d_subvectors + d]);
            dist += temp * temp;
          }
          param.dist_mat[i * subspace_mat_size + j * param.n_clusters + k] =
              dist;
          param.dist_mat[i * subspace_mat_size + k * param.n_clusters + j] =
              dist;
        }
      }
    }
  }
  size_t get_data_size() { return data_size_; }

  DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void *get_dist_func_param() { return &param; }

  ~PQ_L2Space() { free(param.dist_mat); }
};
} // namespace hnswlib
