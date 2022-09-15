#pragma once
#include "hnswlib.h"
#include <math.h>
#include <memory>
#include <stdint.h>
namespace hnswlib {

typedef struct pq_dist_param_s {
  size_t n_subvectors;
  size_t n_clusters;
  size_t batch_len;
  float *batch_dtable;
} pq_dist_param_t;

template <typename CODETYPE>
static float PQLookup(const void *pVect1v, const void *pVect2v,
                      const void *qty_ptr, const local_state_t *local_state) {
  CODETYPE *pVect2 = (CODETYPE *)pVect2v;
  pq_dist_param_t *qty = (pq_dist_param_t *)qty_ptr;
  if (qty->batch_len <= local_state->batch_index ||
      qty->batch_dtable == nullptr) {
    // Since all the batch_index actually manage by us
    throw std::runtime_error("Row index exceeds or batch distance table "
                             "uninitialized, most likely an internal bug!");
  }
  size_t n_clusters = qty->n_clusters;
  size_t row_step =
      (qty->n_clusters * qty->n_subvectors) * local_state->batch_index;
  const float *dtable = qty->batch_dtable;
  float res = 0;

  for (size_t i = 0; i < qty->n_subvectors; i++) {
    res += dtable[row_step + i * n_clusters + (*pVect2)];
    pVect2++;
  }
  return res;
}

template <typename CODETYPE> class PQ_Space : public SpaceInterface<float> {
  DISTFUNC<float> fstdistfunc_;
  size_t data_size_, d_subvectors;
  pq_dist_param_t param;
  bool ip_enable;

public:
  PQ_Space(const std::string &space_name, size_t n_subvectors,
           size_t n_clusters, size_t d_subvectors, float *codebook)
      : d_subvectors(d_subvectors) {
    param.n_subvectors = n_subvectors;
    param.n_clusters = n_clusters;
    data_size_ = n_subvectors * sizeof(CODETYPE);
    fstdistfunc_ = PQLookup<CODETYPE>;
  }

  void attach_local_data(const void *local_data) {
    hnswlib::pq_local_data_t *pq_param = (hnswlib::pq_local_data_t *)local_data;
    param.batch_dtable = pq_param->data;
    param.batch_len = pq_param->batch_len;
  }

  void detach_local_data() {
    param.batch_dtable = nullptr;
    param.batch_len = 0;
  };

  size_t get_data_size() { return data_size_; }

  DISTFUNC<float> get_dist_func() { return fstdistfunc_; }

  void *get_dist_func_param() { return &param; }

  ~PQ_Space() {}
};
} // namespace hnswlib
