#pragma once
#ifndef NO_MANUAL_VECTORIZATION
#ifdef __SSE__
#define USE_SSE
#ifdef __AVX__
#define USE_AVX
#endif
#endif
#endif

#if defined(USE_AVX) || defined(USE_SSE)
#ifdef _MSC_VER
#include <intrin.h>
#include <stdexcept>
#else
#include <x86intrin.h>
#endif

#if defined(__GNUC__)
#define PORTABLE_ALIGN32 __attribute__((aligned(32)))
#else
#define PORTABLE_ALIGN32 __declspec(align(32))
#endif
#endif

#include <iostream>
#include <queue>
#include <string.h>
#include <vector>

namespace hnswlib {
typedef size_t labeltype;

typedef struct local_state_s {
  size_t batch_index;
} local_state_t;

typedef struct pq_local_data_s {
  float *data;
  size_t batch_len;
} pq_local_data_t;

template <typename T> class pairGreater {
public:
  bool operator()(const T &p1, const T &p2) { return p1.first > p2.first; }
};

template <typename T>
static void writeBinaryPOD(std::ostream &out, const T &podRef) {
  out.write((char *)&podRef, sizeof(T));
}

template <typename T> static void readBinaryPOD(std::istream &in, T &podRef) {
  in.read((char *)&podRef, sizeof(T));
}

template <typename MTYPE>
using DISTFUNC = MTYPE (*)(const void *, const void *, const void *,
                           const local_state_t *);

template <typename MTYPE> class SpaceInterface {
public:
  // virtual void search(void *);
  virtual size_t get_data_size() = 0;

  virtual DISTFUNC<MTYPE> get_dist_func() = 0;

  virtual void attach_local_data(const void *) = 0;

  virtual void detach_local_data() = 0;

  virtual void *get_dist_func_param() = 0;

  virtual ~SpaceInterface() {}
};

template <typename dist_t> class AlgorithmInterface {
public:
  virtual void addPoint(const void *datapoint, labeltype label,
                        size_t batch_index) = 0;
  virtual std::priority_queue<std::pair<dist_t, labeltype>>
  searchKnn(const void *, size_t, size_t) const = 0;

  // Return k nearest neighbor in the order of closer fist
  virtual std::vector<std::pair<dist_t, labeltype>>
  searchKnnCloserFirst(const void *query_data, size_t k,
                       size_t batch_index) const;

  virtual void saveIndex(const std::string &location) = 0;
  virtual ~AlgorithmInterface() {}
};

template <typename dist_t>
std::vector<std::pair<dist_t, labeltype>>
AlgorithmInterface<dist_t>::searchKnnCloserFirst(const void *query_data,
                                                 size_t k,
                                                 size_t batch_index) const {
  std::vector<std::pair<dist_t, labeltype>> result;

  // here searchKnn returns the result in the order of further first
  auto ret = searchKnn(query_data, k, batch_index);
  {
    size_t sz = ret.size();
    result.resize(sz);
    while (!ret.empty()) {
      result[--sz] = ret.top();
      ret.pop();
    }
  }

  return result;
}

} // namespace hnswlib

#include "bruteforce.h"
#include "fusefilter.h"
#include "hnswalg.h"
#include "space_ip.h"
#include "space_l2.h"
#include "space_pq.h"
