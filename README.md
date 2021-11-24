# PQLite

`PQLite` is a blaze fast **Approximate Nearest Neighbor Search** (ANNS) library.

## WARNING

- `PQLite` is still in the very early stages of development. APIs can and will change (now is the time to make suggestions!). Important features are missing. Documentation is sparse.
- `PQLite` contains code that must be compiled to be used. The build is prepared in `setup.py`, users only need to `pip install .` from the root directory.

## About

- **Features**: A quick overview of PQlite's features.
- **Roadmap**: The PQLite team's development plan.
- **Introducing PQLite**: A blog post covering some of PQLite's features

## Quick Start

### Setup

```bash
$ git clone https://github.com/jina-ai/pqlite.git \
  && cd pqlite \
  && pip install .
```
## How to use?

1. Create a new `pqlite`

```python
import random
import numpy as np
from pqlite import PQLite

N = 10000 # number of data points
Nt = 2000
Nq = 10
D = 128 # dimentionality / number of features

Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training

# the column schema: (name:str, dtype:type, create_index: bool)
pqlite = PQLite(d_vector=D, n_cells=64, n_subvectors=8, columns=[('x', float, True)])
pqlite.fit(Xt)
```

2. Add new data

```python
X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed

tags = [{'x': random.random()} for _ in range(N)]
pqlite.add(X, ids=list(range(len(X))), doc_tags=tags)
```

3. Search with Filtering

```python
query = np.random.random((Nq, D)).astype(np.float32)  # a 128-dim query vector

# without filtering
dists, ids = pqlite.search(query, k=5)

print(f'the result without filtering:')
for i, (dist, idx) in enumerate(zip(dists, ids)):
    print(f'query [{i}]: {dist} {idx}')

# with filtering
# condition schema: (column_name: str, relation: str, value: any)
conditions = [('x', '<', 0.3)]
dists, ids = pqlite.search(query, conditions=conditions, k=5)

print(f'the result with filtering:')
for i, (dist, idx) in enumerate(zip(dists, ids)):
    print(f'query [{i}]: {dist} {idx}')
```
4. Update data

```python
Xn = np.random.random((10, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed

tags = [{'x': random.random()} for _ in range(10)]
pqlite.update(Xn, ids=list(range(len(Xn))), doc_tags=tags)
```

5. Delete data

```python
pqlite.delete(ids=['1', '2'])
```
## Benchmark

All experiments were performed with a Intel(R) Xeon(R) CPU @ 2.00GHz and Nvidia Tesla T4 GPU.

- [Yandex Research](https://research.yandex.com/datasets/biganns) Benchmarks for Billion-Scale Similarity Search

## TODO

- [Scalene](https://github.com/plasma-umass/scalene) a high-performance, high-precision CPU, GPU, and memory profiler for Python
- [Bolt](https://github.com/dblalock/bolt) 10x faster matrix and vector operations.
- [MADDNESS](https://arxiv.org/abs/2106.10860) Multiplying Matrices Without Multiplying [code](https://github.com/dblalock/bolt)
- [embeddinghub](https://github.com/featureform/embeddinghub) A vector database for machine learning embeddings.
- [mobius](https://github.com/sunbelbd/mobius) Möbius Transformation for Fast Inner Product Search on Graph

## References

- [hyperfine](https://github.com/sharkdp/hyperfine) Good UX example
- [PGM-index](https://github.com/gvinciguerra/PGM-index) State-of-the-art learned data structure that enables fast lookup, predecessor, range searches and updates in arrays of billions of items using orders of magnitude less space than traditional indexes
- [Xor Filters](https://lemire.me/blog/2019/12/19/xor-filters-faster-and-smaller-than-bloom-filters/) Faster and Smaller Than Bloom Filters
- [CVPR20 Tutorial](https://www.youtube.com/watch?v=SKrHs03i08Q&list=PLKQB14e0EJUWaTnwgQogJ3nSLzEFNn9d8&t=849s) Billion-scale Approximate Nearest Neighbor Search
- [XOR-Quantization](https://arxiv.org/pdf/2008.02002.pdf) Fast top-K Cosine Similarity Search through XOR-Friendly Binary Quantization on GPUs
- [NeurIPS21 Challenge](http://big-ann-benchmarks.com/index.html) Billion-Scale Approximate Nearest Neighbor Search Challenge [NeurIPS'21 competition track](https://neurips.cc/Conferences/2021/CompetitionTrack)


## Research foundations of PQLite

- [PAMI 2011](https://hal.inria.fr/inria-00514462v1/document) Product quantization for nearest neighbor search
- [CVPR 2016](https://research.yandex.com/publications/138) Efficient Indexing of Billion-Scale Datasets of Deep Descriptors
- [NIPs 2017](https://papers.nips.cc/paper/2017/file/b6617980ce90f637e68c3ebe8b9be745-Paper.pdf) Multiscale Quantization for Fast Similarity Search
- [NIPs 2018](https://research.yandex.com/publications/187) Non-metric Similarity Graphs for Maximum Inner Product Search
- [ACMMM 2018](https://arxiv.org/abs/1808.03969) Reconfigurable Inverted Index [code](https://github.com/matsui528/rii)
- [ECCV 2018](https://arxiv.org/abs/1802.02422) Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors
- [CVPR 2019](https://research.yandex.com/publications/196) Unsupervised Neural Quantization for Compressed-Domain Similarity Search
- [ICML 2019](https://research.yandex.com/publications/188) Learning to Route in Similarity Graphs
- [ICML 2020](https://research.yandex.com/publications/280) Graph-based Nearest Neighbor Search: From Practice to Theory
