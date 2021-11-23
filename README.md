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

Here we suggest two use cases, the following one is used for small-scale data (e.g., < 10M docs):
```python
from jina import Document, DocumentArray
import random
import numpy as np 
from pqlite import PQLite

N = 10000 # number of data points
Nt = 2000 # number for training
Nq = 10
D = 128 # dimentionality / number of features

pq = PQLite(dim=D) # Create a pqlite that is able to store 128-dim vectors
```

For large-scale data (e.g., > 10M docs), the creating process combine Product Qunantization, IVF and HNSW

To be specific, the process is:
1) train the VQ to conduct IVF index
2) train the PQ to compress embeddings
3) build the IVF-HNSW indexing using pq codes (dtype=np.uint8)

```python
Xt = np.random.random((Nt, D)).astype(np.float32)  # 2,000 128-dim vectors for training

# the column schema: (name:str, dtype:type, create_index: bool)
pq = PQLite(d_vector=D, n_cells=64, n_subvectors=8, columns=[('x', float, True)])
pq.fit(Xt)
```

2. DocumentArray generator

Since we use PQLite to deal with Document, here is a simple function to generate DocumentArray

```python
def gen_docs(num):
  docs = DocumentArray()
  k = np.random.random((num, D)).astype(
    np.float32
  ) # Here we generate 128-dim vectors 
  # Seperately put vectors above into Documents and sieze them into DocumentArray
  for i in range(num):
    doc = Document(id=i,embedding=k[i],tags={'x': random.random()})
    docs.append(doc)
  return docs
```


2. Add(Index) new data

use PQLite.index to add new data

```python
docs_index = gen_docs(N) # generating a DocumentArray containing 10000 documents which have 128-dim vectors for embeddings

pq.index(docs_index)
```

3. Search with Filtering

```python
query = gen_docs(Nq)  # 10 Documents for query

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

Using PQLite.update to update data, the process is from new to old, which means it will update recent data first.

```python
doc_update = gen_docs(100) # 100 document for update

pq.update(doc_update) # this will update the latest 100 data in pq
```

5. Delete data

Delete data according to their id

```python
pq.delete(ids=['1', '2'])
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
