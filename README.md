# PQLite

`PQLite` is an  **Approximate Nearest Neighbor Search** (ANNS) library integrated with the Jina ecosystem.
The `PQLite` class partitions the data into cells at index time, and instantiates a "sub-indexer" in each cell.  Search is performed aggregating results retrieved from cells.

This indexer is recommended to be used when an application requires **search with filters** applied on `Document` tags.
Each filter is a **triplet** (_column_, _operator_, _value_). More than one filter can be applied during search. Therefore, conditions for a filter are specified as a list of triplets.
Each triplet contains:

- column: Column used to filter.
- operator: Binary operation between two values. Supported operators are `['>','<','<=','>=', '=', '!=']`.
- value: value used to compare a candidate.


## WARNING

- `PQLite` is still in the very early stages of development. APIs can and will change (now is the time to make suggestions!). Important features are missing. Documentation is sparse.
- `PQLite` contains code that must be compiled to be used. The build is prepared in `setup.py`. Users only need to `pip install .` from the root directory.

## Quick Start

### Setup

```bash
$ git clone https://github.com/jina-ai/pqlite.git \
  && cd pqlite \
  && pip install -e .
```
## How to use?

1. Create a new `pqlite`

```python
import random
import numpy as np
from jina import Document, DocumentArray
from pqlite import PQLite

N = 10000 # number of data points
Nq = 10 # number of query data
D = 128 # dimentionality / number of features

# the column schema: (name:str, dtype:type, create_index: bool)
pqlite = PQLite(dim=D, columns=[('x', float, True)], data_path='./data')
```

2. Add new data

```python
X = np.random.random((N, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
docs = DocumentArray(
    [
        Document(id=f'{i}', embedding=X[i], tags={'x': random.random()})
        for i in range(N)
    ]
)
pqlite.index(docs)
```

3. Search with filtering

```python
Xq = np.random.random((Nq, D)).astype(np.float32)  # a 128-dim query vector
query = DocumentArray([Document(embedding=Xq[i]) for i in range(Nq)])

# without filtering
pqlite.search(query, limit=10)

print(f'the result without filtering:')
for i, q in enumerate(query):
    print(f'query [{i}]:')
    for m in q.matches:
        print(f'\t{m.id} ({m.scores["euclidean"].value})')

# with filtering
# condition schema: (column_name: str, relation: str, value: any)
conditions = [('x', '<', 0.3)]
pqlite.search(query, conditions=conditions, limit=10)
print(f'the result with filtering:')
for i, q in enumerate(query):
    print(f'query [{i}]:')
    for m in q.matches:
        print(f'\t{m.id} {m.scores["euclidean"].value} (x={m.tags["x"]})')
```

4. Update data

```python
Xn = np.random.random((10, D)).astype(np.float32)  # 10,000 128-dim vectors to be indexed
docs = DocumentArray(
    [
        Document(id=f'{i}', embedding=Xn[i], tags={'x': random.random()})
        for i in range(10)
    ]
)
pqlite.update(docs)
```

5. Delete data

```python
pqlite.delete(['1', '2'])
```

## Benchmark

One can run `executor/benchmark.py` to get a quick performance overview.

|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|
|---|---|---|---|---|
|10000 | 19.766 | 0.002 | 0.016 | 0.133|
|100000 | 325.413 | 0.010 | 0.084 | 0.671|
|500000 | 2413.311 | 0.038 | 0.303 | 2.427|
|1000000 | 5310.749 | 0.075 | 0.579 | 4.634|

## Research foundations of PQLite

- [Xor Filters](https://lemire.me/blog/2019/12/19/xor-filters-faster-and-smaller-than-bloom-filters/) Faster and Smaller Than Bloom Filters
- [CVPR20 Tutorial](https://www.youtube.com/watch?v=SKrHs03i08Q&list=PLKQB14e0EJUWaTnwgQogJ3nSLzEFNn9d8&t=849s) Billion-scale Approximate Nearest Neighbor Search
- [XOR-Quantization](https://arxiv.org/pdf/2008.02002.pdf) Fast top-K Cosine Similarity Search through XOR-Friendly Binary Quantization on GPUs
- [NeurIPS21 Challenge](http://big-ann-benchmarks.com/index.html) Billion-Scale Approximate Nearest Neighbor Search Challenge [NeurIPS'21 competition track](https://neurips.cc/Conferences/2021/CompetitionTrack)
- [PAMI 2011](https://hal.inria.fr/inria-00514462v1/document) Product quantization for nearest neighbor search
- [CVPR 2016](https://research.yandex.com/publications/138) Efficient Indexing of Billion-Scale Datasets of Deep Descriptors
- [NIPs 2017](https://papers.nips.cc/paper/2017/file/b6617980ce90f637e68c3ebe8b9be745-Paper.pdf) Multiscale Quantization for Fast Similarity Search
- [NIPs 2018](https://research.yandex.com/publications/187) Non-metric Similarity Graphs for Maximum Inner Product Search
- [ACMMM 2018](https://arxiv.org/abs/1808.03969) Reconfigurable Inverted Index [code](https://github.com/matsui528/rii)
- [ECCV 2018](https://arxiv.org/abs/1802.02422) Revisiting the Inverted Indices for Billion-Scale Approximate Nearest Neighbors
- [CVPR 2019](https://research.yandex.com/publications/196) Unsupervised Neural Quantization for Compressed-Domain Similarity Search
- [ICML 2019](https://research.yandex.com/publications/188) Learning to Route in Similarity Graphs
- [ICML 2020](https://research.yandex.com/publications/280) Graph-based Nearest Neighbor Search: From Practice to Theory
