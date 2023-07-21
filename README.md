<p align="center">
<br>
<br>
<br>
<img src="https://github.com/jina-ai/annlite/blob/main/.github/assets/logo.svg?raw=true" alt="AnnLite logo: A fast and efficient ann libray" width="200px">
<br>
<br>
<b>A fast embedded library for approximate nearest neighbor search</b>
</p>

<p align=center>
<a href="hhttps://github.com/jina-ai/annlite"><img alt="GitHub" src="https://img.shields.io/github/license/jina-ai/annlite?style=flat-square"></a>
<a href="https://pypi.org/project/annlite/"><img alt="PyPI" src="https://img.shields.io/pypi/v/annlite?label=Release&style=flat-square"></a>
<a href="https://codecov.io/gh/jina-ai/annlite"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/github/jina-ai/annlite/main?logo=Codecov&logoColor=white&style=flat-square"></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-3.1k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>
</p>

<!-- start elevator-pitch -->


## What is AnnLite?

`AnnLite` is a *lightweight* and *embeddable* library for **fast** and **filterable** *approximate nearest neighbor search* (ANNS).
It allows to search for nearest neighbors in a dataset of millions of points with a Pythonic API.


**Highlighted features:**

- 🐥 **Easy-to-use**: a simple API is designed to be used with Python. It is easy to use and intuitive to set up to production.

- 🐎 **Fast**: the library uses a highly optimized approximate nearest neighbor search algorithm (*HNSW*) to search for nearest neighbors.

- 🔎 **Filterable**: the library allows you to search for nearest neighbors within a subset of the dataset.

The library is easy to install and use. It is designed to be used with Python.

<!---
Read more on why should you use `AnnLite`: [here](), and compare to alternatives: [here]().
-->

## Installation

To use AnnLite, you need to first install it. The easiest way to install AnnLite is using `pip`:

```bash
pip install -U annlite
```

or install from source:

```bash
python setup.py install
```

### Index

Then you can create an `AnnIndexer` to index the created `docs` and search for nearest neighbors:

```python
from annlite import AnnLite

docs = [{'id': '0', 'embedding': [], 'price': 8}]

ann = AnnLite(128, metric='cosine', data_path="/tmp/annlite_data")
ann.index(docs)
```

Note that this will create a directory `/tmp/annlite_data` to persist the documents indexed.
If this directory already exists, the index will be loaded from the directory.
And if you want to create a new index, you can delete the directory first.

### Search

Then you can search for nearest neighbors for some query docs with `ann.search()`:

```python
query = []

result = ann.search(query)
```

Then, you can inspect the retrieved docs for each query doc inside `query` matches:
```python
```

### Query

You can get specific document by its id:

```python
doc = ann.get_doc_by_id('<doc_id>')
```

And you can also get the documents with `limit` and `offset`, which is useful for pagination:

```python
docs = ann.get_docs(limit=10, offset=0)
```

Furthermore, you can also get the documents ordered by a specific column from the index:

```python
docs = ann.get_docs(limit=10, offset=0, order_by='x', ascending=True)
```

**Note**: the `order_by` column must be one of the `columns` in the index.

### Update

After you have indexed the `docs`, you can update the docs in the index by calling `ann.update()`:

```python
updated_docs = [{'id': '0', 'embedding': np.random.random([128]).astype(np.float32), 'price': 6}]

ann.update(updated_docs)
```


### Delete

And finally, you can delete the docs from the index by calling `ann.delete()`:

```python
to_delete = [{'id': '0'}]
ann.delete(to_delete)
```

## Search with filters

To support search with filters, the annlite must be created with `colums` parameter, which is a series of fields you want to filter by.
At the query time, the annlite will filter the dataset by providing `conditions` for certain fields.

```python
import annlite

# the column schema: (name:str, dtype:type, create_index: bool)
ann = annlite.AnnLite(128, columns=[('price', float)], data_path="/tmp/annlite_data", metric='euclidean')
```

Then you can insert the docs, in which each doc has a field `price` with a float value contained in the `tags`:


```python
import random
import numpy as np
docs = [dict(id=f'{i}', price=random.random(), embedding=np.random.random([128]).astype(np.float32)) for i in range(1000)]
ann.index(docs)
```

Then you can search for nearest neighbors with filtering conditions as:

```python
queries = [dict(embedding=np.random.random([128]).astype(np.float32)) for i in range(5)]

query_matches = ann.search(queries, filter={"price": {"$lte": 50}}, limit=10)
print(f'the result with filtering:')
for query, matches in enumerate(queries, query_matches):
    print(f'query [{i}]:')
    for m in matches:
        print(f'\t{m["id"]} {m["scores"]["euclidean"]} (price={m["price"]})')
```

The `conditions` parameter is a dictionary of conditions. The key is the field name, and the value is a dictionary of conditions.
The query language is the same as  [MongoDB Query Language](https://docs.mongodb.com/manual/reference/operator/query/).
We currently support a subset of those selectors.

- `$eq` - Equal to (number, string)
- `$ne` - Not equal to (number, string)
- `$gt` - Greater than (number)
- `$gte` - Greater than or equal to (number)
- `$lt` - Less than (number)
- `$lte` - Less than or equal to (number)
- `$in` - Included in an array
- `$nin` - Not included in an array


The query will be performed on the field if the condition is satisfied. The following is an example of a query:

1. A Nike shoes with white color

    ```python
    {
      "brand": {"$eq": "Nike"},
      "category": {"$eq": "Shoes"},
      "color": {"$eq": "White"}
    }
    ```

    We also support boolean operators `$or` and `$and`:

    ```python
    {
      "$and":
        {
          "brand": {"$eq": "Nike"},
          "category": {"$eq": "Shoes"},
          "color": {"$eq": "White"}
        }
    }
    ```

2. A Nike shoes or price less than `100$`:

    ```python
    {
        "$or":
        {
        "brand": {"$eq": "Nike"},
        "price": {"$lte": 100}
        }
    }
    ```

## Dump and Load

By default, the hnsw index is in memory. You can dump the index to `data_path` by calling `.dump()`:

```python

from annlite import AnnLite

ann = AnnLite(128, metric='cosine', data_path="/path/to/data_path")
ann.index(docs)
ann.dump()
```

And you can restore the hnsw index from `data_path` if it exists:

```python
new_ann = AnnLite(128, metric='cosine', data_path="/path/to/data_path")
```

If you didn't dump the hnsw index, the index will be rebuilt from scratch. This will take a while.

## Supported distance metrics

The annlite supports the following distance metrics:

#### Supported distances:

| Distance                                                             |       parameter |                                                Equation |
|----------------------------------------------------------------------|----------------:|--------------------------------------------------------:|
| [Euclidean](https://en.wikipedia.org/wiki/Euclidean_distance)        |     `euclidean` |                                d = sqrt(sum((Ai-Bi)^2)) |
| [Inner product](https://en.wikipedia.org/wiki/Inner_product_space)   | `inner_product` |                                   d = 1.0 - sum(Ai\*Bi) |
| [Cosine similarity](https://en.wikipedia.org/wiki/Cosine_similarity) |        `cosine` | d = 1.0 - sum(Ai\*Bi) / sqrt(sum(Ai\*Ai) * sum(Bi\*Bi)) |

Note that inner product is not an actual metric. An element can be closer to some other element than to itself.
That allows some speedup if you remove all elements that are not the closest to themselves from the index, e.g.,
`inner_product([1.0, 1.0], [1.0. 1.0]) < inner_product([1.0, 1.0], [2.0, 2.0])`


## HNSW algorithm parameters

The HNSW algorithm has several parameters that can be tuned to improve the search performance.

### Search parameters

- `ef_search` - The size of the dynamic list for the nearest neighbors during search (default: `50`).
The larger the value, the more accurate the search results, but the slower the search speed.
The `ef_search` must be larger than `limit` parameter in `search(..., limit)`.

- `limit` - The maximum number of results to return (default: `10`).

## Construction parameters

- `max_connection` - The number of bi-directional links created for every new element during construction (default: `16`).
Reasonable range is from `2` to `100`. Higher values works better for dataset with higher dimensionality and/or high recall.
This parameter also affects the memory consumption during construction, which is roughly `max_connection * 8-10` bytes per stored element.

    As an example for `n_dim=4` random vectors optimal `max_connection` for search is somewhere around `6`,
while for high dimensional datasets, higher `max_connection` are required (e.g. `M=48-64`) for optimal performance at high recall.
The range `max_connection=12-48` is ok for the most of the use cases.
When `max_connection` is changed one has to update the other parameters.
Nonetheless, `ef_search` and `ef_construction` parameters can be roughly estimated by assuming that `max_connection * ef_{construction}` is a constant.


- `ef_construction`: The size of the dynamic list for the nearest neighbors during construction (default: `200`).
Higher values give better accuracy, but increase construction time and memory consumption.
At some point, increasing `ef_construction` does not give any more accuracy.
To set `ef_construction` to a reasonable value, one can measure the recall: if the recall is lower than 0.9, then increase `ef_construction` and re-run the search.

To set the parameters, you can define them when creating the annlite:

```python
from annlite import AnnLite

ann = AnnLite(128, columns=[('price', float)], data_path="/tmp/annlite_data", ef_construction=200, max_connection=16)
```

## Benchmark

One can run `executor/benchmark.py` to get a quick performance overview.

|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|
|---|---|---|---|---|
|10000 | 2.970 | 0.002 | 0.013 | 0.100|
|100000 | 76.474 | 0.011 | 0.078 | 0.649|
|500000 | 467.936 | 0.046 | 0.356 | 2.823|
|1000000 | 1025.506 | 0.091 | 0.695 | 5.778|

Results with filtering can be generated from `examples/benchmark_with_filtering.py`. This script should produce a table similar to:

| Stored data |% same filter| Indexing time | Query size=1 | Query size=8 | Query size=64|
|-----|-----|-----|-----|-----|-----|
| 10000 | 5  | 2.869 | 0.004 | 0.030 | 0.270 |
| 10000 | 15 | 2.869 | 0.004 | 0.035 | 0.294 |
| 10000 | 20 | 3.506 | 0.005 | 0.038 | 0.287 |
| 10000 | 30 | 3.506 | 0.005 | 0.044 | 0.356 |
| 10000 | 50 | 3.506 | 0.008 | 0.064 | 0.484 |
| 10000 | 80 | 2.869 | 0.013 | 0.098 | 0.910 |
| 100000 | 5 | 75.960 | 0.018 | 0.134 | 1.092 |
| 100000 | 15 | 75.960 | 0.026 | 0.211 | 1.736 |
| 100000 | 20 | 78.475 | 0.034 | 0.265 | 2.097 |
| 100000 | 30 | 78.475 | 0.044 | 0.357 | 2.887 |
| 100000 | 50 | 78.475 | 0.068 | 0.565 | 4.383 |
| 100000 | 80 | 75.960 | 0.111 | 0.878 | 6.815 |
| 500000 | 5 | 497.744 | 0.069 | 0.561 | 4.439 |
| 500000 | 15 | 497.744 | 0.134 | 1.064 | 8.469 |
| 500000 | 20 | 440.108 | 0.152 | 1.199 | 9.472 |
| 500000 | 30 | 440.108 | 0.212 | 1.650 | 13.267 |
| 500000 | 50 | 440.108 | 0.328 | 2.637 | 21.961 |
| 500000 | 80 | 497.744 | 0.580 | 4.602 | 36.986 |
| 1000000 | 5 | 1052.388 | 0.131 | 1.031 | 8.212 |
| 1000000 | 15 | 1052.388 | 0.263 | 2.191 | 16.643 |
| 1000000 | 20 | 980.598 | 0.351 | 2.659 | 21.193 |
| 1000000 | 30 | 980.598 | 0.461 | 3.713 | 29.794 |
| 1000000 | 50 | 980.598 | 0.732 | 5.975 | 47.356 |
| 1000000 | 80 | 1052.388 | 1.151 | 9.255 | 73.552 |


Note that:
- query times presented are represented in seconds.
- `% same filter`  indicates the amount of data that verifies a filter in the database.
    - For example, if `% same filter = 10` and `Stored data = 1_000_000` then it means `100_000` example verify the filter.


## 🤝 Contribute and spread the word

We are also looking for contributors who want to help us improve: code, documentation, issues, feedback! Here is how you can get started:

- Have a look through GitHub issues labeled "Good first issue".
- Read our Contributor Covenant Code of Conduct
- Open an issue or submit your pull request!


## License

`AnnLite` is licensed under the [Apache License 2.0]().
