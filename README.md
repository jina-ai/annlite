<p align="center">
<br>
<br>
<br>
<img src="https://github.com/jina-ai/annlite/blob/main/.github/assets/logo.svg?raw=true" alt="AnnLite logo: A fast and efficient ann libray" width="200px">
<br>
<br>
<br>
<b>A fast embedded library for approximate nearest neighbor search</b>
</p>

<p align=center>
<a href="https://pypi.org/project/annlite/"><img alt="PyPI" src="https://img.shields.io/pypi/v/annlite?label=Release&style=flat-square"></a>
<a href="https://slack.jina.ai"><img src="https://img.shields.io/badge/Slack-3.1k-blueviolet?logo=slack&amp;logoColor=white&style=flat-square"></a>
<a href="https://codecov.io/gh/jina-ai/annlite"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/github/jina-ai/annlite/main?logo=Codecov&logoColor=white&style=flat-square"></a>
</p>

<!-- start elevator-pitch -->


## What is AnnLite?

`AnnLite` is a lightweight library for **fast** and **memory efficient** *approximate nearest neighbor search* (ANNS).
It allows to search for nearest neighbors in a dataset of millions of points with a Pythonic API.


**Highlighted features:**

- 🐥 **Easy-to-use**: a simple API is designed to be used with Python. It is easy to use and intuitive to set up to production.

- 🐎 **Fast**: the library uses a highly optimized approximate nearest neighbor search algorithm (*HNSW*) to search for nearest neighbors.

- 🔎 **Filterable**: the library allows you to search for nearest neighbors within a subset of the dataset.

- 🍱 **Integration**: Smooth integration with neural search ecosystem including [Jina](https://github.com/jina-ai/jina) and [DocArray](https://github.com/jina-ai/docarray),
    so that users can easily expose search API with **gRPC** and/or **HTTP**.

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

## Quick start

Before you start, you need to know some experience about [DocArray](https://github.com/jina-ai/docarray).
`AnnLite` is designed to be used with [DocArray](https://github.com/jina-ai/docarray), so you need to know how to use `DocArray` first.

For example, you can create a `DocArray` with `1000` random vectors with `128` dimensions:

```python
from docarray import DocumentArray
import numpy as np

docs = DocumentArray.empty(1000)
docs.embeddings = np.random.random([1000, 128]).astype(np.float32)
```


### Index

Then you can create an `AnnIndexer` to index the created `docs` and search for nearest neighbors:

```python
from annlite import AnnLite

ann = AnnLite(128, metric='cosine', data_path="/tmp/annlite_index")
ann.index(docs)
```

### Search

Then you can search for nearest neighbors for some query docs with `ann.search()`:

```python
query = DocumentArray.empty(5)
query.embeddings = np.random.random([5, 128]).astype(np.float32)

result = ann.search(query)
print(result['@m', ('id', 'scores__cosine')])
```

Then, you can inspect the retrieved docs for each query doc inside `query` matches:

```bash
Query ddbae2073416527bad66ff186543eff8
0: 47dcf7f3fdbe3f0b8d73b87d2a1b266f {'value': 0.17575037}
1: 7f2cbb8a6c2a3ec7be024b750964f317 {'value': 0.17735684}
2: 2e7eed87f45a87d3c65c306256566abb {'value': 0.17917466}
Query dda90782f6514ebe4be4705054f74452
0: 6616eecba99bd10d9581d0d5092d59ce {'value': 0.14570713}
1: d4e3147fc430de1a57c9883615c252c6 {'value': 0.15338594}
2: 5c7b8b969d4381f405b8f07bc68f8148 {'value': 0.15743542}
...
```

Or shorten the loop as one-liner using the element & attribute selector:

```python
print(query['@m', ('id', 'scores__cosine')])
```

### Update

After you have indexed the docs, you can update the docs by calling `ann.update()`:

```python
```


### Delete

And finally, you can delete the docs by calling `ann.delete()`:

```python
```

## Search with filters

To support search with filters, the annlite must be created with `fields` parameter, which is a series of fields you want to filter by.
At the query time, the annlite will filter the dataset by providing `conditions` for certain fields.

```python
import annlite

ann = annlite.AnnLite(dataset_path='data/dataset.csv', fields=['city'])
ann.search(query_point=[1, 2, 3], k=3, conditions={'distance': {'$lt': 1}})
```

The `conditions` parameter is a dictionary of conditions. The key is the field name, and the value is a dictionary of conditions.
The query language is the same as MongoDB. The following is an example of a query:

```python
{
    'distance': {'$lt': 1},
    'city': {'$eq': 'Beijing'}
}
```
We also support boolean operators:

```python
{
    'city': {'$eq': 'Beijing'},
    '$or': [
        {'city': {'$eq': 'Beijing'}},
        {'city': {'$eq': 'Shanghai'}}
    ]
}
```
For more information, please refer to [MongoDB Query Language](https://docs.mongodb.com/manual/reference/operator/query/).


The query will be performed on the field if the condition is satisfied.

## Next steps

If you already have experience with Jina and DocArray, you can start using `AnnLite` right away.

Otherwise, you can check out this advanced tutorial to learn how to use `AnnLite`: [here]() in practice.


## 🙋 FAQ

**1. Why should I use `AnnLite`?**

    `AnnLite` is easy to use and intuitive to set up in production. It is also very fast and memory efficient, making it a great choice for approximate nearest neighbor search.

2. How do I use `AnnLite` with Jina?

```python
```

2. How do I use `AnnLite` with DocArray?

```python
```

3. How do I use `AnnLite` with other search engines?

```python
```

4. How to reduce the memory footprint of `AnnLite`?

```python
```

5. What's the difference between `AnnLite` and other alternatives?

```python
```

6. How to expose search API with gRPC and/or HTTP?

```python
```

7. Does `AnnLite` support search with filters?

```text
Yes.
```


## Documentation

You can find the documentation on [Github]() and [ReadTheDocs]()

## 🤝 Contribute and spread the word

We are also looking for contributors who want to help us improve: code, documentation, issues, feedback! Here is how you can get started:

- Have a look through GitHub issues labeled "Good first issue".
- Read our Contributor Covenant Code of Conduct
- Open an issue or submit your pull request!


## License

`AnnLite` is licensed under the [Apache License 2.0]().
