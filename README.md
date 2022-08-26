# AnnLite

`AnnLite` is a lightweight library for **fast** and **memory efficient** *approximate nearest neighbor search* (ANNS).
It allows to search for nearest neighbors in a dataset of millions of points with a Pythonic API.

- 🐥 **Easy-to-use**: a simple API is designed to be used with Python. It is easy to use and intuitive to set up to production.
- 🐎 **Fast**: the library uses a highly optimized approximate nearest neighbor search algorithm. It allows you to
    search for nearest neighbors in a dataset of millions of points in a fraction of a second.
- 💡 **Hybrid-search**: employ an efficient pre-filtering algorithm to support hybrid search. It supports filtering
    by distance and by attributes.
- 🍱 **Integration**: Smooth integration with neural search ecosystem including Jina and DocArray, so that users can easily
    expose search API with gRPC and/or HTTP.

Read more on why should you use `AnnLite`: [here](), and compare to alternatives: [here]().


## Installation

```bash
pip install annlite
```

## Usage

In this example, we will search for nearest neighbors in the dataset [Totally looks like]()

```python
import annlite
from docarray import DocArray

# Load the dataset

data = DocArray.load('data/dataset.json')

ann = annlite.AnnLite(dataset_path='data/dataset.csv')

# Index the dataset
ann.index(data)

# Search for nearest neighbors
ann.search(query_point=[1, 2, 3], k=3)
```

And to see the results, run the following command:

```bash
python -m annlite.cli --dataset_path data/dataset.csv --query_point [1, 2, 3] --k 3
```

We can also use the `--help` option to see the available options.

To support search with filters, we can use the `conditions` argument to filter the indexed dataset during search.

```python
import annlite

ann = annlite.AnnLite(dataset_path='data/dataset.csv', filter_fields=['city'])
ann.search(query_point=[1, 2, 3], k=3, conditions={'distance': {'$lt': 1}})
```

## Benchmark

## FAQ

1. How do I use `AnnLite` with Jina?

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

## Contributing

We are also looking for contributors who want to help us improve the library.
Please open an issue or pull request on [Github]().

## License

`AnnLite` is licensed under the [Apache License 2.0]().
