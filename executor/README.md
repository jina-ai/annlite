# PQLiteIndexer

`PQLiteIndexer` uses the [PQLite](https://github.com/jina-ai/pqlite) class for indexing Jina `Document` objects.
The `PQLite` class partitions the data into cells at index time, and instantiates a "sub-indexer" in each cell.  Search is performed aggregating results retrieved from cells.

This indexer is recommended to be used when an application requires **search with filters** applied on `Document` tags.
The `filtering query language` is based on [MongoDB's query and projection operators](https://docs.mongodb.com/manual/reference/operator/query/). We currently support a subset of those selectors.
The tags filters can be combined with `$and` and `$or`:

- `$eq` - Equal to (number, string)
- `$ne` - Not equal to (number, string)
- `$gt` - Greater than (number)
- `$gte` - Greater than or equal to (number)
- `$lt` - Less than (number)
- `$lte` - Less than or equal to (number)

For example, we want to search for a product with a price no more than `50$`.
```python
index.search(query, filter={"price": {"$lte": 50}})
```

More example filter expresses

- A Nike shoes with white color

```JSON
{
  "brand": {"$eq": "Nike"},
  "category": {"$eq": "Shoes"},
  "color": {"$eq": "White"}
}
```

Or

```JSON
{
  "$and":
    {
      "brand": {"$eq": "Nike"},
      "category": {"$eq": "Shoes"},
      "color": {"$eq": "White"}
    }
}
```


- A Nike shoes or price less than `100$`

```JSON
{
  "$or":
    {
      "brand": {"$eq": "Nike"},
      "price": {"$lt": 100}
    }
}
```

## Basic Usage

`PQLiteIndexer` stores  `Document` objects at the  `workspace` directory, specified under the [`metas`](https://docs.jina.ai/fundamentals/executor/executor-built-in-features/#meta-attributes) attribute.

#### Example: Selecting items whose 'price' is less than 50

If documents have a tag `'price'`  that stores floating point values this indexer allows searching documents with a filter, such as  `price <= 50`.

```python
columns = [('price', 'float')]

f = Flow().add(
    uses='jinahub://PQLiteIndexer',
    uses_with={
      'dim': 256,
      'columns': columns,
      'metric': 'euclidean'
    },
    uses_metas={'workspace': '/my/tmp_folder'}
)

search_filter = {"price": {"$lte": 50}}
with f:
    f.post(on='/index', inputs=docs)
    query_res = f.post(on='/search',
                       inputs=query_docs,
                       return_results=True,
                       parameters={'filter': search_filter})
```

## Performance

One can run `benchmark.py` to get a quick performance overview.

|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|
|---|---|---|---|---|
|10000 | 1.882 | 0.002 | 0.017 | 0.133|
|100000 | 143.641 | 0.011 | 0.085 | 0.670|
|500000 | 1502.146 | 0.039 | 0.307 | 2.442|
|1000000 | 3496.651 | 0.072 | 0.582 | 4.617|

## CRUD operations

You can perform all the usual operations on the respective endpoints

- `/index` Add documents
- `/search` Search with query documents.
- `/update` Update documents
- `/delete` Delete documents
- `/clear` Clear the index
- `/status` Return the status of index
  - `total_docs`: the total number of indexed documents
  - `dim`: the dimension of the embeddings
  - `metric`: the distance metric type
  - `is_trained`: whether the index is already trained
