# PQLiteIndexer

`PQLiteIndexer` uses the [PQLite](https://github.com/jina-ai/pqlite) class for indexing Jina `Document` objects.
The `PQLite` class partitions the data into cells at index time, and instantiates a "sub-indexer" in each cell.  Search is performed aggregating results retrieved from cells.

This indexer is recommended to be used when an application requires **search with filters** applied on `Document` tags.
Each filter is a **triplet** (_column_, _operator_, _value_). More than one filter can be applied during search. Therefore, conditions for a filter are specified as a list of triplets.
Each triplet contains:

- column: Column used to filter.
- operator: Binary operation between two values. Supported operators are `['>','<','<=','>=', '=', '!=']`.
- value: value used to compare a candidate.

## Basic Usage

`PQLiteIndexer` stores  `Document` objects at the  `workspace` directory, specified under the [`metas`](https://docs.jina.ai/fundamentals/executor/executor-built-in-features/#meta-attributes) attribute.

#### Example: Selecting items whose 'price' is less than 50

If documents have a tag `'price'`  that stores floating point values this indexer allows searching documents with a filter, such as  `price <= 50`.

```python
columns = [('price', 'float', 'True')]

f = Flow().add(
    uses='jinahub://PQLiteIndexer',
    uses_with={
      'dim': 256,
      'columns': columns,
      'metric': 'euclidean'
    },
    uses_metas={'workspace': '/my/tmp_folder'}
)

conditions = [['price', '<', '50.']]
with f:
    f.post(on='/index', inputs=docs)
    query_res = f.post(on='/search',
                       inputs=docs_query,
                       return_results=True,
                       parameters={'conditions': conditions})
```

## Performance

One can run `benchmark.py` to get a quick performance overview.

|Stored data| Indexing time | Query size=1 | Query size=8 | Query size=64|
|---|---|---|---|---|
|10000 | 19.766 | 0.002 | 0.016 | 0.133|
|100000 | 325.413 | 0.010 | 0.084 | 0.671|
|500000 | 2413.311 | 0.038 | 0.303 | 2.427|
|1000000 | 5310.749 | 0.075 | 0.579 | 4.634|

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
