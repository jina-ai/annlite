# PQLiteIndexer

`PQLiteIndexer` uses the `PQLite` class for indexing Jina `Document` objects. The `PQLite` class partitions the data into cells at index time, and instantiates a "sub-indexer" in each cell.  Search is performed aggregating results retrieved from cells. 

This indexer is recommended to be used when an application requires search with filters applied on `Document` tags. For example, if documents have a tag `'price'`  that stores floating point values this indexer allows searching documents with a filter, such as  `price <= 100`.

By default, it uses the `euclidean` distance to rank results.

## Basic Usage

`PQLiteIndexer` stores  `Document` objects at the  `workspace` directory, specified under the [`metas`](https://docs.jina.ai/fundamentals/executor/executor-built-in-features/#meta-attributes) attribute. 
You can override the default configuration as below,

```python
f = Flow().add(
    uses='jinahub://PQLiteIndexer',
    uses_with={'dim': 256, 'metric': 'euclidean'},
    uses_metas={'workspace': '/my/tmp_folder'})
```

Find more information about how to override `metas` attributes at [Jina Docs](https://docs.jina.ai/fundamentals/flow/add-exec-to-flow/#override-metas-configuration)

## Filtering nearest neighbor candidates

Search can be performed with candidate filtering. Filters are a triplet (column,operator,value).
More than a filter can be applied during search. Therefore, conditions for a filter are specified as a list triplets.
Each triplet contains:

- column: Column used to filter.
- operator: Binary operation between two values. Some supported operators include `['>','<','=','<=','>=']`.
- value: value used to compare a candidate.

#### Example: Selecting items whose 'price' is less than 50 
```
columns = [('price', 'float', 'True')]

f = Flow().add(uses=PQLiteIndexer,
               uses_with={'dim': D, 'columns': columns})

conditions = [['price', '<', '50.']]
with f:
    f.post(on='/index', inputs=docs)
    query_res = f.post(on='/search',
                       inputs=docs_query,
                       return_results=True,
                       parameters={'conditions': conditions})
```

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