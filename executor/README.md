# PQLiteIndexer

`PQLiteIndexer` uses the `PQLite` class for indexing Jina `Document` objects. The `PQLite` class partitions the data into cells at index time, and instanciates a "sub-indexer" in each cell.  Search is performed agregating results retrieved from cells. 

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

## CRUD operations

You can perform all the usual operations on the respective endpoints

- `/index` Add documents
- `/search` Search with query documents.
- `/update` Update documents 
- `/delete` Delete documents
- `/clear` Clear the index
- `/status` Return the status of index
