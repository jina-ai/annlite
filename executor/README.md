# PQLiteIndexer

`PQLiteIndexer` uses `PQLite` for indexing `Document`. It is recommended to be used when you want hybrid search supported. 
By default, it calculates the `cosine` distance and returns all the indexed `Document`.

## Basic Usage

`PQLiteIndexer` stores the `Document` at the directory, which is specified by `workspace` field under the [`metas`](https://docs.jina.ai/fundamentals/executor/executor-built-in-features/#meta-attributes) attribute. 
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
