import threading
import time
import traceback
import warnings
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

from docarray import Document, DocumentArray
from jina import Executor, requests
from jina.logging.logger import JinaLogger

INDEX_BATCH_SIZE = 1024


class AnnLiteIndexer(Executor):
    """A simple indexer that wraps the AnnLite indexer and adds a simple interface for indexing and searching.

    :param n_dim: Dimensionality of vectors to index
    :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
    :param limit: Number of results to get for each query document in search
    :param data_path: the workspace of the AnnLiteIndexer.
    :param ef_construction: The construction time/accuracy trade-off
    :param ef_search: The query time accuracy/speed trade-off
    :param max_connection: The maximum number of outgoing connections in the
        graph (the "M" parameter)
    :param include_metadata: If True, return the document metadata in response
    :param index_access_paths: Default traversal paths on docs
            (used for indexing, delete and update), e.g. '@r', '@c', '@r,c'
    :param search_access_paths: Default traversal paths on docs
    (used for search), e.g. '@r', '@c', '@r,c'
    :param columns: A list or dict of column names to index.
    :param dim: Deprecated, use n_dim instead
    """

    def __init__(
        self,
        n_dim: int = 0,
        metric: str = 'cosine',
        limit: int = 10,
        data_path: Optional[str] = None,
        ef_construction: Optional[int] = None,
        ef_search: Optional[int] = None,
        max_connection: Optional[int] = None,
        include_metadata: bool = True,
        index_access_paths: str = '@r',
        search_access_paths: str = '@r',
        columns: Optional[Union[List[Tuple[str, str]], Dict[str, str]]] = None,
        dim: int = None,
        *args,
        **kwargs,
    ):

        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        n_dim = n_dim or dim
        if not n_dim:
            raise ValueError('Please specify the dimension of the vectors to index!')

        self.metric = metric
        self.limit = limit
        self.include_metadata = include_metadata

        self.index_access_paths = index_access_paths
        if 'index_traversal_paths' in kwargs:
            warnings.warn(
                f'`index_traversal_paths` is deprecated. Use `index_access_paths` instead.'
            )
            self.index_access_paths = kwargs['index_traversal_paths']

        self.search_access_paths = search_access_paths
        if 'search_traversal_paths' in kwargs:
            warnings.warn(
                f'`search_traversal_paths` is deprecated. Use `search_access_paths` instead.'
            )
            self.search_access_paths = kwargs['search_traversal_paths']

        self._data_buffer = DocumentArray()
        self._index_batch_size = INDEX_BATCH_SIZE
        self._max_length_queue = 2 * self._index_batch_size
        self._index_lock = threading.Lock()

        self.logger = JinaLogger(getattr(self.metas, 'name', self.__class__.__name__))

        if self.runtime_args.shards > 1 and data_path:
            raise ValueError(f'`data_path` is not supported in shards mode')

        config = {
            'n_dim': n_dim,
            'metric': metric,
            'ef_construction': ef_construction,
            'ef_search': ef_search,
            'max_connection': max_connection,
            'data_path': data_path or self.workspace or './workspace',
            'columns': columns,
        }
        self._index = DocumentArray(storage='annlite', config=config)

        # start indexing thread in background to group indexing requests
        # together and perform batch indexing at once
        self._start_index_loop()

    @requests(on='/index')
    def index(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Index new documents

        :param docs: the Documents to index
        :param parameters: dictionary with options for indexing
        Keys accepted:
            - 'access_paths': traversal paths on docs, e.g. '@r', '@c', '@r,c'
        """

        if not docs:
            return

        access_paths = parameters.get('access_paths', self.index_access_paths)
        flat_docs = docs[access_paths]
        if len(flat_docs) == 0:
            return

        while len(self._data_buffer) >= self._max_length_queue:
            time.sleep(0.001)

        with self._index_lock:
            self._data_buffer.extend(flat_docs)

    def _start_index_loop(self):
        """Start the indexing loop in background.

        This loop is responsible for batch indexing the documents in the buffer.
        """

        def _index_loop():
            try:
                while True:
                    # if the buffer is none, will break the loop
                    if self._data_buffer is None:
                        break

                    # if the buffer is empty, will wait for new documents to be added
                    if len(self._data_buffer) == 0:
                        time.sleep(0.1)  # sleep for 100ms
                        continue

                    # acquire the lock to prevent threading issues
                    with self._index_lock:
                        batch_docs = self._data_buffer.pop(
                            range(
                                self._index_batch_size
                                if len(self._data_buffer) > self._index_batch_size
                                else len(self._data_buffer)
                            )
                        )
                        self._index.extend(batch_docs)
                        self.logger.debug(f'indexing {len(batch_docs)} docs done...')
            except Exception as e:
                self.logger.error(traceback.format_exc())
                raise e

        self._index_thread = Thread(target=_index_loop, daemon=False)
        self._index_thread.start()

    @requests(on='/update')
    def update(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Update existing documents

        :param docs: the Documents to update
        :param parameters: dictionary with options for updating
        Keys accepted:
            - 'access_paths': traversal paths on docs, e.g. '@r', '@c', '@r,c'
            - 'raise_errors_on_not_found': if True, raise an error if a document is not found. Default is False.
        """

        if not docs:
            return

        access_paths = parameters.get('access_paths', self.index_access_paths)
        raise_errors_on_not_found = parameters.get('raise_errors_on_not_found', False)
        flat_docs = docs[access_paths]
        if len(flat_docs) == 0:
            return

        with self._index_lock:
            if len(self._data_buffer) > 0:
                raise RuntimeError(
                    f'Cannot update documents while the pending documents in the buffer are not indexed yet. '
                    'Please wait for the pending documents to be indexed.'
                )
            for doc in flat_docs:
                try:
                    self._index[doc.id] = doc
                except IndexError:
                    if raise_errors_on_not_found:
                        raise Exception(
                            f'The document (id={doc.id}) cannot be updated as'
                            f'it is not found in the index'
                        )
                    else:
                        self.logger.warning(
                            f'cannot update doc {doc.id} as it does not exist in storage'
                        )

    @requests(on='/delete')
    def delete(self, parameters: dict = {}, **kwargs):
        """Delete existing documents

        Delete entries from the index by id
        :param parameters: parameters to the request
        """

        delete_ids = parameters.get('ids', [])
        if len(delete_ids) == 0:
            return

        with self._index_lock:
            if len(self._data_buffer) > 0:
                raise RuntimeError(
                    f'Cannot delete documents while the pending documents in the buffer are not indexed yet. '
                    'Please wait for the pending documents to be indexed.'
                )

            del self._index[delete_ids]

    @requests(on='/search')
    def search(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Perform a vector similarity search and retrieve Document matches

        Search can be performed with candidate filtering. Filters are a triplet (column,operator,value).
        More than a filter can be applied during search. Therefore, conditions for a filter are specified as a list triplets.
        Each triplet contains:

        - column: Column used to filter.
        - operator: Binary operation between two values. Some supported operators include `['>','<','=','<=','>=']`.
        - value: value used to compare a candidate.

        :param docs: the Documents to search with
        :param parameters: dictionary for parameters for the search operation
        Keys accepted:
            - 'access_paths' (str): traversal paths on docs, e.g. '@r', '@c', '@r,c'
            - 'filter' (dict): the filtering conditions on document tags
            - 'limit' (int): nr of matches to get per Document
        """

        if not docs:
            return

        limit = int(parameters.get('limit', self.limit))
        search_filter = parameters.get('filter', None)
        access_paths = parameters.get('access_paths', self.search_access_paths)
        flat_docs = docs[access_paths]

        with self._index_lock:
            if len(self._data_buffer) > 0:
                raise RuntimeError(
                    f'Cannot search documents while the pending documents in the buffer are not indexed yet. '
                    'Please wait for the pending documents to be indexed.'
                )

            flat_docs.match(self._index, filter=search_filter, limit=limit)

    @requests(on='/filter')
    def filter(self, parameters: Dict, **kwargs):
        """
        Query documents from the indexer by the filter `query` object in parameters. The `query` object must follow the
        specifications in the `find` method of `DocumentArray` using annlite: https://docarray.jina.ai/fundamentals/documentarray/find/#filter-with-query-operators
        :param parameters: Dictionary to define the `filter` that you want to use.
        """

        return self._index.find(parameters.get('filter', None))

    @requests(on='/fill_embedding')
    def fill_embedding(self, docs: DocumentArray, **kwargs):
        """
        retrieve embedding of Documents by id
        :param docs: DocumentArray to search with
        """

        for doc in docs:
            doc.embedding = self._index[doc.id].embedding

    @requests(on='/status')
    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(
            tags={
                'appending_size': len(self._data_buffer),
                'total_docs': len(self._index),
                'index_size': len(self._index),
            }
        )
        return DocumentArray([status])

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index of all entries."""
        with self._index_lock:
            self._data_buffer = DocumentArray()
        self._index.clear()

    def close(self, **kwargs):
        """Close the index."""
        super().close()

        while len(self._data_buffer) > 0:
            time.sleep(0.1)

        # wait for the index thread to finish
        with self._index_lock:
            self._data_buffer = None
            self._index_thread.join()

            del self._index
