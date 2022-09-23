import threading
import time
import traceback
import warnings
from threading import Thread
from typing import Dict, List, Optional, Tuple, Union

from docarray import Document, DocumentArray
from jina import Executor, requests
from jina.logging.logger import JinaLogger

from .index import AnnLite


class AnnLiteIndexer(Executor):
    """
    A simple indexer that wraps the AnnLite indexer and adds a simple interface for indexing and searching.
    """

    def __init__(
        self,
        n_dim: int = 0,
        metric: str = 'cosine',
        limit: int = 10,
        ef_construction: int = 200,
        ef_query: int = 50,
        max_connection: int = 16,
        include_metadata: bool = True,
        index_access_paths: str = '@r',
        search_access_paths: str = '@r',
        columns: Optional[Union[Dict, List]] = None,
        filterable_attrs: Optional[Dict] = None,
        serialize_config: Optional[Dict] = None,
        dim: int = None,
        *args,
        **kwargs,
    ):
        """
        :param n_dim: Dimensionality of vectors to index
        :param dim: Deprecated, use n_dim instead
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        :param include_metadata: If True, return the document metadata in response
        :param limit: Number of results to get for each query document in search
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_query: The query time accuracy/speed trade-off
        :param max_connection: The maximum number of outgoing connections in the
            graph (the "M" parameter)
        :param index_access_paths: Default traversal paths on docs
                (used for indexing, delete and update), e.g. '@r', '@c', '@r,c'
        :param search_access_paths: Default traversal paths on docs
        (used for search), e.g. '@r', '@c', '@r,c'
        :param columns: A list or dict of column names to index.
        :param filterable_attrs: Dict of attributes that can be used for filtering. The key is the attribute name and the
                value is a string that can be parsed as a valid Python type ['float', 'str', 'int']. It only works if
                `columns` is None.
        :param serialize_config: The configurations used for serializing documents, e.g., {'protocol': 'pickle'}
        """
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
        self._index_batch_size = 1024
        self._max_length_queue = 2 * self._index_batch_size
        self._index_lock = threading.Lock()

        self.logger = JinaLogger(getattr(self.metas, 'name', self.__class__.__name__))

        self._index = AnnLite(
            n_dim=n_dim,
            metric=metric,
            columns=columns,
            filterable_attrs=filterable_attrs,
            ef_construction=ef_construction,
            ef_query=ef_query,
            max_connection=max_connection,
            data_path=self.workspace or './workspace',
            serialize_config=serialize_config or {},
            **kwargs,
        )

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
                        self._index.index(batch_docs)
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
            self._index.update(
                flat_docs,
                raise_errors_on_not_found=raise_errors_on_not_found,
                insert_if_not_found=False,
            )

    @requests(on='/delete')
    def delete(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Delete existing documents

        :param docs: the Documents to delete
        :param parameters: dictionary with options for deletion

        Keys accepted:
            - 'access_paths': traversal paths on docs, e.g. '@r', '@c', '@r,c'
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
                    f'Cannot delete documents while the pending documents in the buffer are not indexed yet. '
                    'Please wait for the pending documents to be indexed.'
                )

            self._index.delete(
                flat_docs, raise_errors_on_not_found=raise_errors_on_not_found
            )

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
        search_filter = parameters.get('filter', {})
        include_metadata = bool(
            parameters.get('include_metadata', self.include_metadata)
        )

        access_paths = parameters.get('access_paths', self.search_access_paths)
        flat_docs = docs[access_paths]
        if len(flat_docs) == 0:
            return
        with self._index_lock:
            self._index.search(
                flat_docs,
                filter=search_filter,
                limit=limit,
                include_metadata=include_metadata,
            )

    @requests(on='/status')
    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(
            tags={'appending_size': len(self._data_buffer), **self._index.stat}
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
        while len(self._data_buffer) > 0:
            time.sleep(0.1)

        with self._index_lock:
            self._data_buffer = None
            self._index_thread.join()
            self._index.close()
