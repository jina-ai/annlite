from typing import Dict, List, Optional, Tuple

from docarray import Document, DocumentArray
from jina import Executor, requests
from jina.logging.logger import JinaLogger

from .index import AnnLite


class AnnLiteIndexer(Executor):
    """
    A simple Indexer based on PQLite that stores all the Document data together in a local LMDB store.

    To be used as a hybrid indexer, supporting pre-filtering searching.
    """

    def __init__(
        self,
        dim: int = 0,
        metric: str = 'cosine',
        limit: int = 10,
        ef_construction: int = 200,
        ef_query: int = 50,
        max_connection: int = 16,
        include_metadata: bool = True,
        index_traversal_paths: str = '@r',
        search_traversal_paths: str = '@r',
        columns: Optional[List[Tuple[str, str]]] = None,
        serialize_config: Optional[Dict] = None,
        *args,
        **kwargs,
    ):
        """
        :param dim: Dimensionality of vectors to index
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        :param include_metadata: If True, return the document metadata in response
        :param limit: Number of results to get for each query document in search
        :param ef_construction: The construction time/accuracy trade-off
        :param ef_query: The query time accuracy/speed trade-off
        :param max_connection: The maximum number of outgoing connections in the
            graph (the "M" parameter)
        :param index_traversal_paths: Default traversal paths on docs
                (used for indexing, delete and update), e.g. '@r', '@c', '@r,c'
        :param search_traversal_paths: Default traversal paths on docs
        (used for search), e.g. '@r', '@c', '@r,c'
        :param columns: List of tuples of the form (column_name, str_type). Here str_type must be a string that can be
                parsed as a valid Python type.
        :param serialize_config: The configurations used for serializing documents, e.g., {'protocol': 'pickle'}
        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        assert dim > 0, 'Please specify the dimension of the vectors to index!'

        self.metric = metric
        self.limit = limit
        self.include_metadata = include_metadata
        self.index_traversal_paths = index_traversal_paths
        self.search_traversal_paths = search_traversal_paths
        self._valid_input_columns = ['str', 'float', 'int']

        if columns:
            cols = []
            for n, t in columns:
                assert (
                    t in self._valid_input_columns
                ), f'column of type={t} is not supported. Supported types are {self._valid_input_columns}'
                cols.append((n, eval(t)))
            columns = cols

        self._index = AnnLite(
            dim=dim,
            metric=metric,
            columns=columns,
            ef_construction=ef_construction,
            ef_query=ef_query,
            max_connection=max_connection,
            data_path=self.workspace or './workspace',
            serialize_config=serialize_config or {},
            **kwargs,
        )

    def index(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Index new documents

        :param docs: the Documents to index
        :param parameters: dictionary with options for indexing
        Keys accepted:
            - 'traversal_paths' (str): traversal path for the docs
        """
        if not docs:
            return

        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs[traversal_paths]
        if len(flat_docs) == 0:
            return

        self._index.index(flat_docs)

    def update(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Update existing documents

        :param docs: the Documents to update
        :param parameters: dictionary with options for updating
        Keys accepted:

            - 'traversal_paths' (str): traversal path for the docs
        """

        if not docs:
            return

        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs[traversal_paths]
        if len(flat_docs) == 0:
            return

        self._index.update(flat_docs)

    def delete(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Delete existing documents

        :param docs: the Documents to delete
        :param parameters: dictionary with options for deletion

        Keys accepted:
            - 'traversal_paths' (str): traversal path for the docs
        """
        if not docs:
            return

        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs[traversal_paths]
        if len(flat_docs) == 0:
            return

        self._index.delete(flat_docs)

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

            - 'filter' (dict): the filtering conditions on document tags
            - 'traversal_paths' (str): traversal paths for the docs
            - 'limit' (int): nr of matches to get per Document
        """
        if not docs:
            return

        limit = int(parameters.get('limit', self.limit))
        search_filter = parameters.get('filter', {})
        include_metadata = bool(
            parameters.get('include_metadata', self.include_metadata)
        )

        traversal_paths = parameters.get('traversal_paths', self.search_traversal_paths)
        flat_docs = docs[traversal_paths]
        if len(flat_docs) == 0:
            return

        self._index.search(
            flat_docs,
            filter=search_filter,
            limit=limit,
            include_metadata=include_metadata,
        )

    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(tags=self._index.stat)
        return DocumentArray([status])

    def clear(self, **kwargs):
        """Clear the index of all entries."""
        self._index.clear()

    def close(self, **kwargs):
        """Close the index."""
        self._index.close()
