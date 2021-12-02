from typing import Optional, Iterable
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
import pqlite


class PQLiteIndexer(Executor):
    """
    A simple Indexer based on PQLite that stores all the Document data together in a local LMDB store.

    To be used as a hybrid indexer, supporting pre-filtering searching.
    """
    def __init__(
        self,
        dim: int = 0,
        metric: str = 'euclidean',
        limit: int = 10,
        index_traversal_paths: str = 'r',
        search_traversal_paths: str = 'r',
        *args,
        **kwargs
    ):
        """
        :param dim: Dimensionality of vectors to index
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        :param limit: Number of results to get for each query document in search
        :param index_traversal_paths: Default traversal paths on docs
        (used for indexing, delete and update), e.g. 'r', 'c', 'r,c'
        :param search_traversal_paths: Default traversal paths on docs
        (used for search), e.g. 'r', 'c', 'r,c'

        """
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        self.metric = metric
        self.limit = limit
        self.index_traversal_paths = index_traversal_paths
        self.search_traversal_paths = search_traversal_paths

        self._index = pqlite.PQLite(dim=dim, metric=metric, data_path=self.workspace, **kwargs)

    @requests(on='/index')
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
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        self._index.index(flat_docs)

    @requests(on='/update')
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
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        self._index.update(flat_docs)

    @requests(on='/delete')
    def delete(self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs):
        """Delete existing documents

        :param docs: the Documents to delete
        :param parameters: dictionary with options for deletion

        Keys accepted:
            - 'traversal_paths' (str): traversal path for the docs
        """
        if not docs:
            return

        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        self._index.delete(flat_docs)

    @requests(on='/search')
    def search(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        """Perform a vector similarity search and retrieve the full Document match

        :param docs: the Documents to search with
        :param parameters: dictionary for parameters for the search operation
        Keys accepted:

            - 'conditions' (dict): the filtering conditions on document tags
            - 'traversal_paths' (str): traversal paths for the docs
            - 'limit' (int): nr of matches to get per Document
        """
        if not docs:
            return
        limit = int(parameters.get('limit', self.limit))
        traversal_paths = parameters.get('traversal_paths', self.search_traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        self._index.search(flat_docs, conditions=parameters.get('conditions', None), limit=limit)

    @requests(on='/status')
    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(tags=self._index.stat)
        return DocumentArray([status])

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index of all entries."""
        self._index.clear()
