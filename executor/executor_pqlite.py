from typing import Optional, Iterable
from jina import Document, DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger
import pqlite


class PQLiteIndexer(Executor):
    def __init__(
        self,
        metric: str = 'euclidean',
        limit: int = 10,
        index_traversal_paths: str = 'r',
        search_traversal_paths: str = 'r',
        is_distance: bool = True,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        self.metric = metric
        self.limit = limit
        self.is_distance = is_distance
        self.index_traversal_paths = index_traversal_paths
        self.search_traversal_paths = search_traversal_paths

        self._index = pqlite.PQLite(metric=metric, **kwargs)

    @requests(on='/index')
    def index(self, docs: DocumentArray, parameters: Optional[dict] = {}, **kwargs):
        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        self._index.index(flat_docs)

    @requests(on='/update')
    def update(self, docs: DocumentArray, parameters: Optional[dict] = {}, **kwargs):
        traversal_paths = parameters.get('traversal_paths', self.index_traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        self._index.update(flat_docs)

    @requests(on='/search')
    def search(self, docs: DocumentArray, parameters: Optional[dict] = {}, **kwargs):
        limit = int(parameters.get('limit', self.limit))
        traversal_paths = parameters.get('traversal_paths', self.search_traversal_paths)
        flat_docs = docs.traverse_flat(traversal_paths)
        if len(flat_docs) == 0:
            return

        self._index.search(flat_docs, limit=limit)

    @requests(on='/status')
    def status(self, **kwargs) -> DocumentArray:
        """Return the document containing status information about the indexer.

        The status will contain information on the total number of indexed and deleted
        documents, and on the number of (searchable) documents currently in the index.
        """

        status = Document(tags={})
        return DocumentArray([status])

    @requests(on='/clear')
    def clear(self, **kwargs):
        """Clear the index of all entries."""
        pass
