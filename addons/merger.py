import warnings
from typing import List, Tuple

from jina import DocumentArray, Executor, requests


class MatchMerger(Executor):
    """
    The MatchMerger merges the results of shards by appending all matches..
    """

    def __init__(self, default_traversal_paths: Tuple[str, ...] = '@r', **kwargs):
        """
        :param default_traversal_paths: traverse path on docs, e.g. '@r', '@c'
        """
        super().__init__(**kwargs)
        warnings.warn(
            'The functionality of MatchMerger is subsumed by the default behaviour starting with'
            'Jina 3. Consider dropping MatchMerger from your flows. MatchMerger might stop working'
            'with future versions of Jina or Jina Hub.',
            DeprecationWarning,
        )

        self.default_traversal_paths = default_traversal_paths

    @requests
    def merge(
        self, docs_matrix: List[DocumentArray] = [], parameters: dict = {}, **kwargs
    ):
        traversal_paths = parameters.get(
            'traversal_paths', self.default_traversal_paths
        )
        results = {}

        if not docs_matrix:
            return
        for docs in docs_matrix:
            self._merge_shard(results, docs, traversal_paths)
        return DocumentArray(list(results.values()))

    def _merge_shard(self, results, docs, traversal_paths):
        for doc in docs[traversal_paths]:
            if doc.id in results:
                results[doc.id].matches.extend(doc.matches)
            else:
                results[doc.id] = doc
