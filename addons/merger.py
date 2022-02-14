from typing import Optional, Tuple

from jina import DocumentArray, Executor, requests


class MatchMerger(Executor):
    """
    The MatchMerger merges the results of shards by appending all matches..
    """

    def __init__(
        self, metric: str = 'cosine', traversal_paths: Tuple[str, ...] = '@r', **kwargs
    ):
        """
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        :param traversal_paths: traverse path on docs, e.g. '@r', '@c'
        """
        super().__init__(**kwargs)
        self.metric = metric
        self.traversal_paths = traversal_paths

    @requests(on='/search')
    def merge(
        self, docs: Optional[DocumentArray] = None, parameters: dict = {}, **kwargs
    ):
        traversal_paths = (
            parameters.get('traversal_paths', self.traversal_paths) or '@r'
        )

        if not docs:
            return

        for doc in docs[traversal_paths]:
            doc.matches = sorted(doc.matches, key=lambda m: m.scores[self.metric].value)
