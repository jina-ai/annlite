from typing import List, Tuple

from jina import DocumentArray, Executor, requests
from jina.logging.logger import JinaLogger


class MatchMerger(Executor):
    """
    The MatchMerger merges the results of shards by appending all matches..
    """

    def __init__(self, metric: str = 'cosine', **kwargs):
        """
        :param metric: Distance metric type. Can be 'euclidean', 'inner_product', or 'cosine'
        """
        super().__init__(**kwargs)
        self.logger = JinaLogger(self.__class__.__name__)

        self.metric = metric

    @requests(on='/search')
    def merge(self, docs_matrix: List['DocumentArray'], **kwargs):
        if docs_matrix:
            da = docs_matrix[0]
            da.reduce_all(docs_matrix[1:])

            for doc in da:
                doc.matches = sorted(
                    doc.matches, key=lambda m: m.scores[self.metric].value
                )

            return da
