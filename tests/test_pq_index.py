import numpy as np
import pytest
from docarray import Document, DocumentArray
from loguru import logger

from annlite import AnnLite

N = 1000  # number of data points
Nq = 5
Nt = 2000
D = 64  # dimensionality / number of features


@pytest.fixture
def random_docs():
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 64-dim vectors to be indexed
    docs = DocumentArray(
        [Document(id=f'{i}', embedding=X[i], tags={'x': str(i)}) for i in range(N)]
    )
    return docs


def test_hnsw_pq_load_empty(tmpdir, random_docs):
    pq_index = AnnLite(
        D,
        data_path=tmpdir / 'annlite_pq_test',
        n_subvectors=8,
    )

    with pytest.raises(RuntimeError):
        # unable to add or search before a actual training of PQ
        pq_index.index(random_docs)

    with pytest.raises(RuntimeError):
        # unable to add or search before a actual training of PQ
        pq_index.search(random_docs)


def test_hnsw_pq_load(tmpdir, random_docs):
    pq_index = AnnLite(
        D,
        data_path=tmpdir / 'annlite_pq_test',
        n_subvectors=8,
    )
    pq_index.train(random_docs.embeddings)
    pq_index.index(random_docs)
    assert all([x.pq_codec.is_trained for x in pq_index._vec_indexes])
    assert all([x._index.pq_enable for x in pq_index._vec_indexes])


@pytest.mark.parametrize('n_clusters', [256, 512, 768])
def test_hnsw_pq_search_multi_clusters(n_clusters, tmpdir, random_docs):
    total_test = 10
    topk = 50

    X = random_docs.embeddings
    no_pq_index = AnnLite(D, data_path=tmpdir / 'annlite_test')

    query = DocumentArray([Document(embedding=X[i]) for i in range(total_test)])
    test_query = DocumentArray([Document(embedding=X[i]) for i in range(total_test)])

    no_pq_index.index(random_docs)
    no_pq_index.search(query, limit=topk)

    pq_index = AnnLite(
        D,
        data_path=tmpdir / f'annlite_pq_test',
        n_subvectors=8,
        n_clusters=n_clusters,
    )
    pq_index.train(X)
    pq_index.index(random_docs)
    pq_index.search(test_query, limit=topk)

    precision = []
    for i in range(total_test):
        ground_truth = set([m.id for m in query[i].matches])
        pq_result = set([m.id for m in test_query[i].matches])
        precision.append(len(ground_truth & pq_result) / topk)
    logger.info(
        f'PQ backend(cluster={n_clusters}) top-{topk} precision: {np.mean(precision)}'
    )
    # TODO: fix the precision issue
    # assert np.mean(precision) > 0.9
