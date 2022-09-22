import numpy as np
import pytest
from docarray import Document, DocumentArray
from loguru import logger

from annlite import AnnLite, pq_bind
from annlite.core.codec.pq import PQCodec
from annlite.core.index.pq_index import PQIndex
from annlite.math import cosine, l2_normalize

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


def test_pq_index_dist_mat(random_docs):
    X = random_docs.embeddings
    N, D = X.shape
    pq_class = PQCodec(dim=D)
    pq_class.fit(X)

    test_cases = X[:10]
    # -------------- true
    pq_bind_re = []
    for i in range(len(test_cases)):
        query = test_cases[i]
        dtable = pq_bind.precompute_adc_table(
            query, pq_class.d_subvector, pq_class.n_clusters, pq_class.codebooks
        )
        pq_bind_re.append(dtable)
    pq_bind_re = np.stack(pq_bind_re)
    # -------------- test
    pq_dist_mat = pq_class.get_dist_mat(test_cases)

    assert np.allclose(pq_bind_re, pq_dist_mat)


def test_hnsw_pq_load_empty(tmpfile, random_docs):
    pq_index = AnnLite(
        D,
        data_path=tmpfile,
        n_subvectors=8,
    )

    with pytest.raises(RuntimeError):
        # unable to add or search before a actual training of PQ
        pq_index.index(random_docs)

    with pytest.raises(RuntimeError):
        # unable to add or search before a actual training of PQ
        pq_index.search(random_docs)


def test_hnsw_pq_load(tmpfile, random_docs):
    pq_index = AnnLite(
        D,
        data_path=tmpfile,
        n_subvectors=8,
    )
    pq_index.train(random_docs.embeddings)
    pq_index.index(random_docs)
    assert all([x.pq_codec.is_trained for x in pq_index._vec_indexes])
    assert all([x._index.pq_enable for x in pq_index._vec_indexes])


@pytest.mark.parametrize('n_clusters', [256, 512, 768])
def test_hnsw_pq_search_multi_clusters(n_clusters, tmpfile, random_docs):
    total_test = 100
    topk = 10

    X = random_docs.embeddings
    N, Dim = X.shape
    computed_dist = cosine(X, X)
    computed_labels = np.argsort(computed_dist, axis=1)[:, :topk]

    query = DocumentArray([Document(embedding=X[i]) for i in range(total_test)])
    test_query = DocumentArray([Document(embedding=X[i]) for i in range(total_test)])

    # HNSW search with float----------------------------------
    no_pq_index = AnnLite(D, data_path=tmpfile + '_no_pq')

    no_pq_index.index(random_docs)
    no_pq_index.search(query, limit=topk)
    # ----------------------------------

    # HNSW search with quantization---------------------------
    pq_index = AnnLite(
        D,
        data_path=tmpfile + '_pq',
        n_subvectors=8,
        n_clusters=n_clusters,
    )
    pq_index.train(X)
    pq_index.index(random_docs)
    pq_index.search(test_query, limit=topk)
    # ----------------------------------

    # PQ linear search----------------------------------
    _pq_codec = pq_index.pq_codec
    ids = np.array([int(doc.id) for doc in random_docs])
    norm_x = l2_normalize(X)
    linear_pq_index = PQIndex(Dim, _pq_codec)
    linear_pq_index.add_with_ids(norm_x, ids)

    search_x = l2_normalize(test_query.embeddings)
    pq_dists = []
    linear_results = []
    for i in range(total_test):
        pq_dist, linear_result = linear_pq_index.search(search_x[i], limit=topk)
        pq_dists.append(pq_dist)
        linear_results.append(linear_result)
    # ----------------------------------

    precision = []
    original_precision = []
    pq_precision = []
    for i in range(total_test):
        real_ground_truth = set([str(i) for i in computed_labels[i]])
        ground_truth = set([m.id for m in query[i].matches])
        pq_result = set([m.id for m in test_query[i].matches])
        linear_pq_result = set([str(i_id) for i_id in linear_results[i]])
        original_precision.append(len(real_ground_truth & ground_truth) / topk)
        pq_precision.append(len(real_ground_truth & linear_pq_result) / topk)
        precision.append(len(real_ground_truth & pq_result) / topk)
    logger.info(f'Total test {total_test}')
    logger.info(f'PQ backend top-{topk} precision: {np.mean(pq_precision)}')
    logger.info(f'HNSW backend top-{topk} precision: {np.mean(original_precision)}')
    logger.info(f'HNSW PQ backend top-{topk} precision: {np.mean(precision)}')
    # TODO: fix the precision issue
    # assert np.mean(precision) > 0.9
