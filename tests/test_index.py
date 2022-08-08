import enum
import operator
import random
from collections import namedtuple
from distutils.command.build import build
from pathlib import Path

import numpy as np
import pytest
from docarray import Document, DocumentArray

import annlite
from annlite import AnnLite
from annlite.core.codec.pq import PQCodec
from annlite.core.index import hnsw

N = 1000  # number of data points
Nq = 5
Nt = 2000
D = 128  # dimensionality / number of features
# pq params below -----------
n_clusters = 256
n_subvectors = 8
d_subvector = int(D / n_subvectors)

numeric_operators = {
    '$gte': operator.ge,
    '$gt': operator.gt,
    '$lte': operator.le,
    '$lt': operator.lt,
    '$eq': operator.eq,
    '$neq': operator.ne,
}

categorical_operators = {'$eq': operator.eq, '$neq': operator.ne}


@pytest.fixture
def vanilla_annlite_hash():
    import hashlib

    n_cells = 1
    metric_name = 'COSINE'
    key = f'{n_cells} x {n_subvectors} x {metric_name}'
    return hashlib.md5(key.encode()).hexdigest()


@pytest.fixture
def random_docs():
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed
    docs = DocumentArray(
        [Document(id=f'{i}', embedding=X[i], tags={'x': str(i)}) for i in range(N)]
    )
    return docs


@pytest.fixture
def annlite_index(tmpdir):
    Xt = np.random.random((Nt, D)).astype(
        np.float32
    )  # 2,000 128-dim vectors for training

    index = AnnLite(dim=D, data_path=tmpdir / 'annlite_test')
    return index


@pytest.fixture
def annlite_with_data(tmpdir):
    columns = [('x', float)]
    index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'annlite_test')

    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray(
        [
            Document(id=f'{i}', embedding=X[i], tags={'x': random.random()})
            for i in range(N)
        ]
    )
    index.index(docs)
    return index


@pytest.fixture
def heterogenenous_da(tmpdir):
    prices = [10.0, 25.0, 50.0, 100.0]
    categories = ['comics', 'movies', 'audiobook']

    columns = [('price', float), ('category', str)]
    index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'annlite_test')

    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed
    docs = [
        Document(
            id=f'{i}',
            embedding=X[i],
            tags={
                'price': np.random.choice(prices),
                'category': np.random.choice(categories),
            },
        )
        for i in range(N)
    ]
    da = DocumentArray(docs)

    return da


@pytest.fixture
def annlite_with_heterogeneous_tags(tmpdir, heterogenenous_da):

    columns = [('price', float), ('category', str)]
    index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'annlite_test')
    index.index(heterogenenous_da)
    return index


def test_index(annlite_index):
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(N)])
    annlite_index.index(docs)


def test_delete(annlite_with_data):
    annlite_with_data.delete(['0', '1'])


def test_update(annlite_with_data):
    X = np.random.random((5, D)).astype(np.float32)  # 5 128-dim vectors to be indexed
    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(5)])
    annlite_with_data.update(docs)


def test_query(annlite_with_data):
    X = np.random.random((Nq, D)).astype(np.float32)  # a 128-dim query vector
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])

    annlite_with_data.search(query)

    for i in range(len(query[0].matches) - 1):
        assert (
            query[0].matches[i].scores['euclidean'].value
            <= query[0].matches[i + 1].scores['euclidean'].value
        )


def test_index_query_with_filtering_sorted_results(annlite_with_data):
    X = np.random.random((Nq, D)).astype(np.float32)
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])
    annlite_with_data.search(query, filter={'x': {'$gt': 0.6}}, include_metadata=True)

    for i in range(len(query[0].matches) - 1):
        assert (
            query[0].matches[i].scores['euclidean'].value
            <= query[0].matches[i + 1].scores['euclidean'].value
        )
        assert query[0].matches[i].tags['x'] > 0.6


@pytest.mark.parametrize('operator', list(numeric_operators.keys()))
def test_query_search_filter_float_type(annlite_with_heterogeneous_tags, operator):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_da = DocumentArray([Document(embedding=X[i]) for i in range(Nq)])

    thresholds = [20, 50, 100, 400]

    for threshold in thresholds:
        annlite_with_heterogeneous_tags.search(
            query_da, filter={'price': {operator: threshold}}, include_metadata=True
        )
        for query in query_da:
            assert all(
                [
                    numeric_operators[operator](m.tags['price'], threshold)
                    for m in query.matches
                ]
            )


@pytest.mark.parametrize('operator', list(numeric_operators.keys()))
def test_query_search_numpy_filter_float_type(
    annlite_with_heterogeneous_tags, heterogenenous_da, operator
):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_np = np.array([X[i] for i in range(Nq)])
    da = heterogenenous_da
    thresholds = [20, 50, 100, 400]

    for threshold in thresholds:
        dists, doc_ids = annlite_with_heterogeneous_tags.search_numpy(
            query_np, filter={'price': {operator: threshold}}, include_metadata=True
        )
        for doc_ids_query_k in doc_ids:
            assert all(
                [
                    numeric_operators[operator](
                        da[int(doc_id)].tags['price'], threshold
                    )
                    for doc_id in doc_ids_query_k
                ]
            )


@pytest.mark.parametrize('operator', list(categorical_operators.keys()))
def test_search_filter_str(annlite_with_heterogeneous_tags, operator):
    X = np.random.random((Nq, D)).astype(np.float32)
    query_da = DocumentArray([Document(embedding=X[i]) for i in range(Nq)])

    categories = ['comics', 'movies', 'audiobook']
    for category in categories:
        annlite_with_heterogeneous_tags.search(
            query_da, filter={'category': {operator: category}}, include_metadata=True
        )
        for query in query_da:
            assert all(
                [
                    numeric_operators[operator](m.tags['category'], category)
                    for m in query.matches
                ]
            )


@pytest.mark.parametrize('operator', list(categorical_operators.keys()))
def test_search_numpy_filter_str(
    annlite_with_heterogeneous_tags, heterogenenous_da, operator
):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_np = np.array([X[i] for i in range(Nq)])
    da = heterogenenous_da

    categories = ['comics', 'movies', 'audiobook']
    for category in categories:
        dists, doc_ids = annlite_with_heterogeneous_tags.search_numpy(
            query_np, filter={'category': {operator: category}}, include_metadata=True
        )
        for doc_ids_query_k in doc_ids:
            assert all(
                [
                    numeric_operators[operator](
                        da[int(doc_id)].tags['category'], category
                    )
                    for doc_id in doc_ids_query_k
                ]
            )


def test_search_numpy_membership_filter(
    annlite_with_heterogeneous_tags, heterogenenous_da
):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_np = np.array([X[i] for i in range(Nq)])
    da = heterogenenous_da

    dists, doc_ids = annlite_with_heterogeneous_tags.search_numpy(
        query_np,
        filter={'category': {'$in': ['comics', 'audiobook']}},
        include_metadata=True,
    )
    for doc_ids_query_k in doc_ids:
        assert len(doc_ids)
        assert all(
            [
                da[int(doc_id)].tags['category'] in ['comics', 'audiobook']
                for doc_id in doc_ids_query_k
            ]
        )

    dists, doc_ids = annlite_with_heterogeneous_tags.search_numpy(
        query_np,
        filter={'category': {'$nin': ['comics', 'audiobook']}},
        include_metadata=True,
    )
    for doc_ids_query_k in doc_ids:
        assert len(doc_ids)
        assert all(
            [
                da[int(doc_id)].tags['category'] not in ['comics', 'audiobook']
                for doc_id in doc_ids_query_k
            ]
        )


def test_annlite_hnsw_pq_load_empty(tmpdir, random_docs):
    pq_index = AnnLite(
        dim=D,
        data_path=tmpdir / 'annlite_pq_test',
        n_subvectors=n_subvectors,
    )
    with pytest.raises(RuntimeError):
        # unable to add or search before a actual training of PQ
        pq_index.index(random_docs)
    with pytest.raises(RuntimeError):
        # unable to add or search before a actual training of PQ
        pq_index.search(random_docs)


def test_annlite_hnsw_pq_load_recover(tmpdir, random_docs, vanilla_annlite_hash):
    X = random_docs.embeddings
    # building PQ from X
    build_pq_codec = PQCodec(dim=D, n_subvectors=n_subvectors, n_clusters=n_clusters)
    build_pq_codec.fit(X)

    tmpdir = Path(str(tmpdir))
    pq_dump_path = (tmpdir / 'annlite_pq_test') / vanilla_annlite_hash
    if not pq_dump_path.exists():
        pq_dump_path.mkdir(parents=True, exist_ok=True)
    build_pq_codec.dump(pq_dump_path / 'pq_codec.bin')

    pq_index = AnnLite(
        dim=D,
        data_path=tmpdir / 'annlite_pq_test',
        n_subvectors=n_subvectors,
    )
    assert all([x._index.pq_enable for x in pq_index._vec_indexes])


def test_annlite_hnsw_pq_load(tmpdir, random_docs):
    pq_index = AnnLite(
        dim=D,
        data_path=tmpdir / 'annlite_pq_test',
        n_subvectors=n_subvectors,
    )
    pq_index.train(random_docs.embeddings)
    pq_index.index(random_docs)
    assert all([x.pq_codec.is_trained for x in pq_index._vec_indexes])
    assert all([x._index.pq_enable for x in pq_index._vec_indexes])


def test_annlite_hnsw_pq_search_multi_clusters(tmpdir, random_docs):
    test_n_clusters = [256, 512, 768]
    columns = [('x', str)]
    total_test = 10
    topk = 50

    X = random_docs.embeddings
    no_pq_index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'annlite_test')

    query = DocumentArray([Document(embedding=X[i]) for i in range(total_test)])
    test_queries = [
        DocumentArray([Document(embedding=X[i]) for i in range(total_test)])
        for _ in test_n_clusters
    ]

    no_pq_index.index(random_docs)
    no_pq_index.search(query, limit=topk)
    # ------------------
    precisions = []
    for index, test_n_cluster in enumerate(test_n_clusters):
        pq_index_in = AnnLite(
            dim=D,
            columns=columns,
            data_path=tmpdir / f'annlite_pq_test_{index}',
            n_subvectors=n_subvectors,
            n_clusters=test_n_cluster,
        )
        pq_index_in.train(X)
        pq_index_in.index(random_docs)
        pq_index_in.search(test_queries[index], limit=topk)

        precision = []
        for i in range(total_test):
            ground_truth = set([i.tags['x'] for i in query[i].matches])
            pq_result = set([i.tags['x'] for i in test_queries[index][i].matches])
            precision.append(len(ground_truth.intersection(pq_result)) / topk)
        precisions.append(np.mean(precision))
    for i in range(len(precisions) - 1):
        assert precisions[i] < precisions[i + 1]


@pytest.mark.parametrize('test_n_cluster', [256])
def test_annlite_hnsw_pq_search_recover(tmpdir, random_docs, test_n_cluster):
    columns = [('x', str)]
    total_test = 10
    topk = 50

    X = random_docs.embeddings
    no_pq_index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'annlite_test')

    query = DocumentArray([Document(embedding=X[i]) for i in range(total_test)])
    query_pq = DocumentArray([Document(embedding=X[i]) for i in range(total_test)])
    query_pq_trained = DocumentArray(
        [Document(embedding=X[i]) for i in range(total_test)]
    )

    no_pq_index.index(random_docs)
    no_pq_index.search(query, limit=topk)
    # ------------------
    pq_index_in = AnnLite(
        dim=D,
        columns=columns,
        data_path=tmpdir / 'annlite_pq_test',
        n_subvectors=n_subvectors,
        n_clusters=test_n_cluster,
    )
    pq_index_in.train(X)
    pq_index_in.index(random_docs)
    pq_index_in.search(query_pq_trained, limit=topk)

    precision = []
    for i in range(total_test):
        ground_truth = set([i.tags['x'] for i in query[i].matches])
        pq_result = set([i.tags['x'] for i in query_pq_trained[i].matches])
        precision.append(len(ground_truth.intersection(pq_result)) / topk)

    # ------------------
    pq_index = AnnLite(
        dim=D,
        columns=columns,
        data_path=tmpdir / 'annlite_pq_test',
        n_subvectors=n_subvectors,
    )

    assert all([x._index.pq_enable for x in pq_index._vec_indexes])
    pq_index.search(query_pq, limit=topk)

    precision_recover = []
    for i in range(total_test):
        ground_truth = set([i.tags['x'] for i in query[i].matches])
        pq_result = set([i.tags['x'] for i in query_pq[i].matches])
        precision_recover.append(len(ground_truth.intersection(pq_result)) / topk)

    assert abs(np.mean(precision) - np.mean(precision_recover)) < 0.1 * np.mean(
        precision
    )
