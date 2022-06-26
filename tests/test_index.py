import operator
import random
from collections import namedtuple

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
n_examples = 512
n_clusters = 32
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
def build_pq_data():
    Xt = np.random.random((n_examples, D)).astype(np.float32)
    return Xt


@pytest.fixture
def build_pq_codec(build_pq_data):
    Xt = build_pq_data
    pq_codec = PQCodec(dim=D, n_subvectors=n_subvectors, n_clusters=n_clusters)
    pq_codec.fit(Xt)
    return pq_codec


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


def test_annlite_hnsw_pq_init(tmpdir, build_pq_codec):
    index = AnnLite(
        dim=D, data_path=tmpdir / 'annlite_test', hnsw_using_pq=build_pq_codec
    )


def test_annlite_hnsw_pq_interface(tmpdir, build_pq_codec):
    missing_method = namedtuple('PQCodec', ['encode', 'get_codebook'])(0, 0)
    with pytest.raises(IndexError):
        AnnLite(dim=D, data_path=tmpdir / 'annlite_test', hnsw_using_pq=missing_method)

    wrong_attrs = namedtuple(
        'PQCodec', ['encode', 'get_codebook', 'get_subspace_splitting']
    )(1, 1, 1)
    with pytest.raises(AttributeError):
        AnnLite(dim=D, data_path=tmpdir / 'annlite_test', hnsw_using_pq=wrong_attrs)

    wrong_dims = PQCodec(dim=D * 2, n_subvectors=n_subvectors, n_clusters=n_clusters)
    with pytest.raises(ValueError):
        AnnLite(dim=D, data_path=tmpdir / 'annlite_test', hnsw_using_pq=wrong_dims)

    assert (
        n_subvectors,
        n_clusters,
        d_subvector,
    ) == build_pq_codec.get_subspace_splitting()
    assert (
        n_subvectors,
        n_clusters,
        d_subvector,
    ) == build_pq_codec.get_codebook().shape
