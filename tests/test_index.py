import operator
import random

import numpy as np
import pytest
from docarray import Document, DocumentArray

from annlite import AnnLite

N = 1000  # number of data points
Nq = 5
Nt = 2000
D = 128  # dimensionality / number of features

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
def pqlite_index(tmpdir):
    Xt = np.random.random((Nt, D)).astype(
        np.float32
    )  # 2,000 128-dim vectors for training

    index = AnnLite(dim=D, data_path=tmpdir / 'pqlite_test')
    return index


@pytest.fixture
def pqlite_with_data(tmpdir):
    columns = [('x', float)]
    index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'pqlite_test')

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
    index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'pqlite_test')

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
def pqlite_with_heterogeneous_tags(tmpdir, heterogenenous_da):

    columns = [('price', float), ('category', str)]
    index = AnnLite(dim=D, columns=columns, data_path=tmpdir / 'pqlite_test')
    index.index(heterogenenous_da)
    return index


def test_index(pqlite_index):
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(N)])
    pqlite_index.index(docs)


def test_delete(pqlite_with_data):
    pqlite_with_data.delete(['0', '1'])


def test_update(pqlite_with_data):
    X = np.random.random((5, D)).astype(np.float32)  # 5 128-dim vectors to be indexed
    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(5)])
    pqlite_with_data.update(docs)


def test_query(pqlite_with_data):
    X = np.random.random((Nq, D)).astype(np.float32)  # a 128-dim query vector
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])

    pqlite_with_data.search(query)

    for i in range(len(query[0].matches) - 1):
        assert (
            query[0].matches[i].scores['euclidean'].value
            <= query[0].matches[i + 1].scores['euclidean'].value
        )


def test_index_query_with_filtering_sorted_results(pqlite_with_data):
    X = np.random.random((Nq, D)).astype(np.float32)
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])
    pqlite_with_data.search(query, filter={'x': {'$gt': 0.6}}, include_metadata=True)

    for i in range(len(query[0].matches) - 1):
        assert (
            query[0].matches[i].scores['euclidean'].value
            <= query[0].matches[i + 1].scores['euclidean'].value
        )
        assert query[0].matches[i].tags['x'] > 0.6


@pytest.mark.parametrize('operator', list(numeric_operators.keys()))
def test_query_search_filter_float_type(pqlite_with_heterogeneous_tags, operator):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_da = DocumentArray([Document(embedding=X[i]) for i in range(Nq)])

    thresholds = [20, 50, 100, 400]

    for threshold in thresholds:
        pqlite_with_heterogeneous_tags.search(
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
    pqlite_with_heterogeneous_tags, heterogenenous_da, operator
):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_np = np.array([X[i] for i in range(Nq)])
    da = heterogenenous_da
    thresholds = [20, 50, 100, 400]

    for threshold in thresholds:
        dists, doc_ids = pqlite_with_heterogeneous_tags.search_numpy(
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
def test_search_filter_str(pqlite_with_heterogeneous_tags, operator):
    X = np.random.random((Nq, D)).astype(np.float32)
    query_da = DocumentArray([Document(embedding=X[i]) for i in range(Nq)])

    categories = ['comics', 'movies', 'audiobook']
    for category in categories:
        pqlite_with_heterogeneous_tags.search(
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
    pqlite_with_heterogeneous_tags, heterogenenous_da, operator
):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_np = np.array([X[i] for i in range(Nq)])
    da = heterogenenous_da

    categories = ['comics', 'movies', 'audiobook']
    for category in categories:
        dists, doc_ids = pqlite_with_heterogeneous_tags.search_numpy(
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
    pqlite_with_heterogeneous_tags, heterogenenous_da
):

    X = np.random.random((Nq, D)).astype(np.float32)
    query_np = np.array([X[i] for i in range(Nq)])
    da = heterogenenous_da

    dists, doc_ids = pqlite_with_heterogeneous_tags.search_numpy(
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

    dists, doc_ids = pqlite_with_heterogeneous_tags.search_numpy(
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
