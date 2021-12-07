import pytest
import random
import numpy as np
from jina import Document, DocumentArray
from pqlite import PQLite
import operator

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

    index = PQLite(dim=D, data_path=tmpdir / 'pqlite_test')
    return index


@pytest.fixture
def pqlite_with_data(tmpdir):
    columns = [('x', float)]
    index = PQLite(dim=D, columns=columns, data_path=tmpdir / 'pqlite_test')

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
def pqlite_with_heterogeneous_tags(tmpdir):
    prices = [10.0, 25.0, 50.0, 100.0]
    categories = ['comics', 'movies', 'audiobook']

    columns = [('price', float), ('category', str)]
    index = PQLite(dim=D, columns=columns, data_path=tmpdir / 'pqlite_test')

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

    index.index(da)
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
