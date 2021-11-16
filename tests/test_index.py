import pytest
import random
import numpy as np
from jina import Document, DocumentArray
from pqlite import PQLite

N = 1000  # number of data points
Nq = 5
Nt = 2000
D = 128  # dimensionality / number of features


@pytest.fixture
def pqlite(tmpdir):
    Xt = np.random.random((Nt, D)).astype(
        np.float32
    )  # 2,000 128-dim vectors for training

    pqlite = PQLite(dim=D, data_path=tmpdir / 'pqlite_test')
    return pqlite


@pytest.fixture
def pqlite_with_data(tmpdir):
    columns = [('x', float, True)]
    pqlite = PQLite(dim=D, columns=columns, data_path=tmpdir / 'pqlite_test')

    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray(
        [
            Document(id=f'{i}', embedding=X[i], tags={'x': random.random()})
            for i in range(N)
        ]
    )
    pqlite.index(docs)
    return pqlite


def test_index(pqlite):
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(N)])
    pqlite.index(docs)


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


def test_index_query_with_filtering(pqlite_with_data):
    X = np.random.random((Nq, D)).astype(np.float32)
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])
    conditions = [('x', '>', 0.6)]
    pqlite_with_data.search(query, conditions=conditions)

    for i in range(len(query[0].matches) - 1):
        assert (
            query[0].matches[i].scores['euclidean'].value
            <= query[0].matches[i + 1].scores['euclidean'].value
        )
