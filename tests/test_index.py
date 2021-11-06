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
def pqlite():
    Xt = np.random.random((Nt, D)).astype(
        np.float32
    )  # 2,000 128-dim vectors for training

    pqlite = PQLite(d_vector=D, n_cells=8, n_subvectors=4)
    pqlite.fit(Xt)
    return pqlite


@pytest.fixture
def pqlite_with_data():
    columns = [('x', float, True)]
    pqlite = PQLite(d_vector=D, n_cells=8, n_subvectors=4, columns=columns)

    Xt = np.random.random((Nt, D)).astype(np.float32)
    pqlite.fit(Xt)

    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray([
        Document(embedding=X[i], tags={'x': random.random()}) for i in range(N)
    ])
    pqlite.add(docs)
    return pqlite


def test_index_add(pqlite):
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray([Document(embedding=X[i]) for i in range(N)])
    pqlite.add(docs)


def test_index_delete(pqlite_with_data):
    pqlite_with_data.delete(['0', '1'])


def test_index_update(pqlite_with_data):
    X = np.random.random((5, D)).astype(np.float32)  # 5 128-dim vectors to be indexed
    docs = DocumentArray([Document(embedding=X[i]) for i in range(5)])
    pqlite_with_data.update(docs)


def test_index_query(pqlite_with_data):
    X = np.random.random((Nq, D)).astype(np.float32)  # a 128-dim query vector
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])
    dists, ids = pqlite_with_data.search(query)


def test_index_query_with_filtering(pqlite_with_data):
    X = np.random.random((Nq, D)).astype(np.float32)
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])
    conditions = [('x', '>', 0.1)]
    dists, ids = pqlite_with_data.search(query, conditions=conditions)
    print(f'dists: {dists}')
    print(f'ids: {ids}')
