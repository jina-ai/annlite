import pytest
import random
import numpy as np
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
    ids = list(range(len(X)))
    tags = [{'x': random.random()} for _ in range(N)]
    pqlite.add(X, ids, doc_tags=tags)
    return pqlite


def test_index_add(pqlite):
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed
    ids = list(range(len(X)))
    pqlite.add(X, ids)


def test_index_add_with_tags(pqlite):
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed
    ids = list(range(len(X)))
    tags = [{'x': random.random()} for _ in range(N)]
    pqlite.add(X, ids, doc_tags=tags)


def test_index_query(pqlite_with_data):
    query = np.random.random((Nq, D)).astype(np.float32)  # a 128-dim query vector
    # print(f'query: {query}')
    dists, ids = pqlite_with_data.search(query)


def test_index_query_with_filtering(pqlite_with_data):
    query = np.random.random((Nq, D)).astype(np.float32)
    conditions = [('x', '>', 0.1)]
    dists, ids = pqlite_with_data.search(query, conditions=conditions)
    print(f'dists: {dists}')
    print(f'ids: {ids}')
