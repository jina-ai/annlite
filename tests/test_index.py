import operator
import os
import random
import uuid
from unittest.mock import patch

import hubble
import numpy as np
import pytest
from docarray import Document, DocumentArray

from annlite import AnnLite

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

token = 'ed17d158d95d3f53f60eed445d783c80'


@pytest.fixture
def annlite_index(tmpfile):
    Xt = np.random.random((Nt, D)).astype(
        np.float32
    )  # 2,000 128-dim vectors for training

    index = AnnLite(n_dim=D, data_path=tmpfile)
    return index


@pytest.fixture
def annlite_with_data(tmpfile):
    columns = [('x', float)]
    index = AnnLite(n_dim=D, columns=columns, data_path=tmpfile)

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
def heterogenenous_da(tmpfile):
    prices = [10.0, 25.0, 50.0, 100.0]
    categories = ['comics', 'movies', 'audiobook']

    columns = [('price', float), ('category', str)]
    index = AnnLite(n_dim=D, columns=columns, data_path=tmpfile)

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
def annlite_with_heterogeneous_tags(tmpfile, heterogenenous_da):
    columns = [('price', float), ('category', str)]
    index = AnnLite(n_dim=D, columns=columns, data_path=tmpfile)
    index.index(heterogenenous_da)
    return index


def test_index(annlite_index):
    X = np.random.random((N, D)).astype(
        np.float32
    )  # 10,000 128-dim vectors to be indexed

    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(N)])
    annlite_index.index(docs)


@pytest.mark.parametrize('dtype', [np.int64, np.float32, np.float64])
def test_dtype(annlite_index, dtype):
    X = np.random.random((N, D)).astype(dtype)  # 10,000 128-dim vectors to be indexed

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


def delete_artifact(tmpname):
    client = hubble.Client(token=token, max_retries=None, jsonify=True)
    art_list = client.list_artifacts(filter={'metaData.name': tmpname})
    for art in art_list['data']:
        client.delete_artifact(id=art['_id'])


def test_local_backup_restore(tmpdir):
    X = np.random.random((N, D))
    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(N)])
    index = AnnLite(n_dim=D, data_path=tmpdir / 'workspace' / '0')
    index.index(docs)

    tmpname = uuid.uuid4().hex
    index.backup()
    index.close()

    index = AnnLite(n_dim=D, data_path=tmpdir / 'workspace' / '0')
    index.restore()

    status = index.stat
    assert int(status['total_docs']) == N
    assert int(status['index_size']) == N


@pytest.mark.skip(reason='This test requires a running hubble instance')
def test_remote_backup_restore(tmpdir):
    X = np.random.random((N, D))
    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(N)])
    index = AnnLite(n_dim=D, data_path=tmpdir / 'workspace' / '0')
    index.index(docs)

    tmpname = uuid.uuid4().hex
    index.backup(target_name='test_remote_backup_restore', token=token)

    index = AnnLite(n_dim=D, data_path=tmpdir / 'workspace' / '0')
    index.restore(source_name='test_remote_backup_restore', token=token)

    delete_artifact(tmpname)
    status = index.stat
    assert int(status['total_docs']) == N
    assert int(status['index_size']) == N
