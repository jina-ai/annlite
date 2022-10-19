import os
import time
from unittest.mock import patch

import hubble
import numpy as np
import pytest
from jina import Document, DocumentArray, Executor, Flow

from annlite.executor import AnnLiteIndexer

N = 1000  # number of data points
Nt = 2000
Nu = 999  # number of data update
Nq = 10
D = 128  # dimentionality / number of features


def gen_docs(num):
    res = DocumentArray()
    k = np.random.random((num, D)).astype(np.float32)
    for i in range(num):
        doc = Document(id=f'{i}', embedding=k[i])
        res.append(doc)
    return res


def docs_with_tags(N):
    prices = [10.0, 25.0, 50.0, 100.0]
    categories = ['comics', 'movies', 'audiobook']
    X = np.random.random((N, D)).astype(np.float32)
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


def clear_hubble():
    client = hubble.Client(max_retries=None, jsonify=True)
    art_list = client.list_artifacts()
    for art in art_list['data']:
        client.delete_artifact(id=art['_id'])


def test_index(tmpfile):
    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
        },
        workspace=tmpfile,
    )
    with f:
        result = f.post(on='/index', inputs=docs, return_results=True)
        assert len(result) == N


def test_update(tmpfile):
    docs = gen_docs(N)
    docs_update = gen_docs(Nu)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
        },
        workspace=tmpfile,
    )
    with f:
        f.post(on='/index', inputs=docs)

        time.sleep(2)

        update_res = f.post(on='/update', inputs=docs_update, return_results=True)
        assert len(update_res) == Nu

        status = f.post(on='/status', return_results=True)[0]

        assert int(status.tags['total_docs']) == N
        assert int(status.tags['index_size']) == N


def test_search(tmpfile):
    docs = gen_docs(N)
    docs_query = gen_docs(Nq)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
        },
        workspace=tmpfile,
    )
    with f:
        f.post(on='/index', inputs=docs)

        time.sleep(2)

        query_res = f.post(on='/search', inputs=docs_query, return_results=True)
        assert len(query_res) == Nq

        for i in range(len(query_res[0].matches) - 1):
            assert (
                query_res[0].matches[i].scores['cosine'].value
                <= query_res[0].matches[i + 1].scores['cosine'].value
            )


@pytest.mark.parametrize(
    'columns',
    [[('price', 'float'), ('category', 'str')], {'price': 'float', 'category': 'str'}],
)
def test_search_with_filtering(tmpfile, columns):
    docs = docs_with_tags(N)
    docs_query = gen_docs(1)

    f = Flow().add(
        uses=AnnLiteIndexer, uses_with={'dim': D, 'columns': columns}, workspace=tmpfile
    )

    with f:
        f.post(on='/index', inputs=docs)
        time.sleep(2)

        query_res = f.post(
            on='/search',
            inputs=docs_query,
            return_results=True,
            parameters={'filter': {'price': {'$lt': 50.0}}, 'include_metadata': True},
        )
        assert all([m.tags['price'] < 50 for m in query_res[0].matches])


def test_delete(tmpfile):
    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'dim': D,
        },
        workspace=tmpfile,
    )
    with f:
        f.post(on='/index', inputs=docs)
        time.sleep(2)

        status = f.post(on='/status', return_results=True)[0]
        assert int(status.tags['total_docs']) == N
        assert int(status.tags['index_size']) == N

        f.post(on='/delete', parameters={'ids': [d.id for d in docs[:5]]})
        status = f.post(on='/status', return_results=True)[0]
        assert int(status.tags['total_docs']) == N - 5
        assert int(status.tags['index_size']) == N - 5

        docs_query = gen_docs(Nq)
        query_res = f.post(on='/search', inputs=docs_query, return_results=True)


def test_status(tmpfile):
    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'dim': D,
        },
        workspace=tmpfile,
    )
    with f:
        f.post(on='/index', inputs=docs)
        time.sleep(2)
        status = f.post(on='/status', return_results=True)[0]
        assert int(status.tags['total_docs']) == N
        assert int(status.tags['index_size']) == N


def test_clear(tmpfile):
    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'dim': D,
        },
        workspace=tmpfile,
    )
    with f:
        f.post(on='/index', inputs=docs)
        f.post(on='/clear')
        status = f.post(on='/status', return_results=True)[0]
        assert int(status.tags['total_docs']) == 0
        assert int(status.tags['index_size']) == 0


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': ''})
def test_remote_storage(tmpfile):
    os.environ['JINA_AUTH_TOKEN'] = 'ed17d158d95d3f53f60eed445d783c80'
    clear_hubble()

    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
        },
        workspace=tmpfile,
        shards=1,
    )
    with f:
        f.post(on='/index', inputs=docs)
        time.sleep(2)
        f.post(on='/backup', parameters={'target_name': 'backup_docs'})
        time.sleep(2)

    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={'n_dim': D, 'restore_key': 'backup_docs'},
        workspace=tmpfile,
        shards=1,
    )
    with f:
        status = f.post(on='/status', return_results=True)[0]

    assert int(status.tags['total_docs']) == N
    assert int(status.tags['index_size']) == N


def test_local_storage(tmpfile):
    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
        },
        workspace=tmpfile,
        shards=1,
    )
    with f:
        f.post(on='/index', inputs=docs)
        time.sleep(2)
        f.post(on='/backup')
        time.sleep(2)

    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={'n_dim': D},
        workspace=tmpfile,
        shards=1,
    )
    with f:
        status = f.post(on='/status', return_results=True)[0]

    assert int(status.tags['total_docs']) == N
    assert int(status.tags['index_size']) == N


@patch.dict(os.environ, {'JINA_AUTH_TOKEN': ''})
def test_remote_storage_with_shards(tmpfile):
    os.environ['JINA_AUTH_TOKEN'] = 'ed17d158d95d3f53f60eed445d783c80'
    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
        },
        workspace=tmpfile,
        shards=3,
        polling={'/index': 'ANY', '/search': 'ALL', '/backup': 'ALL', '/status': 'ALL'},
    )
    with f:
        f.post(on='/index', inputs=docs)
        time.sleep(2)
        f.post(
            on='/backup',
            parameters={'target_name': 'backup_docs_with_shards'},
        )
        time.sleep(2)

    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
            'restore_key': 'backup_docs_with_shards',
        },
        workspace=tmpfile,
        shards=3,
        polling={'/index': 'ANY', '/search': 'ALL', '/backup': 'ALL', '/status': 'ALL'},
    )
    with f:
        status = f.post(on='/status', return_results=True)

    total_docs = 0
    index_size = 0
    for stat in status:
        total_docs += stat.tags['total_docs']
        index_size += stat.tags['index_size']
    assert total_docs == N
    assert index_size == N
    clear_hubble()


def test_local_storage_with_shards(tmpfile):
    docs = gen_docs(N)
    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={
            'n_dim': D,
        },
        workspace=tmpfile,
        shards=3,
        polling={'/index': 'ANY', '/search': 'ALL', '/backup': 'ALL', '/status': 'ALL'},
    )
    with f:
        f.post(on='/index', inputs=docs)
        time.sleep(2)
        f.post(on='/backup')
        time.sleep(2)

    f = Flow().add(
        uses=AnnLiteIndexer,
        uses_with={'n_dim': D},
        workspace=tmpfile,
        shards=3,
        polling={'/index': 'ANY', '/search': 'ALL', '/backup': 'ALL', '/status': 'ALL'},
    )
    with f:
        status = f.post(on='/status', return_results=True)

    total_docs = 0
    index_size = 0
    for stat in status:
        total_docs += stat.tags['total_docs']
        index_size += stat.tags['index_size']
    assert total_docs == N
    assert index_size == N
