from jina import Document, DocumentArray, Flow, Executor

from executor_pqlite import PQLiteIndexer
import numpy as np
import pytest


N = 1000  # number of data points
Nt = 2000
Nu = 999  # number of data update
Nq = 10
D = 128  # dimentionality / number of features


def gen_docs(num):
    res = DocumentArray()
    k = np.random.random((num, D)).astype(np.float32)
    for i in range(num):
        doc = Document(id=i, embedding=k[i])
        res.append(doc)
    return res

def docs_with_tags(N):
    prices = [10., 25., 50., 100.]
    categories = ['comics', 'movies', 'audiobook']
    X = np.random.random((N, D)).astype(
        np.float32
    )
    docs = [Document(id=f'{i}', embedding=X[i], tags={'price': np.random.choice(prices),
                                                      'category': np.random.choice(categories)}) for i in range(N)]
    da = DocumentArray(docs)

    return da


def test_index(tmpdir):
    metas = {'workspace': str(tmpdir)}
    docs = gen_docs(N)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
        uses_metas=metas,
    )
    with f:
        result = f.post(on='/index', inputs=docs, return_results=True)
        assert sum([len(r.docs) for r in result]) == N


def test_update(tmpdir):
    metas = {'workspace': str(tmpdir)}
    docs = gen_docs(N)
    docs_update = gen_docs(Nu)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
        uses_metas=metas,
    )
    with f:
        f.post(on='/index', inputs=docs)
        update_res = f.post(on='/update', inputs=docs_update, return_results=True)
        assert sum([len(r.docs) for r in update_res]) == Nu

        res = f.post(on='/status', return_results=True)
        assert int(res[0].docs[0].tags['total_docs']) == N
        assert int(res[0].docs[0].tags['index_size']) == N


def test_search(tmpdir):
    metas = {'workspace': str(tmpdir)}
    docs = gen_docs(N)
    docs_query = gen_docs(Nq)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
        uses_metas=metas,
    )
    with f:
        f.post(on='/index', inputs=docs)
        query_res = f.post(on='/search', inputs=docs_query, return_results=True)
        assert sum([len(r.docs) for r in query_res]) == Nq

        for i in range(len(query_res[0].docs[0].matches) - 1):
            assert (
                    query_res[0].docs[0].matches[i].scores['euclidean'].value
                    <= query_res[0].docs[0].matches[i+1].scores['euclidean'].value
            )



def test_search_with_filtering(tmpdir):
    metas = {'workspace': str(tmpdir)}

    docs = docs_with_tags(N)
    docs_query = gen_docs(1)
    columns = [('price', 'float', 'True'), ('category', 'str', 'True')]

    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
            'columns': columns
        },
        uses_metas=metas,
    )

    conditions = [['price', '<', '50.']]
    with f:
        f.post(on='/index', inputs=docs)
        query_res = f.post(on='/search',
                           inputs=docs_query,
                           return_results=True,
                           parameters={'conditions': conditions}
                           )
        assert all([m.tags['price'] < 50 for m in query_res[0].docs[0].matches])




def test_delete(tmpdir):
    metas = {'workspace': str(tmpdir)}
    docs = gen_docs(N)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
        uses_metas=metas,
    )
    with f:
        f.post(on='/index', inputs=docs)
        res = f.post(on='/status', return_results=True)
        assert int(res[0].docs[0].tags['total_docs']) == N
        assert int(res[0].docs[0].tags['index_size']) == N

        f.post(on='/delete', inputs=docs[:5])
        res = f.post(on='/status', return_results=True)
        assert int(res[0].docs[0].tags['total_docs']) == N - 5
        assert int(res[0].docs[0].tags['index_size']) == N - 5

        docs_query = gen_docs(Nq)
        query_res = f.post(on='/search', inputs=docs_query, return_results=True)



def test_status(tmpdir):
    metas = {'workspace': str(tmpdir)}
    docs = gen_docs(N)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
        uses_metas=metas,
    )
    with f:
        f.post(on='/index', inputs=docs)
        res = f.post(on='/status', return_results=True)
        assert int(res[0].docs[0].tags['total_docs']) == N
        assert int(res[0].docs[0].tags['index_size']) == N


def test_clear(tmpdir):
    metas = {'workspace': str(tmpdir)}
    docs = gen_docs(N)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
        uses_metas=metas,
    )
    with f:
        f.post(on='/index', inputs=docs)
        f.post(on='/clear', return_results=True)
        res = f.post(on='/status', return_results=True)
        assert int(res[0].docs[0].tags['total_docs']) == 0
        assert int(res[0].docs[0].tags['index_size']) == 0
