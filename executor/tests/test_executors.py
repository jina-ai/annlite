from jina import Document, DocumentArray, Flow, Executor
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
from executor_pqlite import PQLiteIndexer
import random
import numpy as np
import pytest


N = 1000  # number of data points
Nt = 2000
Nu = 999 # number of data update
Nq = 10
D = 128  # dimentionality / number of features


def gen_docs(num):
    res = DocumentArray()
    k = np.random.random((num, D)).astype(
        np.float32
    )
    for i in range(num):
        doc = Document(id=i,embedding=k[i],tags={'x': random.random()})
        res.append(doc)
    return res


# currently the executor don't have function for training
# def test_train():
#     Xt = np.random.random((Nt, D)).astype(
#         np.float32
#     )  # 2,000 128-dim vectors for training
#     pq = PQLite(dim=D, n_cells=64, n_subvectors=8, columns=[('x', float, True)])
#     pq.fit(Xt)
#     return pq


def test_index():
    docs = gen_docs(N)
    f = Flow().add(
        uses = PQLiteIndexer,
        uses_with={
            'dim': D,
        },
    )
    with f:
        result = f.post(on='/index', inputs = docs,return_results=True)
        # assert len(result[0].data)==1000


def test_update():
    docs = gen_docs(N)
    docs_update = gen_docs(Nu)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
    )
    with f:
        f.post(on='/index', inputs=docs)
        update_res = f.post(on='/update', inputs=docs_update, return_results=True)
        # assert len(update_res[0].data)==1000


def test_search():
    docs = gen_docs(N)
    docs_query = gen_docs(Nq)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
    )
    with f:
        f.post(on='/index', inputs=docs)
        query_res = f.post(on='/query', inputs=docs_query, return_results=True)


def test_status():
    docs = gen_docs(N)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
    )
    with f:
        f.post(on='/index', inputs=docs)
        res = f.post(on='/status',return_results=True)


def test_clear():
    docs = gen_docs(N)
    f = Flow().add(
        uses=PQLiteIndexer,
        uses_with={
            'dim': D,
        },
    )
    with f:
        f.post(on='/index', inputs=docs)
        res = f.post(on='/clear',return_results=True)
        # assert len(res[0].data)==0

