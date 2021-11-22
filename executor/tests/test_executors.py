from jina import Document, DocumentArray
# from executor import PQLiteIndexer
from pqlite import PQLite
import random
import numpy as np

N = 10000  # number of data points
Nt = 2000
Nq = 10
D = 128  # dimentionality / number of features


def gen_docs(num):
    res = DocumentArray()
    k = np.random.random((num, 128)).astype(
        np.float32
    )
    for i in range(num):
        # k[i][0] = float(i)
        doc = Document(id=i,embedding=k[i],tags={'x': random.random()})
        res.append(doc)
    return res


def test_train():
    Xt = np.random.random((Nt, D)).astype(
        np.float32
    )  # 2,000 128-dim vectors for training
    pq = PQLite(dim=D, n_cells=64, n_subvectors=8, columns=[('x', float, True)])
    pq.fit(Xt)
    return pq

def test_index():
    pq = PQLite(dim=128)
    docs = gen_docs(1000)
    pq.index(docs)
    # len = pq.doc_size
    # assert len == 1000


def test_search(pq):
    doc = gen_docs(5)
    # conditions = [('x', '>', 0.6)]
    print(doc[0].id)
    dists, ids = pq.search(docs= doc)
    for i, (dist, idx) in enumerate(zip(dists, ids)):
        print(f'query [{i}]: {dist} {idx}')


def test_update():
    pq = PQLite(dim=128)
    docs = gen_docs(1000)
    pq.index(docs)
    docs_update = gen_docs(999)
    pq.update(docs_update)
    # len = pq.doc_size
    # assert len == 1000


def test_delete():
    pq = PQLite(dim=128)
    docs = gen_docs(100)
    pq.delete(['0','66'])
    # len = pq.doc_size
    # assert len == 98


if __name__ == "__main__":
    # pq = test_train()
    test_index()
    test_update()
    test_delete()
    # test_search(pq)
