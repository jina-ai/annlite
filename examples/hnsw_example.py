import numpy as np
import random
import tempfile
from pqlite import PQLite
from docarray import Document, DocumentArray

N = 1000  # number of data points
Nq = 5
Nt = 2000
D = 128  # dimensionality / number of features

dirpath = tempfile.mkdtemp()

with tempfile.TemporaryDirectory() as tmpdirname:

    index = PQLite(dim=D, columns=[('x', float)], data_path=tmpdirname, include_metadata=True)

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


    X = np.random.random((Nq, D)).astype(np.float32)  # a 128-dim query vector
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])

    index.search(query, filter={'x': {'$lt': 0.2}}, limit=10, include_metadata=True)

    for m in query[0].matches:
        print(f'{m.scores["euclidean"].value} -> x={m.tags["x"]}')
        assert m.tags["x"] < 0.2

    print(f'====')

    index.search(query, filter={'x': {'$gte': 0.9}}, limit=10, include_metadata=True)

    for m in query[0].matches:
        print(f'{m.scores["euclidean"].value} -> x={m.tags["x"]}')
        assert m.tags["x"] >= 0.9

#
# print(f'{[m.scores["euclidean"].value for m in query[0].matches]}')
# for i in range(len(query[0].matches) - 1):
#     assert (
#         query[0].matches[i].scores['euclidean'].value
#         <= query[0].matches[i + 1].scores['euclidean'].value
#     )