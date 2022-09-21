import numpy as np
import pytest
from docarray import Document, DocumentArray

from annlite import AnnLite

np.random.seed(123456)

D = 50
N = 1000


@pytest.fixture
def index_data():
    index_data = DocumentArray()
    for i in range(N):
        index_data.append(Document(id=str(i)))
    half_embedding = np.random.random((N, D // 2))
    index_data.embeddings = np.concatenate([half_embedding, half_embedding], axis=1)
    return index_data


def test_dump_load(tmpfile, index_data):
    query = index_data[0:1]

    index = AnnLite(D, data_path=tmpfile)
    index.index(index_data)
    index.search(query, limit=10)
    gt = [m.id for m in query['@m']]
    index.dump()
    index.close()

    new_index = AnnLite(D, data_path=tmpfile)
    new_index.search(query, limit=10)
    new_gt = [m.id for m in query['@m']]
    assert len(set(gt) & set(new_gt)) / len(gt) == 1.0
    new_index.close()

    new_index = AnnLite(D, n_components=D // 2, data_path=tmpfile)
    new_index.search(query, limit=10)
    new_gt = [m.id for m in query['@m']]
    assert len(set(gt) & set(new_gt)) / len(gt) > 0.6
