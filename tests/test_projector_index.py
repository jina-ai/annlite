import numpy as np
import pytest
from docarray import Document, DocumentArray

from annlite.index import AnnLite

n_examples = 1000
n_features = 512
n_components = 128
batch_size = 200


@pytest.fixture
def build_data():
    Xt = np.random.random((n_examples, n_features)).astype(np.float32)
    return Xt


@pytest.fixture
def build_projector_annlite(tmpfile):
    index = AnnLite(n_dim=n_features, data_path=tmpfile)
    return index


@pytest.fixture
def projector_annlite_with_data(build_data, build_projector_annlite):
    Xt = build_data
    indexer = build_projector_annlite

    docs = DocumentArray(
        [Document(id=f'{i}', embedding=Xt[i]) for i in range(n_examples)]
    )
    indexer.index(docs)
    return indexer


def test_delete(projector_annlite_with_data):
    indexer = projector_annlite_with_data
    indexer.delete(['0', '1'])


def test_update(projector_annlite_with_data):
    indexer = projector_annlite_with_data
    X = np.random.random((5, n_features)).astype(np.float32)
    docs = DocumentArray([Document(id=f'{i}', embedding=X[i]) for i in range(5)])
    indexer.update(docs)


def test_query(projector_annlite_with_data):
    indexer = projector_annlite_with_data
    X = np.random.random((5, n_features)).astype(np.float32)  # a 128-dim query vector
    query = DocumentArray([Document(embedding=X[i]) for i in range(5)])

    indexer.search(query)

    for i in range(len(query[0].matches) - 1):
        assert (
            query[0].matches[i].scores['euclidean'].value
            <= query[0].matches[i + 1].scores['euclidean'].value
        )
