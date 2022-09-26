import random

import numpy as np
import pytest
from docarray import Document, DocumentArray

from annlite import AnnLite

N = 1000
Nt = 5
D = 128


@pytest.fixture
def annlite_with_data(tmpfile):
    columns = [('x', float)]
    index = AnnLite(n_dim=D, columns=columns, data_path=tmpfile)

    X = np.random.random((N, D)).astype(np.float32)

    docs = DocumentArray(
        [
            Document(id=f'{i}', embedding=X[i], tags={'x': random.random()})
            for i in range(N)
        ]
    )
    index.index(docs)
    return index


@pytest.mark.parametrize('filter', [None, {'x': {'$lt': 0.5}}])
@pytest.mark.parametrize('limit', [-1, 1, 3, 5])
def test_get(annlite_with_data, filter, limit):
    index = annlite_with_data

    docs = index.get_docs(filter=filter, limit=limit)
    if limit > 0:
        assert len(docs) == limit
    elif limit == -1 and filter is None:
        assert len(docs) == N

    if filter:
        for doc in docs:
            assert doc.tags['x'] < 0.5


def test_update_legal(annlite_with_data):
    index = annlite_with_data

    updated_X = np.random.random((Nt, D)).astype(np.float32)
    updated_docs = DocumentArray(
        [
            Document(id=f'{i}', embedding=updated_X[i], tags={'x': random.random()})
            for i in range(Nt)
        ]
    )

    index.update(updated_docs)
    index.search(updated_docs)
    for i in range(Nt):
        np.testing.assert_array_almost_equal(
            updated_docs[i].embedding, updated_docs[i].matches[0].embedding, decimal=5
        )


def test_update_illegal(annlite_with_data):
    index = annlite_with_data

    updated_X = np.random.random((Nt, D)).astype(np.float32)
    updated_docs = DocumentArray(
        [
            Document(
                id=f'{i}_wrong', embedding=updated_X[i], tags={'x': random.random()}
            )
            for i in range(Nt)
        ]
    )

    with pytest.raises(Exception):
        index.update(
            updated_docs, raise_errors_on_not_found=True, insert_if_not_found=False
        )
    with pytest.warns(RuntimeWarning):
        index.update(
            updated_docs, raise_errors_on_not_found=False, insert_if_not_found=False
        )
    index.update(updated_docs, raise_errors_on_not_found=True, insert_if_not_found=True)


def test_delete_legal(annlite_with_data):
    index = annlite_with_data

    deleted_docs = DocumentArray([Document(id=f'{i}') for i in range(Nt)])

    index.delete(deleted_docs)
    assert index.stat['total_docs'] == N - Nt


def test_delete_illegal(annlite_with_data):
    index = annlite_with_data

    deleted_docs = DocumentArray([Document(id=f'{i}_wrong') for i in range(Nt)])

    with pytest.raises(Exception):
        index.delete(deleted_docs, raise_errors_on_not_found=True)
