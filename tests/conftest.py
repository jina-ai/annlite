import tempfile

import numpy as np
import pytest


@pytest.fixture(scope='session')
def docs():
    return [
            dict(id='doc1', embedding=np.array([1, 0, 0, 0])),
            dict(id='doc2', embedding=np.array([0, 1, 0, 0])),
            dict(id='doc3', embedding=np.array([0, 0, 1, 0])),
            dict(id='doc4', embedding=np.array([0, 0, 0, 1])),
            dict(id='doc5', embedding=np.array([1, 0, 1, 0])),
            dict(id='doc6', embedding=np.array([0, 1, 0, 1])),
        ]


@pytest.fixture(scope='session')
def update_docs():
    return [
            dict(id='doc1', embedding=np.array([0, 0, 0, 1])),
        ]


@pytest.fixture(autouse=True)
def tmpfile(tmpdir):
    tmpfile = f'annlite_test_{next(tempfile._get_candidate_names())}.db'
    return tmpdir / tmpfile
