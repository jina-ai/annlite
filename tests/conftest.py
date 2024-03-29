import tempfile

import numpy as np
import pytest
from docarray import Document, DocumentArray


@pytest.fixture(scope='session')
def docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.array([1, 0, 0, 0])),
            Document(id='doc2', embedding=np.array([0, 1, 0, 0])),
            Document(id='doc3', embedding=np.array([0, 0, 1, 0])),
            Document(id='doc4', embedding=np.array([0, 0, 0, 1])),
            Document(id='doc5', embedding=np.array([1, 0, 1, 0])),
            Document(id='doc6', embedding=np.array([0, 1, 0, 1])),
        ]
    )


@pytest.fixture(scope='session')
def update_docs():
    return DocumentArray(
        [
            Document(id='doc1', embedding=np.array([0, 0, 0, 1])),
        ]
    )


@pytest.fixture(autouse=True)
def tmpfile(tmpdir):
    tmpfile = f'annlite_test_{next(tempfile._get_candidate_names())}.db'
    return tmpdir / tmpfile


@pytest.fixture(autouse=True)
def test_disable_telemetry(monkeypatch):
    monkeypatch.setenv('JINA_OPTOUT_TELEMETRY', 'True')
