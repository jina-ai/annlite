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


@pytest.fixture(scope='module')
def start_storage():
    import os

    os.system(
        f'docker-compose -f {compose_yml} --project-directory . up  --build -d '
        f'--remove-orphans'
    )
    from elasticsearch import Elasticsearch

    es = Elasticsearch(hosts='http://localhost:9200/')
    while not es.ping():
        time.sleep(0.5)

    yield
    os.system(
        f'docker-compose -f {compose_yml} --project-directory . down '
        f'--remove-orphans'
    )
