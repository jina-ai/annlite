import numpy as np
import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize('nrof_docs', [10, 100, 10_000, 10_100, 20_000, 20_100])
def test_success_get_bulk_data(nrof_docs):
    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
        },
    )

    with annlite_doc:
        annlite_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ]
        )

    assert len(annlite_doc[:, 'id']) == nrof_docs


def test_error_get_bulk_data_id_not_exist():
    nrof_docs = 10

    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
        },
    )

    with annlite_doc:
        annlite_doc.extend(
            [
                Document(id=f'r{i}', embedding=np.ones((3,)) * i)
                for i in range(nrof_docs)
            ]
        )

    with pytest.raises(KeyError) as e:
        annlite_doc[['r1', 'r11', 'r21'], 'id']
