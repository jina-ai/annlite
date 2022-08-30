import pytest
from docarray import Document, DocumentArray


def test_add():
    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
        },
    )

    annlite_doc.extend(
        [
            Document(id='r0', embedding=[0, 0, 0]),
            Document(id='r1', embedding=[1, 1, 1]),
            Document(id='r2', embedding=[2, 2, 2]),
            Document(id='r3', embedding=[3, 3, 3]),
            Document(id='r4', embedding=[4, 4, 4]),
        ]
    )

    assert len(annlite_doc) == len(annlite_doc[:, 'embedding'])
    assert len(annlite_doc[:, 'embedding']) == 5


def test_add_conflict_id():
    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
        },
    )

    annlite_doc.extend(
        [
            Document(id='r0', embedding=[0, 0, 0]),
            Document(id='r1', embedding=[1, 1, 1]),
            Document(id='r2', embedding=[2, 2, 2]),
            Document(id='r3', embedding=[3, 3, 3]),
            Document(id='r4', embedding=[4, 4, 4]),
        ]
    )

    from sqlite3 import IntegrityError

    with pytest.raises(IntegrityError):
        annlite_doc.extend(
            [
                Document(id='r0', embedding=[0, 0, 0]),
                Document(id='r1', embedding=[1, 1, 1]),
            ]
        )
