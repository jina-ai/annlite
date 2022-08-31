import pytest
from docarray import DocumentArray


def test_add(docs):
    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 4,
        },
    )

    annlite_doc.extend(docs)

    assert len(annlite_doc) == len(annlite_doc[:, 'embedding'])
    assert len(annlite_doc[:, 'embedding']) == 6


def test_add_conflict_id(docs, update_docs):
    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 4,
        },
    )

    annlite_doc.extend(docs)

    from sqlite3 import IntegrityError

    with pytest.raises(IntegrityError):
        annlite_doc.extend(update_docs)
