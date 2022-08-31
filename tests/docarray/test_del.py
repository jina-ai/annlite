import pytest
from docarray import Document, DocumentArray


@pytest.mark.parametrize('deleted_elmnts', [[0, 1], ['r0', 'r1']])
def test_delete_success(deleted_elmnts):
    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
        },
    )

    with annlite_doc:
        annlite_doc.extend(
            [
                Document(id='r0', embedding=[0, 0, 0]),
                Document(id='r1', embedding=[1, 1, 1]),
                Document(id='r2', embedding=[2, 2, 2]),
                Document(id='r3', embedding=[3, 3, 3]),
                Document(id='r4', embedding=[4, 4, 4]),
                Document(id='r5', embedding=[5, 5, 5]),
                Document(id='r6', embedding=[6, 6, 6]),
                Document(id='r7', embedding=[7, 7, 7]),
            ]
        )

    expected_ids_after_del = ['r2', 'r3', 'r4', 'r5', 'r6', 'r7']

    with annlite_doc:
        del annlite_doc[deleted_elmnts]

    assert len(annlite_doc._offset2ids.ids) == 6
    assert len(annlite_doc[:, 'embedding']) == 6

    for id in expected_ids_after_del:
        assert id == annlite_doc[id].id


@pytest.mark.parametrize('expected_failed_deleted_elmnts', [['r2', 'r3']])
def test_delete_not_found(expected_failed_deleted_elmnts):
    annlite_doc = DocumentArray(
        storage='annlite',
        config={
            'n_dim': 3,
        },
    )
    with annlite_doc:
        annlite_doc.extend(
            [
                Document(id='r0', embedding=[0, 0, 0]),
                Document(id='r1', embedding=[1, 1, 1]),
            ]
        )

    with pytest.raises(ValueError):
        with annlite_doc:
            for deleted_elmnts in expected_failed_deleted_elmnts:
                del annlite_doc[deleted_elmnts]
