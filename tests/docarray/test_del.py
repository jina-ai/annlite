import numpy as np
import pytest
from docarray import Document, DocumentArray


@pytest.fixture()
def docs():
    return DocumentArray([Document(id=f'{i}') for i in range(1, 10)])


@pytest.mark.parametrize('deleted_elements', [[0, 1], ['r0', 'r1']])
def test_delete_success(deleted_elements):
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
        del annlite_doc[deleted_elements]

    assert len(annlite_doc._offset2ids.ids) == 6
    assert len(annlite_doc[:, 'embedding']) == 6

    for id in expected_ids_after_del:
        assert id == annlite_doc[id].id


@pytest.mark.parametrize('expected_failed_deleted_elements', [['r2', 'r3']])
def test_delete_not_found(expected_failed_deleted_elements):
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
            for deleted_elmnts in expected_failed_deleted_elements:
                del annlite_doc[deleted_elmnts]


@pytest.mark.parametrize(
    'storage, config',
    [
        ('annlite', {'n_dim': 3, 'metric': 'Euclidean'}),
    ],
)
def test_del_subindex(storage, config):

    D = 3

    da = DocumentArray(
        storage='annlite',
        config={'n_dim': D, 'metric': 'Euclidean'},
        subindex_configs={'@c': {'n_dim': 2}},
    )

    with da:
        da.extend(
            [
                Document(
                    id=str(i),
                    embedding=i * np.ones(D),
                    chunks=[
                        Document(id=str(i) + '_0', embedding=np.array([i, i])),
                        Document(id=str(i) + '_1', embedding=np.array([i, i])),
                    ],
                )
                for i in range(10)
            ]
        )

    del da['0']
    assert len(da) == 9
    assert len(da._subindices['@c']) == 18

    del da[-2:]
    assert len(da) == 7
    assert len(da._subindices['@c']) == 14


def test_del_subindex_annlite_multimodal():
    from docarray import dataclass
    from docarray.typing import Text

    @dataclass
    class MMDoc:
        my_text: Text
        my_other_text: Text

    n_dim = 3
    da = DocumentArray(
        storage='annlite',
        config={'n_dim': n_dim, 'metric': 'Euclidean'},
        subindex_configs={'@.[my_text, my_other_text]': {'n_dim': 2}},
    )

    num_docs = 10
    docs_to_add = DocumentArray(
        [
            Document(MMDoc(my_text='hello', my_other_text='world'))
            for _ in range(num_docs)
        ]
    )
    for i, d in enumerate(docs_to_add):
        d.id = str(i)
        d.embedding = i * np.ones(n_dim)
        d.my_text.id = str(i) + '_0'
        d.my_text.embedding = [i, i]
        d.my_other_text.id = str(i) + '_1'
        d.my_other_text.embedding = [i, i]

    with da:
        da.extend(docs_to_add)

    del da['0']
    assert len(da) == 9
    assert len(da._subindices['@.[my_text, my_other_text]']) == 18
