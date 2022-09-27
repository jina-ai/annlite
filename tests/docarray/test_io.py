import os
import uuid

import numpy as np
import pytest
from docarray import Document, DocumentArray
from docarray.array.annlite import AnnliteConfig, DocumentArrayAnnlite
from docarray.helper import random_identity

from tests import random_docs


@pytest.fixture
def docs():
    return random_docs(100)


@pytest.mark.slow
@pytest.mark.parametrize('method', ['json', 'binary'])
@pytest.mark.parametrize('encoding', ['utf-8', 'cp1252'])
def test_document_save_load(docs, method, encoding, tmpfile):

    da = DocumentArrayAnnlite(docs, config=AnnliteConfig(n_dim=10))
    # da.insert(2, Document(id='new'))
    da.save(tmpfile, file_format=method, encoding=encoding)

    del da

    da_r = DocumentArrayAnnlite.load(
        tmpfile, file_format=method, encoding=encoding, config=AnnliteConfig(n_dim=10)
    )

    # assert da_r[2].id == 'new'
    # assert len(da_r) == len(docs)
    # for d, d_r in zip(docs, da_r):
    #     assert d.id == d_r.id
    #     np.testing.assert_equal(d.embedding, d_r.embedding)
    #     assert d.content == d_r.content


@pytest.mark.parametrize('flatten_tags', [True, False])
def test_da_csv_write(docs, flatten_tags, tmpdir):
    tmpfile = os.path.join(tmpdir, 'test.csv')
    da = DocumentArrayAnnlite(docs, config=AnnliteConfig(n_dim=10))
    da.save_csv(tmpfile, flatten_tags)
    with open(tmpfile) as fp:
        assert len([v for v in fp]) == len(da) + 1


def test_from_ndarray():
    _da = DocumentArrayAnnlite.from_ndarray(
        np.random.random([10, 256]), config=AnnliteConfig(n_dim=256)
    )

    assert len(_da) == 10


def test_from_files():
    assert (
        len(
            DocumentArrayAnnlite.from_files(
                patterns='*.*',
                to_dataturi=True,
                size=1,
                config=AnnliteConfig(n_dim=256),
            )
        )
        == 1
    )


cur_dir = os.path.dirname(os.path.abspath(__file__))


def test_from_files_exclude():
    da1 = DocumentArray.from_files(f'{cur_dir}/*.*')
    has_init = False
    for s in da1[:, 'uri']:
        if s.endswith('__init__.py'):
            has_init = True
            break
    assert has_init
    da2 = DocumentArray.from_files('*.*', exclude_regex=r'__.*\.py')
    has_init = False
    for s in da2[:, 'uri']:
        if s.endswith('__init__.py'):
            has_init = True
            break
    assert not has_init


def test_from_ndjson():
    with open(os.path.join(cur_dir, 'docs.jsonlines')) as fp:
        _da = DocumentArrayAnnlite.from_ndjson(fp, config=AnnliteConfig(n_dim=256))
        assert len(_da) == 2


def test_from_to_pd_dataframe():
    df = DocumentArrayAnnlite.empty(2, config=AnnliteConfig(n_dim=3)).to_dataframe()
    assert (
        len(DocumentArrayAnnlite.from_dataframe(df, config=AnnliteConfig(n_dim=3))) == 2
    )

    # more complicated
    da = DocumentArrayAnnlite.empty(2, config=AnnliteConfig(n_dim=3))

    da[:, 'embedding'] = [[1, 2, 3], [4, 5, 6]]
    da[:, 'tensor'] = [[1, 2], [2, 1]]
    da[0, 'tags'] = {'hello': 'world'}
    df = da.to_dataframe()

    da2 = DocumentArrayAnnlite.from_dataframe(df, config=AnnliteConfig(n_dim=3))

    assert da2[0].tags == {'hello': 'world'}
    assert da2[1].tags == {}


def test_from_to_bytes():
    # simple
    assert (
        len(
            DocumentArrayAnnlite.load_binary(
                bytes(DocumentArrayAnnlite.empty(2, config=AnnliteConfig(n_dim=3)))
            )
        )
        == 2
    )

    da = DocumentArrayAnnlite.empty(2, config=AnnliteConfig(n_dim=3))

    da[:, 'embedding'] = [[1, 2, 3], [4, 5, 6]]
    da[:, 'tensor'] = [[1, 2], [2, 1]]
    da[0, 'tags'] = {'hello': 'world'}
    da2 = DocumentArrayAnnlite.load_binary(bytes(da))
    assert da2.tensors == [[1, 2], [2, 1]]
    import numpy as np

    np.testing.assert_array_equal(da2.embeddings, [[1, 2, 3], [4, 5, 6]])
    # assert da2.embeddings == [[1, 2, 3], [4, 5, 6]]
    assert da2[0].tags == {'hello': 'world'}
    assert da2[1].tags == {}


@pytest.mark.parametrize('show_progress', [True, False])
def test_push_pull_io(show_progress):
    da1 = DocumentArrayAnnlite.empty(10, config=AnnliteConfig(n_dim=256))

    da1[:, 'embedding'] = np.random.random([len(da1), 256])
    random_texts = [str(uuid.uuid1()) for _ in da1]
    da1[:, 'text'] = random_texts

    name = f'docarray_ci_{random_identity()}'

    da1.push(name, show_progress=show_progress)

    da2 = DocumentArrayAnnlite.pull(
        name, show_progress=show_progress, config=AnnliteConfig(n_dim=256)
    )

    assert len(da1) == len(da2) == 10
    assert da1.texts == da2.texts == random_texts

    all_names = DocumentArray.cloud_list()

    assert name in all_names

    DocumentArray.cloud_delete(name)

    all_names = DocumentArray.cloud_list()

    assert name not in all_names
