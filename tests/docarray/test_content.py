import numpy as np
import pytest
from docarray import DocumentArray
from docarray.array.annlite import AnnliteConfig, DocumentArrayAnnlite


@pytest.mark.parametrize(
    'content_attr', ['texts', 'embeddings', 'tensors', 'blobs', 'contents']
)
def test_content_empty_getter_return_none(content_attr):
    da = DocumentArrayAnnlite(config={'n_dim': 3})
    assert getattr(da, content_attr) is None


@pytest.mark.parametrize(
    'content_attr',
    [
        ('texts', ''),
        ('embeddings', np.array([])),
        ('tensors', np.array([])),
        ('blobs', []),
        ('contents', []),
    ],
)
def test_content_empty_setter(content_attr):
    da = DocumentArrayAnnlite(config={'n_dim': 3})

    setattr(da, content_attr[0], content_attr[1])
    assert getattr(da, content_attr[0]) is None


@pytest.mark.parametrize(
    'content_attr',
    [
        ('texts', ['s'] * 10),
        ('tensors', np.random.random([10, 2])),
        ('blobs', [b's'] * 10),
    ],
)
def test_content_getter_setter(content_attr):
    da = DocumentArrayAnnlite.empty(10, config=AnnliteConfig(n_dim=128))
    setattr(da, content_attr[0], content_attr[1])
    np.testing.assert_equal(da.contents, content_attr[1])
    da.contents = content_attr[1]
    np.testing.assert_equal(da.contents, content_attr[1])
    np.testing.assert_equal(getattr(da, content_attr[0]), content_attr[1])
    da.contents = None
    assert da.contents is None


@pytest.mark.parametrize('da_len', [0, 1, 2])
def test_content_empty(da_len):
    da = DocumentArrayAnnlite.empty(da_len, config=AnnliteConfig(n_dim=128))

    assert not da.contents
    assert not da.tensors
    if da_len == 0:
        assert not da.texts
        assert not da.blobs
    else:
        assert da.texts == [''] * da_len
        assert da.blobs == [b''] * da_len

    da.texts = ['hello'] * da_len
    if da_len == 0:
        assert not da.contents
    else:
        assert da.contents == ['hello'] * da_len
        assert da.texts == ['hello'] * da_len
        assert not da.tensors
        assert da.blobs == [b''] * da_len


@pytest.mark.parametrize('da_len', [0, 1, 2])
def test_embeddings_setter(da_len):
    da = DocumentArrayAnnlite.empty(da_len, config=AnnliteConfig(n_dim=5))

    da.embeddings = np.random.rand(da_len, 5)
    for doc in da:
        assert doc.embedding.shape == (5,)
