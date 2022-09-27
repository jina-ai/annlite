import numpy as np
import pytest
import scipy.sparse
from docarray import Document, DocumentArray
from docarray.array.annlite import AnnliteConfig, DocumentArrayAnnlite

from tests import random_docs

rand_array = np.random.random([10, 3])


@pytest.fixture()
def docs():
    rand_docs = random_docs(100)
    return rand_docs


@pytest.fixture()
def nested_docs():
    docs = [
        Document(id='r1', chunks=[Document(id='c1'), Document(id='c2')]),
        Document(id='r2', matches=[Document(id='m1'), Document(id='m2')]),
    ]
    return docs


def test_da_get_embeddings(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))

    da.extend(docs)
    np.testing.assert_almost_equal(da._get_attributes('embedding'), da.embeddings)
    np.testing.assert_almost_equal(da[:, 'embedding'], da.embeddings)


def test_embeddings_setter_da(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))

    da.extend(docs)
    emb = np.random.random((100, 10))
    da[:, 'embedding'] = emb
    np.testing.assert_almost_equal(da.embeddings, emb)

    for x, doc in zip(emb, da):
        np.testing.assert_almost_equal(x, doc.embedding)

    da[:, 'embedding'] = None
    if hasattr(da, 'flush'):
        da.flush()
    assert da.embeddings is None or not np.any(da.embeddings)


def test_embeddings_wrong_len(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))

    da.extend(docs)
    embeddings = np.ones((2, 10))

    with pytest.raises(ValueError):
        da.embeddings = embeddings


def test_tensors_getter_da(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))

    da.extend(docs)
    tensors = np.random.random((100, 10, 10))
    da.tensors = tensors
    assert len(da) == 100
    np.testing.assert_almost_equal(da.tensors, tensors)

    da.tensors = None
    assert da.tensors is None


def test_texts_getter_da(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))

    da.extend(docs)
    assert len(da.texts) == 100
    assert da.texts == da[:, 'text']
    texts = ['text' for _ in range(100)]
    da.texts = texts
    assert da.texts == texts

    for x, doc in zip(texts, da):
        assert x == doc.text

    da.texts = None
    if hasattr(da, 'flush'):
        da.flush()

    # unfortunately protobuf does not distinguish None and '' on string
    # so non-set str field in Pb is ''
    assert set(da.texts) == set([''])


def test_setter_by_sequences_in_selected_docs_da(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))
    da.extend(docs)
    da[[0, 1, 2], 'text'] = 'test'
    assert da[[0, 1, 2], 'text'] == ['test', 'test', 'test']

    da[[3, 4], 'text'] = ['test', 'test']
    assert da[[3, 4], 'text'] == ['test', 'test']

    da[[0], 'text'] = ['jina']
    assert da[[0], 'text'] == ['jina']

    da[[6], 'text'] = ['test']
    assert da[[6], 'text'] == ['test']

    # test that ID not present in da works
    da[[0], 'id'] = '999'
    assert ['999'] == da[[0], 'id']

    da[[0, 1], 'id'] = ['101', '102']
    assert ['101', '102'] == da[[0, 1], 'id']


def test_texts_wrong_len(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))
    da.extend(docs)
    texts = ['hello']

    with pytest.raises(ValueError):
        da.texts = texts


def test_tensors_wrong_len(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))
    da.extend(docs)
    tensors = np.ones((2, 10, 10))

    with pytest.raises(ValueError):
        da.tensors = tensors


def test_blobs_getter_setter(docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))
    da.extend(docs)
    with pytest.raises(ValueError):
        da.blobs = [b'cc', b'bb', b'aa', b'dd']

    da.blobs = [b'aa'] * len(da)
    assert da.blobs == [b'aa'] * len(da)

    da.blobs = None
    if hasattr(da, 'flush'):
        da.flush()

    # unfortunately protobuf does not distinguish None and '' on string
    # so non-set str field in Pb is ''
    assert set(da.blobs) == set([b''])


def test_ellipsis_getter(nested_docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))
    da.extend(nested_docs)
    flattened = da[...]
    assert len(flattened) == 6
    for d, doc_id in zip(flattened, ['c1', 'c2', 'r1', 'm1', 'm2', 'r2']):
        assert d.id == doc_id


def test_ellipsis_attribute_setter(nested_docs):
    da = DocumentArrayAnnlite(config=AnnliteConfig(n_dim=10))
    da.extend(nested_docs)
    da[..., 'text'] = 'new'
    assert all(d.text == 'new' for d in da[...])


def test_zero_embeddings():
    a = np.zeros([10, 6])
    da = DocumentArrayAnnlite.empty(10, config=AnnliteConfig(n_dim=6))

    # all zero, dense
    da[:, 'embedding'] = a
    np.testing.assert_almost_equal(da.embeddings, a)
    for d in da:
        assert d.embedding.shape == (6,)

    # all zero, sparse
    sp_a = scipy.sparse.coo_matrix(a)
    da[:, 'embedding'] = sp_a
    np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
    for d in da:
        # scipy sparse row-vector can only be a (1, m) not squeezible
        assert d.embedding.shape == (1, 6)

    # near zero, sparse
    a = np.random.random([10, 6])
    a[a > 0.1] = 0
    sp_a = scipy.sparse.coo_matrix(a)
    da[:, 'embedding'] = sp_a
    np.testing.assert_almost_equal(da.embeddings.todense(), sp_a.todense())
    for d in da:
        # scipy sparse row-vector can only be a (1, m) not squeezible
        assert d.embedding.shape == (1, 6)


def embeddings_eq(emb1, emb2):
    b = emb1 == emb2
    if isinstance(b, bool):
        return b
    else:
        return b.all()


def test_getset_subindex():

    n_dim = 3
    subindex_configs = {'@c': {'n_dim': 2}}

    da = DocumentArray(
        storage='annlite',
        config={'n_dim': 3, 'metric': 'Euclidean'},
        subindex_configs=subindex_configs,
    )

    with da:
        da.extend(
            [
                Document(
                    id=str(i),
                    embedding=i * np.ones(n_dim),
                    chunks=[
                        Document(id=str(i) + '_0', embedding=np.array([i, i])),
                        Document(id=str(i) + '_1', embedding=np.array([i, i])),
                    ],
                )
                for i in range(3)
            ]
        )
    with da:
        da[0] = Document(
            embedding=-1 * np.ones(n_dim),
            chunks=[
                Document(id='c_0', embedding=np.array([-1, -1])),
                Document(id='c_1', embedding=np.array([-2, -2])),
            ],
        )

    with da:
        da[1:] = [
            Document(
                embedding=-1 * np.ones(n_dim),
                chunks=[
                    Document(id='c_0' + str(i), embedding=np.array([-1, -1])),
                    Document(id='c_1' + str(i), embedding=np.array([-2, -2])),
                ],
            )
            for i in range(2)
        ]

    # test insert single doc
    assert embeddings_eq(da[0].embedding, -1 * np.ones(n_dim))
    assert embeddings_eq(da[0].chunks[0].embedding, [-1, -1])
    assert embeddings_eq(da[0].chunks[1].embedding, [-2, -2])

    assert embeddings_eq(da._subindices['@c']['c_0'].embedding, [-1, -1])
    assert embeddings_eq(da._subindices['@c']['c_1'].embedding, [-2, -2])

    # test insert slice of docs
    assert embeddings_eq(da[1].embedding, -1 * np.ones(n_dim))
    assert embeddings_eq(da[1].chunks[0].embedding, [-1, -1])
    assert embeddings_eq(da[1].chunks[1].embedding, [-2, -2])

    assert embeddings_eq(da._subindices['@c']['c_00'].embedding, [-1, -1])
    assert embeddings_eq(da._subindices['@c']['c_10'].embedding, [-2, -2])

    assert embeddings_eq(da[2].embedding, -1 * np.ones(n_dim))
    assert embeddings_eq(da[2].chunks[0].embedding, [-1, -1])
    assert embeddings_eq(da[2].chunks[1].embedding, [-2, -2])

    assert embeddings_eq(da._subindices['@c']['c_01'].embedding, [-1, -1])
    assert embeddings_eq(da._subindices['@c']['c_11'].embedding, [-2, -2])


def test_init_subindex():

    num_top_level_docs = 5
    num_chunks_per_doc = 3
    subindex_configs = {'@c': {'n_dim': 2}}

    da = DocumentArray(
        [
            Document(
                chunks=[Document(text=f'{i}{j}') for j in range(num_chunks_per_doc)]
            )
            for i in range(num_top_level_docs)
        ],
        storage='annlite',
        config={'n_dim': 3, 'metric': 'Euclidean'},
        subindex_configs=subindex_configs,
    )

    assert len(da['@c']) == num_top_level_docs * num_chunks_per_doc
    assert len(da._subindices['@c']) == num_top_level_docs * num_chunks_per_doc
    expected_texts = []
    for i in range(num_top_level_docs):
        for j in range(num_chunks_per_doc):
            expected_texts.append(f'{i}{j}')
    assert da['@c'].texts == expected_texts
    assert da._subindices['@c'].texts == expected_texts


def test_set_on_subindex():
    n_dim = 3
    subindex_configs = {'@c': {'n_dim': 2}}

    da = DocumentArray(
        [Document(chunks=[Document() for j in range(3)]) for i in range(5)],
        storage='annlite',
        config={'n_dim': 3, 'metric': 'Euclidean'},
        subindex_configs=subindex_configs,
    )

    embeddings_to_assign = np.random.random((5 * 3, 2))
    with da:
        da['@c'].embeddings = embeddings_to_assign
    assert (da['@c'].embeddings == embeddings_to_assign).all()
    assert (da._subindices['@c'].embeddings == embeddings_to_assign).all()

    with da:
        da['@c'].texts = ['hello' for _ in range(5 * 3)]
    assert da['@c'].texts == ['hello' for _ in range(5 * 3)]
    assert da._subindices['@c'].texts == ['hello' for _ in range(5 * 3)]

    matches = da.find(query=np.random.random(2), on='@c')
    assert matches
    assert len(matches[0].embedding) == 2
