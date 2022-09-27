import operator

import numpy as np
import pytest
from docarray import Document, DocumentArray
from docarray.math import ndarray


@pytest.mark.parametrize('limit', [1, 5, 10])
@pytest.mark.parametrize(
    'query',
    [np.random.random(32), np.random.random((1, 32)), np.random.random((2, 32))],
)
def test_find(limit, query):
    embeddings = np.random.random((20, 32))
    da = DocumentArray(storage='annlite', config={'n_dim': 32})

    da.extend([Document(embedding=v) for v in embeddings])

    result = da.find(query, limit=limit)
    n_rows_query, n_dim = ndarray.get_array_rows(query)

    if n_rows_query == 1 and n_dim == 1:
        # we expect a result to be DocumentArray
        assert len(result) == limit
    elif n_rows_query == 1 and n_dim == 2:
        # we expect a result to be a list with 1 DocumentArray
        assert len(result) == 1
        assert len(result[0]) == limit
    else:
        # check for each row on the query a DocumentArray is returned
        assert len(result) == n_rows_query

    # check returned objects are sorted according to the storage backend metric
    # weaviate uses cosine similarity by default
    # annlite uses cosine distance by default
    if n_dim == 1:
        cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
        assert sorted(cosine_distances, reverse=False) == cosine_distances
    else:
        for da in result:
            cosine_distances = [t['cosine'].value for t in da[:, 'scores']]
            assert sorted(cosine_distances, reverse=False) == cosine_distances


numeric_operators_annlite = {
    '$gte': operator.ge,
    '$gt': operator.gt,
    '$lte': operator.le,
    '$lt': operator.lt,
    '$eq': operator.eq,
    '$neq': operator.ne,
}


@pytest.mark.parametrize(
    'storage,filter_gen,numeric_operators,operator',
    [
        *[
            tuple(
                [
                    'annlite',
                    lambda operator, threshold: {'price': {operator: threshold}},
                    numeric_operators_annlite,
                    operator,
                ]
            )
            for operator in numeric_operators_annlite.keys()
        ],
    ],
)
@pytest.mark.parametrize('columns', [[('price', 'int')], {'price': 'int'}])
def test_search_pre_filtering(
    storage, filter_gen, operator, numeric_operators, columns
):
    np.random.seed(0)
    n_dim = 128

    da = DocumentArray(storage=storage, config={'n_dim': n_dim, 'columns': columns})

    da.extend(
        [
            Document(id=f'r{i}', embedding=np.random.rand(n_dim), tags={'price': i})
            for i in range(50)
        ]
    )
    thresholds = [10, 20, 30]

    for threshold in thresholds:

        filter = filter_gen(operator, threshold)

        results = da.find(np.random.rand(n_dim), filter=filter)

        assert len(results) > 0

        assert all(
            [numeric_operators[operator](r.tags['price'], threshold) for r in results]
        )


@pytest.mark.parametrize(
    'storage,filter_gen,numeric_operators,operator',
    [
        *[
            tuple(
                [
                    'annlite',
                    lambda operator, threshold: {'price': {operator: threshold}},
                    numeric_operators_annlite,
                    operator,
                ]
            )
            for operator in numeric_operators_annlite.keys()
        ],
    ],
)
@pytest.mark.parametrize('columns', [[('price', 'float')], {'price': 'float'}])
def test_filtering(storage, filter_gen, operator, numeric_operators, columns):
    n_dim = 128

    da = DocumentArray(storage=storage, config={'n_dim': n_dim, 'columns': columns})

    da.extend([Document(id=f'r{i}', tags={'price': i}) for i in range(50)])
    thresholds = [10, 20, 30]

    for threshold in thresholds:

        filter = filter_gen(operator, threshold)
        results = da.find(filter=filter)

        assert len(results) > 0

        assert all(
            [numeric_operators[operator](r.tags['price'], threshold) for r in results]
        )


def test_find_subindex():
    n_dim = 3
    subindex_configs = {'@c': None}
    subindex_configs['@c'] = {'n_dim': 2}

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

    closest_docs = da.find(query=np.array([3, 3]), on='@c')

    b = closest_docs[0].embedding == [2, 2]
    if isinstance(b, bool):
        assert b
    else:
        assert b.all()
    for d in closest_docs:
        assert d.id.endswith('_0') or d.id.endswith('_1')


def test_find_subindex_multimodal():
    from docarray import dataclass
    from docarray.typing import Text

    @dataclass
    class MMDoc:
        my_text: Text
        my_other_text: Text
        my_third_text: Text

    n_dim = 3
    subindex_configs = {
        '@.[my_text, my_other_text]': {'n_dim': 2},
        '@.[my_third_text]': {'n_dim': 2},
    }

    da = DocumentArray(
        storage='annlite',
        config={'n_dim': 3, 'metric': 'Euclidean'},
        subindex_configs=subindex_configs,
    )

    num_docs = 3
    docs_to_add = DocumentArray(
        [
            Document(
                MMDoc(
                    my_text='hello', my_other_text='world', my_third_text='hello again'
                )
            )
            for _ in range(num_docs)
        ]
    )
    for i, d in enumerate(docs_to_add):
        d.id = str(i)
        d.embedding = i * np.ones(n_dim)
        d.my_text.id = str(i) + '_0'
        d.my_text.embedding = np.array([i, i])
        d.my_other_text.id = str(i) + '_1'
        d.my_other_text.embedding = np.array([i, i])
        d.my_third_text.id = str(i) + '_2'
        d.my_third_text.embedding = np.array([3 * i, 3 * i])

    with da:
        da.extend(docs_to_add)

    closest_docs = da.find(query=np.array([3, 3]), on='@.[my_text, my_other_text]')
    assert (closest_docs[0].embedding == np.array([2, 2])).all()
    for d in closest_docs:
        assert d.id.endswith('_0') or d.id.endswith('_1')

    closest_docs = da.find(query=np.array([3, 3]), on='@.[my_third_text]')
    assert (closest_docs[0].embedding == np.array([3, 3])).all()
    for d in closest_docs:
        assert d.id.endswith('_2')
