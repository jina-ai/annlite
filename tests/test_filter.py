import random
import tempfile

import numpy as np
import pytest
from docarray import Document, DocumentArray

from annlite import AnnLite
from annlite.filter import Filter


def test_empty_filter():
    f = Filter()
    where_clause, parameters = f.parse_where_clause()
    assert where_clause == ''
    assert parameters == ()


def test_simple_filter():
    f = Filter({'brand': {'$lt': 1}})
    where_clause, parameters = f.parse_where_clause()
    assert where_clause == '(brand < ?)'
    assert parameters == (1,)


def test_logic_operator():
    f = Filter({'$and': {'brand': {'$lt': 1}, 'price': {'$gte': 50}}})
    where_clause, parameters = f.parse_where_clause()
    assert where_clause == '(brand < ?) AND (price >= ?)'
    assert parameters == (1, 50)

    f = Filter({'brand': {'$lt': 1}, 'price': {'$gte': 50}})
    where_clause, parameters = f.parse_where_clause()
    assert where_clause == '(brand < ?) AND (price >= ?)'
    assert parameters == (1, 50)

    f = Filter({'$or': {'brand': {'$lt': 1}, 'price': {'$gte': 50}}})
    where_clause, parameters = f.parse_where_clause()
    assert where_clause == '(brand < ?) OR (price >= ?)'
    assert parameters == (1, 50)


def test_membership_operator():
    f = Filter({'$and': {'brand': {'$in': ['Nike', 'Gucci']}, 'price': {'$gte': 50}}})
    where_clause, parameters = f.parse_where_clause()
    assert where_clause == '(brand IN(?, ?)) AND (price >= ?)'
    assert parameters == ('Nike', 'Gucci', 50)

    f = Filter({'$or': {'brand': {'$nin': ['Nike', 'Gucci']}, 'price': {'$gte': 50}}})
    where_clause, parameters = f.parse_where_clause()
    assert where_clause == '(brand NOT IN(?, ?)) OR (price >= ?)'
    assert parameters == ('Nike', 'Gucci', 50)


def test_cases():
    express = {
        '$and': {
            'price': {'$gte': 0, '$lte': 54},
            'rating': {'$gte': 1},
            'year': {'$gte': 2007, '$lte': 2010},
        }
    }
    f = Filter(express)
    where_clause, parameters = f.parse_where_clause()
    assert (
        where_clause
        == '(price >= ?) AND (price <= ?) AND (rating >= ?) AND (year >= ?) AND (year <= ?)'
    )
    assert parameters == (0, 54, 1, 2007, 2010)

    express = {
        '$and': {
            'price': {'$or': [{'price': {'$gte': 0}}, {'price': {'$lte': 54}}]},
            'rating': {'$gte': 1},
            'year': {'$gte': 2007, '$lte': 2010},
        }
    }
    f = Filter(express)

    where_clause, parameters = f.parse_where_clause()
    assert (
        where_clause
        == '((price >= ?) OR (price <= ?)) AND (rating >= ?) AND (year >= ?) AND (year <= ?)'
    )
    assert parameters == (0, 54, 1, 2007, 2010)

    express = {
        '$and': {
            '$or': [{'price': {'$gte': 0}}, {'price': {'$lte': 54}}],
            'rating': {'$gte': 1},
            'year': {'$gte': 2007, '$lte': 2010},
        }
    }
    f = Filter(express)

    where_clause, parameters = f.parse_where_clause()
    assert (
        where_clause
        == '((price >= ?) OR (price <= ?)) AND (rating >= ?) AND (year >= ?) AND (year <= ?)'
    )
    assert parameters == (0, 54, 1, 2007, 2010)


def test_error_filter():
    f = Filter({'$may': {'brand': {'$lt': 1}, 'price': {'$gte': 50}}})
    with pytest.raises(ValueError):
        f.parse_where_clause()


@pytest.mark.parametrize(
    'columns', [[('x', float)], [('x', 'float')], {'x': 'float'}, {'x': float}]
)
def test_filter_with_columns(tmpfile, columns):
    N = 100
    D = 2
    limit = 3

    index = AnnLite(D, columns=columns, data_path=tmpfile, include_metadata=True)
    X = np.random.random((N, D)).astype(np.float32)

    docs = DocumentArray(
        [
            Document(id=f'{i}', embedding=X[i], tags={'x': random.random()})
            for i in range(N)
        ]
    )
    index.index(docs)

    matches = index.filter(
        filter={'x': {'$lt': 0.5}}, limit=limit, include_metadata=True
    )
    assert len(matches) == limit
    for m in matches:
        assert m.tags['x'] < 0.5


@pytest.mark.parametrize('filterable_attrs', [{'x': 'float'}, {'x': float}])
def test_filter_with_dict(tmpfile, filterable_attrs):
    N = 100
    D = 2
    limit = 3

    index = AnnLite(
        D,
        filterable_attrs=filterable_attrs,
        data_path=tmpfile,
        include_metadata=True,
    )
    X = np.random.random((N, D)).astype(np.float32)

    docs = DocumentArray(
        [
            Document(id=f'{i}', embedding=X[i], tags={'x': random.random()})
            for i in range(N)
        ]
    )
    index.index(docs)

    matches = index.filter(
        filter={'x': {'$lt': 0.5}}, limit=limit, include_metadata=True
    )
    assert len(matches) == limit
    for m in matches:
        assert m.tags['x'] < 0.5


@pytest.mark.parametrize('limit', [1, 5])
@pytest.mark.parametrize('offset', [1, 5])
@pytest.mark.parametrize('order_by', ['x', 'y'])
@pytest.mark.parametrize('ascending', [True, False])
def test_filter_with_limit_offset(tmpfile, limit, offset, order_by, ascending):
    N = 100
    D = 2

    index = AnnLite(
        D,
        filterable_attrs={'x': 'float', 'y': 'float'},
        data_path=tmpfile,
        include_metadata=True,
    )
    X = np.random.random((N, D)).astype(np.float32)

    docs = DocumentArray(
        [
            Document(
                id=f'{i}',
                embedding=X[i],
                tags={'x': random.random(), 'y': random.random()},
            )
            for i in range(N)
        ]
    )
    index.index(docs)

    matches = index.filter(
        filter={'x': {'$lt': 0.5}},
        limit=limit,
        offset=offset,
        order_by=order_by,
        ascending=ascending,
        include_metadata=True,
    )
    assert len(matches) == limit

    for i in range(len(matches) - 1):
        m = matches[i]
        assert m.tags['x'] < 0.5
        if ascending:
            assert m.tags[order_by] <= matches[i + 1].tags[order_by]
        else:
            assert m.tags[order_by] >= matches[i + 1].tags[order_by]


@pytest.mark.parametrize('limit', [1, 5])
def test_filter_with_wrong_columns(tmpfile, limit):
    N = 100
    D = 128

    index = AnnLite(
        D,
        columns=[('price', float)],
        data_path=tmpfile,
    )
    X = np.random.random((N, D)).astype(np.float32)

    docs = DocumentArray(
        [
            Document(id=f'{i}', embedding=X[i], tags={'price': random.random()})
            for i in range(N)
        ]
    )

    index.index(docs)

    matches = index.filter(
        filter={'price': {'$lte': 50}},
        limit=limit,
        include_metadata=True,
    )

    assert len(matches) == limit

    import sqlite3

    with pytest.raises(sqlite3.OperationalError):
        matches = index.filter(
            filter={'price_': {'$lte': 50}},
            include_metadata=True,
        )

    matches = index.filter(
        filter={'price': {'$lte': 50}},
        limit=limit,
        include_metadata=True,
    )

    assert len(matches) == limit
