import pytest

from pqlite.filter import Filter


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


def test_error_filter():
    f = Filter({'$may': {'brand': {'$lt': 1}, 'price': {'$gte': 50}}})
    with pytest.raises(ValueError):
        f.parse_where_clause()
