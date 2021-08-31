import pytest

from pqlite.storage.table import Table


@pytest.fixture
def dummy_table():
    table = Table(name='dummy', in_memory=True)
    table.add_column('name', 'TEXT', is_key=True)
    table.add_column('price', 'FLOAT', is_key=True)
    table.add_column('category', 'TEXT')
    table.create_table()

    return table


@pytest.fixture
def sample_docs():
    return [
        {'_doc_id': '0', 'name': 'orange', 'price': 1.2, 'category': 'fruit'},
        {'_doc_id': '1', 'name': 'banana', 'price': 2, 'category': 'fruit'},
        {'_doc_id': '2', 'name': 'poly', 'price': 5.1, 'category': 'animal'},
        {'_doc_id': '3', 'name': 'bread'},
    ]


def test_create():
    table = Table(name='cell_table_0')
    table.add_column('x', 'float')
    table.create_table()
    assert table.existed()


def test_schema(dummy_table):
    schema = dummy_table.schema
    assert len(schema.split('\n')) == 6


def test_insert_query(dummy_table, sample_docs):
    dummy_table.insert(sample_docs)

    result = list(dummy_table.query([('category', '=', 'fruit'), ('price', '<', 3)]))
    assert len(result) == 2
    assert result[0]['name'] == 'orange'
    assert result[0]['_id'] == 0
